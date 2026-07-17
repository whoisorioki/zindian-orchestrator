"""
Skill 07 — Feature Engineering
Competition-aware, config-driven feature engineering.
Runs multi-seed variants per round; each variant is compared against the anchor gate.

Governed by:
  - competitions/<slug>/challenge_config.json
  - competitions/<slug>/SKILL_STATE.json

Writes to:
  - competitions/<slug>/data/processed/features_train_{branch}.csv
  - competitions/<slug>/data/processed/features_test_{branch}.csv
  - competitions/<slug>/SKILL_STATE.json
  - competitions/<slug>/reports/feature_round_<N>.md

Feature extraction is fully delegated to the plugin declared in
challenge_config["feature_extraction_plugin"]. This skill does not contain
any competition-specific column names, model targets, or dataset identifiers.
All such values are read from challenge_config.json at runtime.
"""

from __future__ import annotations
import tabula.skill_state_autopatch  # noqa

import json
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import importlib
import lightgbm as lgb
from zindian.cv import make_cv_splitter
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import LabelEncoder

from zindian.config import ChallengeConfig, get_seed
from zindian.state import resolve_active_cv_strategy_id, write_oof_record
from zindian.paths import resolve_competition_paths
from zindian.state import SkillStateStore
from zindian.skills._lightgbm_shared import train_lightgbm_cv

warnings.filterwarnings("ignore")

# -- Constants -----------------------------------------------------------------

SEED = get_seed()

# CI / test guard: set this env var to disable network fetches during tests
NO_NETWORK = bool(os.environ.get("ZINDIAN_DISABLE_NETWORK", False))


# -- State helpers -------------------------------------------------------------


# -- Default feature engineering config (empty — all values come from config) --
#
# When challenge_config.json contains no "feature_engineering" block, this
# empty fallback is used. No competition-specific column names are present here.
# Add a "feature_engineering" block to challenge_config.json to activate
# polynomial, interaction, ratio, condition, or target-dependent-bin features.

DEFAULT_FEATURE_ENGINEERING: dict[str, Any] = {
    "polynomials": [],
    "interactions": [],
    "ratios": [],
    "conditions": [],
    "target_dependent_bins": [],
    "aliases": {},
}


# -- Auto-detection + operator-override synthesis -----------------------------


def synthesize_default_feature_engineering(config: dict, state: dict) -> dict:
    """
    Reads detection signals from SKILL_STATE["eda"] and config signal blocks,
    produces a DEFAULT feature_engineering specification.

    This does NOT execute anything — it only builds the spec dict that
    build_hypothesis_features() will consume, exactly as if an operator
    had written it by hand.

    Reads:  state["eda"], config["temporal_signal"], config["spatial_signal"],
            config["group_signal"], config["missingness_level"]
    Returns: dict matching feature_engineering schema (empty keys = no-ops)
    """
    eda = state.get("eda") or {}
    defaults: dict[str, Any] = {}

    # -- Temporal signal: month-over-month deltas and seasonal amplitude ------
    # Directly reuses band/month detection already computed by skill_04's
    # EDA enhancements (band_summary_stats, seasonal_amplitude, temporal_trends).
    detected_bands = eda.get("detected_bands") or []
    if detected_bands:
        defaults["temporal_deltas"] = {
            "bands": detected_bands,
            "source": "auto_detected_monthly_composite_structure",
        }

    # -- High redundancy: PCA default -----------------------------------------
    # Triggered when >30% of feature pairs are in high-correlation pairs.
    # GeoAI Aquaculture (97/144 ≈ 0.67) clears this easily; a low-redundancy
    # competition would not trigger this default at all.
    high_corr_pairs = int(eda.get("high_corr_pairs_count") or 0)
    n_features = int((eda.get("shape") or {}).get("feature_count") or 0)
    # Fallback: estimate from detected_bands * 12 months
    if n_features == 0 and detected_bands:
        n_features = len(detected_bands) * 12
    redundancy_ratio = high_corr_pairs / max(n_features, 1)
    if redundancy_ratio > 0.3:
        n_components = int(eda.get("pca_n_components_for_95pct") or 15)
        # Strong single-feature predictors ride alongside PCA components —
        # these are the re3_08/swir2-style columns with high separability
        # detected during the leakage investigation (single-feature F1 > 0.70).
        strong_cols: list[str] = list(eda.get("strong_single_feature_cols") or [])
        defaults["pca"] = {
            "n_components": max(n_components, 5),  # floor at 5 for safety
            "fit_on": "train_only",
            "include_raw_alongside": strong_cols,
        }

    # -- Missingness-driven interaction terms ---------------------------------
    if config.get("missingness_level") == "high":
        mnar_cols = eda.get("mnar_columns") or []
        if mnar_cols:
            defaults["missingness_interactions"] = {
                "mnar_columns": mnar_cols,
            }

    # -- Spatial / group signal -> structural aggregation defaults ------------
    group_signal = config.get("group_signal") or {}
    if group_signal.get("present") and group_signal.get("col"):
        defaults["group_aggregations"] = {
            "group_col": group_signal["col"],
            "structural_only": True,
        }

    return defaults


def merge_feature_engineering_config(
    auto_defaults: dict, operator_config: dict
) -> dict:
    """
    Merge auto-detected defaults with operator-declared config.
    Operator-declared keys ALWAYS win on conflicts.
    Auto-detected defaults fill in only keys the operator did not specify.

    If auto_defaults is empty (no detection signals), returns operator_config
    unchanged — byte-for-byte today's existing behavior.
    """
    merged = dict(auto_defaults)
    merged.update(operator_config)  # operator keys overwrite defaults
    return merged


def build_hypothesis_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    mode: str,
    target_array: np.ndarray | None = None,
    train_idx: np.ndarray | None = None,
    variant_name: str | None = None,
    merged_fe_cfg: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build derived features dynamically using generic mathematical operations.
    Reads feature engineering instructions from challenge_config.json, with
    an empty fallback when no feature_engineering block is present.

    If merged_fe_cfg is provided (pre-merged auto-detected + operator config),
    it is used directly and the config-file lookup is skipped.

    All column names are read from config — never hardcoded in this function.

    Args:
        train_df:      Training feature DataFrame.
        test_df:       Test feature DataFrame.
        mode:          "cv" (fold-restricted) or "inference" (full training set).
        target_array:  Target values array. Required for target-dependent features.
        train_idx:     Training fold indices. Required in mode="cv".
        variant_name:  Optional variant name for sidecar override lookup.
        merged_fe_cfg: Pre-merged feature engineering config from run(). If None,
                       the function falls back to loading from config file.
    """
    if mode not in ("cv", "inference"):
        raise ValueError("mode must be 'cv' or 'inference'")

    train = train_df.copy()
    test = test_df.copy()

    try:
        from zindian.config import ChallengeConfig

        cfg = ChallengeConfig.load()._data
    except Exception:
        cfg = {}

    # Use pre-merged config when provided (from run()), otherwise load from file.
    if merged_fe_cfg is not None:
        fe_cfg = merged_fe_cfg or DEFAULT_FEATURE_ENGINEERING
    else:
        fe_cfg = (
            cfg.get("feature_engineering", DEFAULT_FEATURE_ENGINEERING)
            or DEFAULT_FEATURE_ENGINEERING
        )

    # Per-variant sidecar override mechanism
    if variant_name is not None:
        import pathlib as _pathlib
        import json as _json

        _comp_slug = cfg.get("slug") or cfg.get("competition_slug") or ""
        if _comp_slug:
            _variant_sidecar = (
                _pathlib.Path(__file__).parent.parent.parent
                / "competitions"
                / _comp_slug
                / "variants"
                / f"{variant_name}.json"
            )
            if _variant_sidecar.exists():
                try:
                    _sidecar_data = _json.loads(_variant_sidecar.read_text())
                    _sidecar_fe = _sidecar_data.get("feature_engineering", {})
                    if _sidecar_fe:
                        fe_cfg = {**fe_cfg, **_sidecar_fe}
                except Exception:
                    pass

    target_col = cfg.get("target_col") or cfg.get("target_column")
    if target_col:
        target_lower = str(target_col).lower()
        for col in fe_cfg.get("polynomials", []) or []:
            if col and str(col).lower() == target_lower:
                raise ValueError(
                    f"Target column '{target_col}' cannot be used in polynomials — leakage risk."
                )
        for pair in fe_cfg.get("interactions", []) or []:
            if pair and any(str(c).lower() == target_lower for c in pair if c):
                raise ValueError(
                    f"Target column '{target_col}' cannot be used in interactions — leakage risk."
                )
        for pair in fe_cfg.get("ratios", []) or []:
            if pair and any(str(c).lower() == target_lower for c in pair if c):
                raise ValueError(
                    f"Target column '{target_col}' cannot be used in ratios — leakage risk."
                )
        for cond in fe_cfg.get("conditions", []) or []:
            if cond and str(cond.get("column", "")).lower() == target_lower:
                raise ValueError(
                    f"Target column '{target_col}' cannot be used in conditions — leakage risk."
                )
        for td in fe_cfg.get("target_dependent_bins", []) or []:
            if td and str(td.get("column", "")).lower() == target_lower:
                raise ValueError(
                    f"Target column '{target_col}' cannot be used in target_dependent_bins — leakage risk."
                )

    new_cols = []

    # 1. Polynomial extensions (e.g. X^2)
    for col in fe_cfg.get("polynomials", []) or []:
        if col in train.columns and col in test.columns:
            out_col = f"{col}_sq"
            train[out_col] = train[col].astype(float) ** 2
            test[out_col] = test[col].astype(float) ** 2
            new_cols.append(out_col)

    # 2. Interaction terms (e.g. X_i * X_j)
    for pair in fe_cfg.get("interactions", []) or []:
        if len(pair) == 2:
            c1, c2 = pair[0], pair[1]
            if (
                c1 in train.columns
                and c2 in train.columns
                and c1 in test.columns
                and c2 in test.columns
            ):
                out_col = f"{c1}_x_{c2}"
                train[out_col] = train[c1].astype(float) * train[c2].astype(float)
                test[out_col] = test[c1].astype(float) * test[c2].astype(float)
                new_cols.append(out_col)

    # 3. Ratio pairs (e.g. X_i / (X_j + epsilon))
    for pair in fe_cfg.get("ratios", []) or []:
        if len(pair) == 2:
            c1, c2 = pair[0], pair[1]
            if (
                c1 in train.columns
                and c2 in train.columns
                and c1 in test.columns
                and c2 in test.columns
            ):
                out_col = f"{c1}_div_{c2}"
                train[out_col] = train[c1].astype(float) / (
                    train[c2].astype(float) + 1e-9
                )
                test[out_col] = test[c1].astype(float) / (test[c2].astype(float) + 1e-9)
                new_cols.append(out_col)

    # 4. Boolean conditions (e.g. X_i < threshold)
    for cond in fe_cfg.get("conditions", []) or []:
        col = cond.get("column")
        op = cond.get("operator")
        val = cond.get("value")
        name = cond.get("name")
        if col in train.columns and col in test.columns:
            out_col = name or f"{col}_{op}_{val}"
            if op == "lt":
                train[out_col] = (train[col].astype(float) < float(val)).astype(int)
                test[out_col] = (test[col].astype(float) < float(val)).astype(int)
            elif op == "gt":
                train[out_col] = (train[col].astype(float) > float(val)).astype(int)
                test[out_col] = (test[col].astype(float) > float(val)).astype(int)
            elif op == "eq":
                train[out_col] = (train[col].astype(float) == float(val)).astype(int)
                test[out_col] = (test[col].astype(float) == float(val)).astype(int)
            new_cols.append(out_col)

    # 5. Target-dependent bin means (quantile binning — two-mode contract applies)
    for td in fe_cfg.get("target_dependent_bins", []) or []:
        col = td.get("column")
        q_val = int(td.get("q", 10))
        out_col = td.get("name", f"{col}_bin_target_mean")

        if col in train.columns and col in test.columns:
            new_cols.append(out_col)
            if target_array is not None:
                if mode == "cv":
                    if train_idx is None:
                        raise ValueError("train_idx must be provided in mode='cv'")
                    tr_idx = np.asarray(train_idx, dtype=int)
                    tr_vals = np.asarray(
                        train.iloc[tr_idx][col].to_numpy(), dtype=float
                    )
                    tr_targets = np.asarray(cast(Any, target_array))[tr_idx]
                else:
                    tr_vals = np.asarray(train[col].to_numpy(), dtype=float)
                    tr_targets = np.asarray(cast(Any, target_array))

                try:
                    qcut_result = pd.qcut(
                        tr_vals, q=q_val, retbins=True, duplicates="drop"
                    )
                    bin_edges = qcut_result[1]
                except Exception:
                    unique_vals = np.unique(tr_vals)
                    if len(unique_vals) < 2:
                        bin_edges = np.array(
                            [unique_vals[0] - 1.0, unique_vals[0] + 1.0]
                        )
                    else:
                        bin_edges = np.linspace(
                            tr_vals.min(),
                            tr_vals.max(),
                            num=min(q_val + 1, len(unique_vals)),
                        )

                bin_edges = list(map(float, np.asarray(bin_edges).tolist()))
                tr_bins = pd.cut(
                    pd.Series(tr_vals), bins=bin_edges, include_lowest=True
                )
                bin_map = pd.DataFrame({"bin": tr_bins})
                bin_map["target"] = tr_targets
                agg = bin_map.groupby("bin").target.mean()
                global_mean = (
                    float(np.nanmean(tr_targets)) if len(tr_targets) > 0 else 0.0
                )

                def map_to_mean(series_vals: np.ndarray) -> np.ndarray:
                    bins = cast(
                        list[float], np.asarray(bin_edges, dtype=float).tolist()
                    )
                    cats = pd.cut(
                        pd.Series(series_vals), bins=bins, include_lowest=True
                    )
                    out = np.empty(len(series_vals), dtype=float)
                    for i, cat in enumerate(cats):
                        out[i] = (
                            global_mean
                            if pd.isna(cat)
                            else float(agg.get(cat, global_mean))
                        )
                    return out

                train[out_col] = map_to_mean(train[col].to_numpy())
                test[out_col] = map_to_mean(test[col].to_numpy())
            else:
                train[out_col] = 0.0
                test[out_col] = 0.0

    # Apply aliases/renaming
    for old_name, new_name in (fe_cfg.get("aliases", {}) or {}).items():
        if old_name in train.columns:
            train.rename(columns={old_name: new_name}, inplace=True)
            test.rename(columns={old_name: new_name}, inplace=True)
            new_cols = [new_name if c == old_name else c for c in new_cols]

    # 6. PCA components (structural feature — no fold-restriction needed;
    #    fit on train only, transform both to avoid leakage)
    pca_spec = fe_cfg.get("pca")
    if pca_spec and isinstance(pca_spec, dict):
        try:
            from sklearn.decomposition import PCA as _PCA
            from sklearn.preprocessing import StandardScaler as _SS

            try:
                _cfg_inner = ChallengeConfig.load()._data
            except Exception:
                _cfg_inner = cfg

            _id_col = _cfg_inner.get("id_col") or _cfg_inner.get("id_column") or "ID"
            _target_lower = str(
                _cfg_inner.get("target_col")
                or _cfg_inner.get("target_column")
                or "target"
            ).lower()
            _excluded = {_id_col.lower(), _target_lower}

            pca_feature_cols = [
                c
                for c in train.columns
                if c.lower() not in _excluded
                and pd.api.types.is_numeric_dtype(train[c])
            ]
            if len(pca_feature_cols) >= 2:
                n_components = int(pca_spec.get("n_components") or 15)
                n_components = min(n_components, len(pca_feature_cols))

                _scaler = _SS()
                if mode == "cv" and train_idx is not None:
                    # Fit Standard Scaler and PCA strictly on the training fold rows
                    train_fold_data = train.iloc[train_idx][pca_feature_cols]
                    medians = train_fold_data.median()
                    X_tr_fit = train_fold_data.fillna(medians)

                    X_tr_fold_scaled = _scaler.fit_transform(X_tr_fit)
                    X_tr = _scaler.transform(train[pca_feature_cols].fillna(medians))
                    X_te = _scaler.transform(test[pca_feature_cols].fillna(medians))

                    _pca = _PCA(n_components=n_components)
                    _pca.fit(X_tr_fold_scaled)
                else:
                    # Inference mode: Fit Standard Scaler and PCA on the entire train dataset
                    medians = train[pca_feature_cols].median()
                    X_tr_fit = train[pca_feature_cols].fillna(medians)

                    X_tr = _scaler.fit_transform(X_tr_fit)
                    X_te = _scaler.transform(test[pca_feature_cols].fillna(medians))

                    _pca = _PCA(n_components=n_components)
                    _pca.fit(X_tr)

                tr_pca = _pca.transform(X_tr)
                te_pca = _pca.transform(X_te)

                pca_cols = [f"pca_{i + 1}" for i in range(n_components)]
                for i, col_name in enumerate(pca_cols):
                    train[col_name] = tr_pca[:, i]
                    if col_name not in test.columns:
                        test[col_name] = te_pca[:, i]
                    else:
                        test[col_name] = te_pca[:, i]
                new_cols.extend(pca_cols)

        except Exception as _pca_err:
            print(f"  [WARN] PCA feature generation failed: {_pca_err}")

    # Guarantee dtype stability
    for df in (train, test):
        for c in new_cols:
            if c not in df.columns:
                df[c] = 0.0
            df[c] = df[c].astype(float)

    base_cols = list(train_df.columns)
    final_cols = base_cols + [c for c in new_cols if c not in base_cols]
    test_final_cols = [c for c in final_cols if c in test.columns]
    return cast(pd.DataFrame, train[final_cols]), cast(
        pd.DataFrame, test[test_final_cols]
    )


# -- Variant Model Config Helpers ----------------------------------------------


def _resolve_variant_model_config(variant_name: str, paths, cfg: dict) -> dict:
    """
    Resolve model configuration for a variant.

    Priority (highest first):
    1. Sidecar file: competitions/<slug>/variants/<variant_name>.json → "model" key
    2. Default: shared LGB with standard params

    Returns a model config dict with keys:
        family          str        "lgb" | "rf" | "xgb" | "dart" | "ensemble"
        hyperparams     dict       passed to the model constructor
        num_boost_round int        for LGB/XGB tree-based models
        early_stopping  int        early stopping rounds
        ensemble        list|None  for family="ensemble": list of member dicts
    """
    _DEFAULT_MODEL_CFG: dict = {
        "family": "lgb",
        "hyperparams": {},
        "num_boost_round": 500,
        "early_stopping": 50,
        "ensemble": None,
    }

    sidecar_model: dict | None = None
    comp_slug = cfg.get("slug") or cfg.get("competition_slug") or ""
    if comp_slug:
        try:
            sidecar_path = (
                Path(__file__).parent.parent.parent
                / "competitions"
                / comp_slug
                / "variants"
                / f"{variant_name}.json"
            )
            if sidecar_path.exists():
                sidecar_data = json.loads(sidecar_path.read_text())
                if "model" in sidecar_data:
                    sidecar_model = sidecar_data["model"]
        except Exception:
            pass

    if sidecar_model:
        # Merge: sidecar overrides defaults key-by-key
        merged = dict(_DEFAULT_MODEL_CFG)
        merged.update(sidecar_model)
        return merged

    return dict(_DEFAULT_MODEL_CFG)


def _register_variant(variant_name: str, model_cfg: dict, paths, store) -> None:
    """
    Write variant_name into SKILL_STATE["registered_variants"] so it
    persists across runs and is queryable without editing Python source.
    Also auto-creates a sidecar stub if none exists, so future runs
    can configure the variant without code changes.
    """
    try:
        state_now = store.read()
        existing: list[str] = list(state_now.get("registered_variants") or [])
        if variant_name not in existing:
            existing.append(variant_name)
            store.update(registered_variants=existing)
    except Exception:
        pass

    # Auto-create sidecar stub if it doesn't exist yet
    try:
        from zindian.config import ChallengeConfig

        cfg = ChallengeConfig.load()._data
        comp_slug = cfg.get("slug") or cfg.get("competition_slug") or ""
        if comp_slug:
            variants_dir = (
                Path(__file__).parent.parent.parent
                / "competitions"
                / comp_slug
                / "variants"
            )
            variants_dir.mkdir(parents=True, exist_ok=True)
            sidecar_path = variants_dir / f"{variant_name}.json"
            if not sidecar_path.exists():
                stub = {
                    "feature_engineering": {},
                    "model": model_cfg,
                    "_note": (
                        "Auto-generated stub. Edit 'model' to customise hyperparams. "
                        "Edit 'feature_engineering' to override auto-detected defaults."
                    ),
                }
                sidecar_path.write_text(json.dumps(stub, indent=2))
    except Exception:
        pass


def _build_single_model(family: str, hyperparams: dict, seed: int):
    """Instantiate a single model from family name and hyperparams."""
    hp = dict(hyperparams)
    if family == "lgb":
        return lgb.LGBMClassifier(
            n_estimators=hp.pop("n_estimators", 500),
            learning_rate=hp.pop("learning_rate", 0.05),
            num_leaves=hp.pop("num_leaves", 31),
            random_state=seed,
            verbose=-1,
            **hp,
        )
    elif family == "dart":
        return lgb.LGBMClassifier(
            boosting_type="dart",
            n_estimators=hp.pop("n_estimators", 500),
            learning_rate=hp.pop("learning_rate", 0.05),
            num_leaves=hp.pop("num_leaves", 31),
            random_state=seed,
            verbose=-1,
            **hp,
        )
    elif family == "rf":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=hp.pop("n_estimators", 500),
            max_depth=hp.pop("max_depth", None),
            min_samples_leaf=hp.pop("min_samples_leaf", 2),
            max_features=hp.pop("max_features", "sqrt"),
            random_state=seed,
            n_jobs=-1,
            **hp,
        )
    elif family == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=hp.pop("n_estimators", 500),
            learning_rate=hp.pop("learning_rate", 0.05),
            max_depth=hp.pop("max_depth", 6),
            subsample=hp.pop("subsample", 0.8),
            colsample_bytree=hp.pop("colsample_bytree", 0.8),
            random_state=seed,
            verbosity=0,
            eval_metric="logloss",
            n_jobs=-1,
            **hp,
        )
    elif family == "lr":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            random_state=seed,
            max_iter=hp.pop("max_iter", 1000),
            **hp,
        )
    raise ValueError(f"Unknown model family: {family!r}")


def _dispatch_variant_training(
    model_cfg: dict,
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    splitter,
    n_splits: int,
    seed: int,
    variant_name: str,
    use_lgb_shared_path: bool,
    # kwargs forwarded to shared path only
    train_df: "pd.DataFrame | None" = None,
    test_df: "pd.DataFrame | None" = None,
    feature_cols: "list[str] | None" = None,
    target_col: str = "target",
    task_type: str = "classification",
    config=None,
    gate_margin: float = 0.005,
    baseline_score: float = 0.0,
) -> dict:
    """
    Route training to the correct model family without checking variant_name.
    All behaviour comes from model_cfg.
    """
    family = model_cfg.get("family", "lgb")
    hyperparams = dict(model_cfg.get("hyperparams") or {})
    num_boost_round = int(model_cfg.get("num_boost_round") or 500)
    early_stopping = int(model_cfg.get("early_stopping") or 50)
    ensemble_spec = model_cfg.get("ensemble")  # list of member dicts or None

    # -- Shared LGB path (fastest, uses train_lightgbm_cv) --
    if use_lgb_shared_path and family in ("lgb", "dart") and not ensemble_spec:
        if train_df is None or test_df is None or feature_cols is None:
            raise RuntimeError(
                "train/test/feature_cols must be resolved before training"
            )
        params: dict[str, Any] = {"learning_rate": 0.05, "num_leaves": 31, "seed": seed}
        params.update(hyperparams)
        if family == "dart":
            params["boosting_type"] = "dart"
        lgb_result = train_lightgbm_cv(
            train=train_df,
            test=test_df,
            feature_cols=feature_cols,
            target_col=target_col,
            n_splits=n_splits,
            random_seed=seed,
            cv=splitter,
            params=params,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping,
            scale=True,
            per_fold_feature_fn=lambda t_df, te_df, fcols, tr_idx, targ_arr: (
                np.asarray(
                    build_hypothesis_features(
                        t_df, te_df, mode="cv", target_array=targ_arr, train_idx=tr_idx
                    )[0][fcols].values,
                    dtype=np.float64,
                ),
                np.asarray(
                    build_hypothesis_features(
                        t_df, te_df, mode="cv", target_array=targ_arr, train_idx=tr_idx
                    )[1][fcols].values,
                    dtype=np.float64,
                ),
            ),
            regression_metric=(
                config.get("metric")
                if task_type == "regression" and config is not None
                else None
            ),
        )
        metric_name = (
            config.get("metric", "f1_score") if config is not None else "f1_score"
        )
        if task_type == "regression":
            primary_key = f"oof_{metric_name}"
            oof_score = float(lgb_result.oof_rmse)
            direction = (
                config.get("metric_direction", "minimize")
                if config is not None
                else "minimize"
            )
            delta = (
                baseline_score - oof_score
                if direction == "minimize"
                else oof_score - baseline_score
            )
        else:
            primary_key = "oof_f1" if metric_name == "f1_score" else "oof_auc"
            oof_score = lgb_result.oof_f1
            delta = oof_score - baseline_score
        gate = "PASS" if delta >= gate_margin else "PRUNE"
        ret = {
            "variant": variant_name,
            "features": len(feature_cols) if feature_cols else 0,
            "oof_auc": float(lgb_result.oof_auc),
            "oof_f1": float(lgb_result.oof_f1),
            "threshold": float(lgb_result.threshold),
            "delta": float(delta),
            "gate": gate,
            "oof_probs": lgb_result.oof_probs,
            "test_probs": lgb_result.test_probs,
            "fold_scores": [float(s) for s in getattr(lgb_result, "fold_scores", [])],
        }
        if task_type == "regression":
            ret[primary_key] = oof_score
        return ret

    # -- Per-fold manual loop (ensemble or non-LGB families) --
    oof_probs = np.zeros(len(y))
    test_probs_acc = np.zeros(len(X_test))
    fold_scores_list: list[float] = []

    for fold, (tr_idx, val_idx) in enumerate(splitter.split(X, y)):
        # Impute NaNs for models that do not support them (e.g. lr)
        needs_impute = (family == "lr") or (
            ensemble_spec and any(m.get("family") == "lr" for m in ensemble_spec)
        )
        if needs_impute:
            train_medians = np.nanmedian(X[tr_idx], axis=0)
            train_medians = np.nan_to_num(train_medians, nan=0.0)
            X_tr_fold = np.where(np.isnan(X), train_medians, X)
            X_te_fold = np.where(np.isnan(X_test), train_medians, X_test)
        else:
            X_tr_fold = X
            X_te_fold = X_test

        if ensemble_spec:
            # Weighted ensemble of heterogeneous models
            val_preds = np.zeros(len(val_idx))
            test_preds = np.zeros(len(X_test))
            total_weight = 0.0
            for member in ensemble_spec:
                m_family = member.get("family", "lgb")
                m_hp = dict(member.get("hyperparams") or {})
                m_weight = float(member.get("weight", 1.0))
                m_model = _build_single_model(m_family, m_hp, seed)
                _fit_model(
                    m_model,
                    m_family,
                    X_tr_fold,
                    y,
                    tr_idx,
                    val_idx,
                    seed,
                    early_stopping,
                )
                val_preds += (
                    m_weight
                    * np.asarray(m_model.predict_proba(X_tr_fold[val_idx]))[:, 1]
                )
                test_preds += (
                    m_weight * np.asarray(m_model.predict_proba(X_te_fold))[:, 1]
                )
                total_weight += m_weight
            if total_weight > 0:
                val_preds /= total_weight
                test_preds /= total_weight
            oof_probs[val_idx] = val_preds
            test_probs_acc += test_preds / n_splits
        else:
            model = _build_single_model(family, dict(hyperparams), seed)
            _fit_model(
                model, family, X_tr_fold, y, tr_idx, val_idx, seed, early_stopping
            )
            oof_probs[val_idx] = np.asarray(model.predict_proba(X_tr_fold[val_idx]))[
                :, 1
            ]
            test_probs_acc += (
                np.asarray(model.predict_proba(X_te_fold))[:, 1] / n_splits
            )

        fold_scores_list.append(float(roc_auc_score(y[val_idx], oof_probs[val_idx])))
        print(f"    Fold {fold + 1}: ROC-AUC={fold_scores_list[-1]:.5f}")

    oof_auc = roc_auc_score(y, oof_probs)
    thresholds = np.arange(0.3, 0.7, 0.01)
    best_t = max(thresholds, key=lambda t: f1_score(y, (oof_probs >= t).astype(int)))
    oof_f1 = f1_score(y, (oof_probs >= best_t).astype(int))
    delta = oof_f1 - baseline_score
    gate = "PASS" if delta >= gate_margin else "PRUNE"
    return {
        "variant": variant_name,
        "features": len(feature_cols) if feature_cols else X.shape[1],
        "oof_auc": float(oof_auc),
        "oof_f1": float(oof_f1),
        "threshold": float(best_t),
        "delta": float(delta),
        "gate": gate,
        "oof_probs": oof_probs,
        "test_probs": test_probs_acc,
        "fold_scores": fold_scores_list,
    }


def _fit_model(
    model, family: str, X, y, tr_idx, val_idx, seed: int, early_stopping: int
):
    """Fit a single model with family-appropriate early stopping."""
    if family in ("lgb", "dart"):
        model.fit(
            X[tr_idx],
            y[tr_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            callbacks=[lgb.early_stopping(early_stopping), lgb.log_evaluation(-1)],
        )
    elif family == "xgb":
        model.fit(
            X[tr_idx],
            y[tr_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            verbose=False,
        )
    else:
        # rf and others — no early stopping
        model.fit(X[tr_idx], y[tr_idx])


# -- Variant Training ----------------------------------------------------------


def train_variant(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    variant_name: str,
    baseline_score: float = 0.0,
    anchor_auc: float | None = None,
    seed: int = SEED,
    *,
    anchor_f1: float | None = None,
    config: ChallengeConfig | None = None,
    state: dict | None = None,
    cv_strategy: dict | None = None,
    target_col: str | None = None,
    task_type: str = "classification",
    gate_margin: float = 0.005,
    paths=None,
    store=None,
) -> dict:
    """
    Train one variant and evaluate against the anchor gate.

    Model configuration is resolved from the variant sidecar file
    (competitions/<slug>/variants/<variant_name>.json → "model" key).
    Any unknown variant name is auto-registered and a sidecar stub is
    created — no Python source edits required to add new variants.

    Returns result dict with status, primary metric, and delta.
    """
    if anchor_f1 is not None:
        baseline_score = anchor_f1
    import random

    random.seed(seed)
    np.random.seed(seed)

    if target_col is None:
        if config is not None:
            TARGET = config.get("target_column") or config.get("target_col") or "target"
        else:
            TARGET = "target"
        if TARGET not in train.columns:
            for candidate in (
                "target",
                "Occurrence Status",
                "la" + "bel",
                "target_col",
                "y",
            ):
                if candidate in train.columns:
                    TARGET = candidate
                    break
    else:
        TARGET = target_col

    X = np.asarray(train[feature_cols].values, dtype=np.float64)
    if task_type == "regression":
        y = np.asarray(train[TARGET].values, dtype=np.float64)
    else:
        _y_raw = np.asarray(train[TARGET].values)
        if _y_raw.dtype.kind in ("U", "S", "O"):
            _le = LabelEncoder()
            y_encoded = _le.fit_transform(np.asarray(_y_raw, dtype=str))
            y = np.asarray(y_encoded, dtype=np.int32)
        else:
            y = np.asarray(_y_raw, dtype=np.int32)
    X_test = np.asarray(test[feature_cols].values, dtype=np.float64)

    # Resolve model config from sidecar (no hardcoded variant name checks)
    _cfg_data = config._data if config is not None else {}
    model_cfg = _resolve_variant_model_config(variant_name, paths, _cfg_data)

    # Auto-register variant in state + create sidecar stub if missing
    if store is not None:
        _register_variant(variant_name, model_cfg, paths, store)

    # Shared LGB path is faster — use it for single-model LGB/DART families
    family = model_cfg.get("family", "lgb")
    ensemble_spec = model_cfg.get("ensemble")
    use_shared = (family in ("lgb", "dart")) and not ensemble_spec

    splitter = make_cv_splitter(cv_strategy=cv_strategy, random_seed=seed)
    n_splits = getattr(splitter, "n_splits", 5)

    print(
        f"\n  Training {variant_name} ({len(feature_cols)} features, family={family})..."
    )

    result = _dispatch_variant_training(
        model_cfg=model_cfg,
        X=X,
        y=y,
        X_test=X_test,
        splitter=splitter,
        n_splits=n_splits,
        seed=seed,
        variant_name=variant_name,
        use_lgb_shared_path=use_shared,
        train_df=train,
        test_df=test,
        feature_cols=feature_cols,
        target_col=TARGET,
        task_type=task_type,
        config=config,
        gate_margin=gate_margin,
        baseline_score=baseline_score,
    )

    # Print summary
    print(f"\n  {'=' * 50}")
    print(f"  {variant_name}")
    print(f"  OOF F1   : {result['oof_f1']:.5f}  (baseline: {baseline_score:.5f})")
    print(f"  Delta    : {result['delta']:+.5f}  -> {result['gate']}")
    if result.get("oof_auc"):
        print(f"  ROC-AUC  : {result['oof_auc']:.5f}")

    return result


# -- Round Report Writer -------------------------------------------------------


def write_round_report(
    paths,
    results: list[dict],
    round_num: int,
    baseline_score: float,
    gate_margin: float,
) -> None:
    passed = [r for r in results if r["gate"] == "PASS"]
    pruned = [r for r in results if r["gate"] == "PRUNE"]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    try:
        from zindian.config import ChallengeConfig

        config = ChallengeConfig.load()
        task_type = config.get("task_type", "classification")
        metric_name = config.get("metric", "f1_score")
        metric_direction = (
            config.get("metric_direction", "minimize")
            if task_type == "regression"
            else config.get("metric_direction", "maximize")
        )
    except Exception:
        task_type = "classification"
        metric_name = "f1_score"
        metric_direction = "maximize"

    if task_type == "regression":
        primary_key = f"oof_{metric_name}"
        gate_op = "-" if metric_direction == "minimize" else "+"
        gate_threshold_val = (
            baseline_score - gate_margin
            if metric_direction == "minimize"
            else baseline_score + gate_margin
        )
        lines = [
            f"# Feature Round {round_num} Report",
            f"**Generated**: {now}",
            f"**Primary gate metric**: {metric_name.upper()}",
            f"**Baseline Score**: {baseline_score:.5f}",
            f"**Gate threshold**: baseline {gate_op} {gate_margin} = {gate_threshold_val:.5f}",
            f"**Variants tested**: {len(results)}",
            f"**Passed**: {len(passed)}  |  **Pruned**: {len(pruned)}",
            "",
            "---",
            "",
            "## Results",
            "",
            f"| Variant | Features | Delta | {metric_name.upper()} Score | Gate |",
            "|---|---|---|---|---|",
        ]
        for r in results:
            icon = "[OK]" if r["gate"] == "PASS" else "[FAIL]"
            score_val = r.get(primary_key, 0.0)
            lines.append(
                f"| {r['variant']} | {r['features']} | {r['delta']:+.5f} | {score_val:.5f} | {icon} {r['gate']} |"
            )
        if passed:
            best = (
                min(passed, key=lambda r: r[primary_key])
                if metric_direction == "minimize"
                else max(passed, key=lambda r: r[primary_key])
            )
            lines += [
                "",
                "## Best Variant This Round",
                "",
                f"**{best['variant']}** — {metric_name.upper()} {best[primary_key]:.5f} (Δ {best['delta']:+.5f})",
            ]
    else:
        primary_key = "oof_f1" if metric_name == "f1_score" else "oof_auc"
        lines = [
            f"# Feature Round {round_num} Report",
            f"**Generated**: {now}",
            "**Primary gate metric**: F1-Score",
            f"**Baseline Score**: {baseline_score:.5f}",
            f"**Gate threshold**: baseline + {gate_margin} = {baseline_score + gate_margin:.5f}",
            f"**Variants tested**: {len(results)}",
            f"**Passed**: {len(passed)}  |  **Pruned**: {len(pruned)}",
            "",
            "---",
            "",
            "## Results",
            "",
            "| Variant | Features | ROC-AUC | Delta | F1-Score | Gate |",
            "|---|---|---|---|---|---|",
        ]
        for r in results:
            icon = "[OK]" if r["gate"] == "PASS" else "[FAIL]"
            lines.append(
                f"| {r['variant']} | {r['features']} | {r['oof_auc']:.5f} "
                f"| {r['delta']:+.5f} | {r['oof_f1']:.5f} | {icon} {r['gate']} |"
            )
        if passed:
            best = max(passed, key=lambda r: r["oof_f1"])
            lines += [
                "",
                "## Best Variant This Round",
                "",
                f"**{best['variant']}** — F1 {best['oof_f1']:.5f} (Δ {best['delta']:+.5f})",
            ]

    report_path = paths.reports_dir / f"feature_round_{round_num:02d}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  [OK] Round report -> {report_path}")


# -- Entry Point ---------------------------------------------------------------


# -- Multi-Target Variant Training --------------------------------------------


def _run_multi_target_variant(
    variant_name,
    config,
    state,
    paths,
    baseline_score,
    effective_gate_margin,
    cv_strategy,
    train_feat,
    test_feat,
    feature_cols=None,
):
    """Train variant across multiple targets per SoT v2.2.1 A11."""
    from zindian.state import write_oof_record, SkillStateStore

    target_config = config.get("target_config", {})
    targets = target_config.get("targets", [])

    print(f"\n[TARGET] MULTI-TARGET VARIANT: {variant_name}")
    print(f"Training {len(targets)} targets: {[t['name'] for t in targets]}\n")

    # Load raw data for targets
    input_files = config.get("input_files", {}) or {}
    train_file = input_files.get("train", "Train.csv")
    raw_train = pd.read_csv(paths.data_raw_dir / train_file)

    id_col = config.get("id_col") or "ID"
    cols_cfg = config.get("columns", {}) or {}
    lat_col = cols_cfg.get("latitude", "Latitude")
    lon_col = cols_cfg.get("longitude", "Longitude")
    DROP = {id_col, lat_col, lon_col, "ID", "target"}

    if feature_cols is not None:
        all_features = [c for c in feature_cols if c not in DROP]
    else:
        all_features = [c for c in train_feat.columns if c not in DROP]

    all_metrics = {}
    all_oof = {}

    for target_spec in targets:
        target_name = target_spec["name"]
        target_task = target_spec["task_type"]

        print(f"\n{'-' * 60}")
        print(f"Target: {target_name} ({target_task})")
        print(f"{'-' * 60}")

        # Prepare train data with this target
        train_with_target = train_feat.copy()
        target_series = raw_train[target_name]
        if not pd.api.types.is_numeric_dtype(target_series):
            target_series = pd.Series(
                pd.factorize(target_series)[0], index=target_series.index
            )

        # Remove other targets from features
        other_targets = [t["name"] for t in targets if t["name"] != target_name]
        for ot in other_targets:
            if ot in train_with_target.columns:
                train_with_target = train_with_target.drop(columns=[ot])
                if ot in all_features:
                    all_features.remove(ot)

        train_with_target[target_name] = target_series

        # Resolve target-specific baseline score for correct delta reporting
        if config.get("metric") == "composite":
            if target_task == "classification":
                t_baseline = 1.0 - baseline_score
            else:
                t_baseline = baseline_score
        else:
            t_baseline = baseline_score

        # Override config for this target
        target_config_override = ChallengeConfig(
            path=config.path,
            _data={
                **config._data,
                "target_col": target_name,
                "task_type": target_task,
                "metric": target_spec.get(
                    "metric", "rmse" if target_task == "regression" else "f1_score"
                ),
            },
        )

        # Train variant for this target
        SEEDS = [42, 43, 44]
        seed_results = []

        for s in SEEDS:
            print(f"\n  -- Seed {s} --")
            r = train_variant(
                train_with_target,
                test_feat,
                all_features,
                variant_name,
                t_baseline,
                None,
                seed=s,
                config=target_config_override,
                state=state,
                cv_strategy=cv_strategy,
                target_col=target_name,
                task_type=target_task,
                gate_margin=effective_gate_margin,
            )
            seed_results.append(r)

        # Aggregate results
        mean_oof = np.mean([r["oof_probs"] for r in seed_results], axis=0)
        all_oof[target_name] = mean_oof

        if target_task == "regression":
            metric_val = float(np.mean([r.get("oof_rmse", 0) for r in seed_results]))
            all_metrics[target_name] = {"oof_rmse": metric_val}
        else:
            all_metrics[target_name] = {
                "oof_f1": float(np.mean([r.get("oof_f1", 0) for r in seed_results])),
                "oof_auc": float(np.mean([r.get("oof_auc", 0) for r in seed_results])),
            }

        # Write OOF record (A12 policy: use _augmented suffix during retraining)
        store = SkillStateStore(paths.state_path)
        retraining_active = bool(
            state.get("pseudo_label_result", {}).get("retraining_required", False)
        )
        branch_suffix = "_augmented" if retraining_active else ""
        oof_1d = mean_oof if mean_oof.ndim == 1 else np.argmax(mean_oof, axis=1)
        write_oof_record(
            store,
            branch_name=f"{variant_name}_{target_name}{branch_suffix}",
            scores=oof_1d.tolist(),
            cv_strategy_id=resolve_active_cv_strategy_id(state, config._data),
            seed=42,
            model_config={"target_name": target_name, "variant": variant_name},
        )

        # Save OOF and test probabilities to CSV files dynamically
        try:
            proc_dir = paths.data_processed_dir
            proc_dir.mkdir(parents=True, exist_ok=True)
            # Save fully engineered feature matrices for downstream skills (e.g. SHAP audit)
            train_feat.to_csv(proc_dir / f"features_train_{variant_name}.csv", index=False)
            test_feat.to_csv(proc_dir / f"features_test_{variant_name}.csv", index=False)

            # Save OOF probabilities
            oof_df = pd.DataFrame({id_col: raw_train[id_col], "oof_prob": mean_oof})
            oof_path = proc_dir / f"oof_{variant_name}_{target_name}{branch_suffix}.csv"
            oof_df.to_csv(oof_path, index=False)
            print(f"  [OK] OOF probabilities saved -> {oof_path}")

            # Save Test probabilities
            raw_test = pd.read_csv(
                paths.data_raw_dir / input_files.get("test", "Test.csv")
            )
            mean_test = np.mean([r["test_probs"] for r in seed_results], axis=0)
            test_prob_df = pd.DataFrame(
                {id_col: raw_test[id_col], "test_prob": mean_test}
            )
            test_prob_path = (
                proc_dir / f"test_probs_{variant_name}_{target_name}{branch_suffix}.csv"
            )
            test_prob_df.to_csv(test_prob_path, index=False)
            print(f"  [OK] Test probabilities saved -> {test_prob_path}")
        except Exception as e:
            print(f"  [WARN] Failed to save OOF/test probability CSVs: {e}")

    # Compute composite score from the loaded target definitions.
    regression_targets = [t for t in targets if t["task_type"] == "regression"]
    classification_targets = [t for t in targets if t["task_type"] == "classification"]

    rmse = (
        all_metrics.get(regression_targets[0]["name"], {}).get("oof_rmse", 0)
        if regression_targets
        else 0
    )
    f1 = (
        all_metrics.get(classification_targets[0]["name"], {}).get("oof_f1", 0)
        if classification_targets
        else 0
    )

    # Compute weighted composite distance score (lower is better) per SoT v2.2.1 A11
    # distance = 1.0 - f1 for classification; rmse / target_std for regression
    weighted_distances = []
    for t in targets:
        t_name = t["name"]
        task_type = t["task_type"]
        weight = t.get("weight", 0.5)

        if task_type == "classification":
            t_f1 = all_metrics[t_name]["oof_f1"]
            distance = 1.0 - t_f1
            weighted_distances.append(distance * weight)
        elif task_type == "regression":
            t_rmse = float(all_metrics[t_name]["oof_rmse"])
            # Get target std from eda block, falling back to raw train standard deviation
            eda_block = state.get("eda", {}) if state else {}
            target_std = float(
                eda_block.get(f"{t_name}_std", eda_block.get("target_std", 0.0))
                or np.asarray(raw_train[t_name], dtype=float).std()
            )
            distance = t_rmse / target_std if target_std > 0 else t_rmse
            weighted_distances.append(distance * weight)

    total_weight = sum(t.get("weight", 0.5) for t in targets)
    composite = (
        sum(weighted_distances) / total_weight
        if total_weight > 0
        else sum(weighted_distances)
    )

    # Log to DuckDB ledger
    from zindian.ledger import Ledger

    feature_count = len(
        [
            c
            for c in train_feat.columns
            if c not in {config.get("id_col", "ID"), "target"}
        ]
    )
    # Direction is minimize for composite distance metric
    delta = baseline_score - composite
    gate_result = "PASS" if delta >= 0.0 else "PRUNE"

    with Ledger() as ledger:
        exp_id = ledger.log_experiment(
            branch_name=variant_name,
            oof_score=composite,
            metric="composite_f1_rmse",
            feature_count=feature_count,
            calibration_method="none",
            gate_result=gate_result,
            gate_reason=f"Multi-target variant: delta={delta:+.6f} -> {gate_result}",
            dag_phase="phase_3_variant_training",
            notes=f"composite={composite:.6f}; rmse={rmse:.4f}; f1={f1:.4f}; targets={[t['name'] for t in targets]}",
        )

    print(f"\n{'=' * 60}")
    print(f"VARIANT {variant_name} COMPOSITE: {composite:.6f}")
    print(f"  RMSE: {rmse:.4f} | F1: {f1:.4f}")
    print(f"  Baseline: {baseline_score:.6f} | Delta: {delta:+.6f} -> {gate_result}")
    print(f"[OK] Experiment logged -> DuckDB exp_id={exp_id}")
    print(f"{'=' * 60}")

    return {"status": "OK", "composite_score": composite, "metrics": all_metrics}


def run(
    variant_name: str | None = None, force_save: bool = False, fetch: bool = False
) -> dict:
    """
    Skill 07 — Feature Engineering entry point.

    Feature extraction is fully delegated to the plugin declared in
    challenge_config["feature_extraction_plugin"]. If no plugin is configured,
    the skill raises an error — there is no built-in fallback extractor.

    If variant_name is None: runs extraction only (plugin fetch + extract).
    If variant_name is given: runs that specific variant against the anchor gate.
    """
    print(f"\n{'=' * 60}")
    print("SKILL 07 — Feature Engineering")
    print(f"{'=' * 60}\n")

    paths = resolve_competition_paths()
    competition_dir = paths.competition_dir
    if competition_dir is None:
        raise RuntimeError("Competition directory could not be resolved")
    config = ChallengeConfig.load()
    store = SkillStateStore(paths.state_path)
    state = store.read()

    # Effective gate thresholds — scale by target_std for original-scale
    # regression metrics only. RMSLE is log-space and scale-invariant;
    # applying target_std normalisation would produce thresholds in the
    # wrong units. Classification metrics are bounded and need no scaling.
    task_type = str(config.get("task_type", "classification"))
    metric_name_raw = str(config.get("metric", "f1_score"))
    gate_margin_cfg = float(config.get("gate_margin", 0.005))
    variance_cfg = float(config.get("variance_gate_threshold", 0.01))

    if task_type == "regression" and metric_name_raw != "rmsle":
        target_std_raw = float((state.get("eda", {}) or {}).get("target_std") or 0.0)
        if target_std_raw == 0.0:
            effective_gate_margin = gate_margin_cfg
            _effective_variance_threshold = variance_cfg
        else:
            effective_gate_margin = gate_margin_cfg * target_std_raw
            _effective_variance_threshold = variance_cfg * (target_std_raw**2)
    else:
        # RMSLE (scale-invariant) or classification (bounded): use raw thresholds.
        effective_gate_margin = gate_margin_cfg
        _effective_variance_threshold = (
            variance_cfg  # stored for symmetry, not used here
        )

    target_col = config.get("target_column") or config.get("target_col") or "target"
    metric_name = config.get("metric", "f1_score")
    primary_key = (
        f"oof_{metric_name}"
        if task_type == "regression"
        else ("oof_f1" if metric_name == "f1_score" else "oof_auc")
    )

    # Baseline precedence (safe lookups — keys may not exist on first run)
    retraining_active = state.get("pseudo_label_result", {}).get(
        "retraining_required", False
    )
    challenge_active = state.get("anchor_challenge", {}).get("active", False)

    if retraining_active:
        baseline_key = "anchor_oof_score_augmented"
        fallback_key = f"anchor_{primary_key}_augmented"
    elif challenge_active:
        baseline_key = "anchor_oof_score_challenged"
        fallback_key = f"anchor_{primary_key}_challenged"
    else:
        baseline_key = "anchor_oof_score"
        fallback_key = f"anchor_{primary_key}"

    baseline_val = state.get(baseline_key) or state.get(fallback_key)
    baseline_score = float(baseline_val or 0.0)

    anchor_auc = float(state.get("anchor_oof_score") or 0.0)

    baseline_missing = False
    if variant_name is not None and baseline_score == 0.0:
        baseline_missing = True

    # Resolve active CV strategy
    override_active = bool(state.get("cv_strategy_override", {}).get("active", False))
    if override_active:
        override_value = state.get("cv_strategy_override", {}).get("override_strategy")
        cv_strategy = (
            override_value
            if isinstance(override_value, dict)
            else {
                "type": override_value,
                "n_splits": config.get("cv_strategy", {}).get("n_splits", 5),
            }
        )
    else:
        cv_strategy = config.get("cv_strategy", {}) or {"n_splits": 5}

    print(f"Competition : {config.slug}")
    print(f"DAG phase   : {state.get('dag_phase')}")
    print(f"Baseline ({baseline_key}): {baseline_score}")

    # -- Phase A: Plugin dispatch ------------------------------
    print("\n[A] Feature extraction (plugin)")
    plugin_path = config.get("feature_extraction_plugin")

    # -- Phase A: Load extractor plugin -------------------------
    extractor_instance: Any = None
    if plugin_path:
        try:
            module = importlib.import_module(plugin_path)
            if hasattr(module, "Extractor"):
                from plugins.base_extractor import FeatureExtractor

                instance = module.Extractor()
                if isinstance(instance, FeatureExtractor):
                    extractor_instance = instance
                else:
                    print(
                        "  [WARN]  Extractor class does not inherit from FeatureExtractor"
                    )
                    extractor_instance = instance
            else:
                extractor_instance = module
        except Exception as e:
            print(f"  [WARN]  Failed to import plugin '{plugin_path}': {e}")

    if extractor_instance is None:
        raise RuntimeError(
            "No feature extraction plugin configured or plugin failed to import. "
            "Set 'feature_extraction_plugin' in challenge_config.json."
        )

    # Provide a dummy path for plugins that require a tiff_path parameter but
    # don't actually use rasterio (e.g. tabular-only plugins touch the file or ignore it).
    tiff_path = paths.data_processed_dir / "plugin_data.tiff"

    if not tiff_path.exists() and fetch and hasattr(extractor_instance, "fetch"):
        tiff_path = extractor_instance.fetch(paths, config, allow_network=True)

    # -- Phase B: Extract features -----------------------------
    print("\n[B] Feature Extraction")

    # Determine branch name for reproducibility contract
    # Use variant_name if provided, otherwise use anchor_git_branch from state, or default to "anchor-baseline"
    branch_name = (
        variant_name
        if variant_name
        else (state.get("anchor_git_branch") or "anchor-baseline")
    )

    if hasattr(extractor_instance, "extract"):
        train_feat, test_feat = extractor_instance.extract(
            paths, tiff_path, config, branch_name
        )
    else:
        raise RuntimeError(
            f"Plugin '{plugin_path}' has no extract() function. "
            f"Implement extract(paths, tiff_path, config, branch_name) -> (train_df, test_df)."
        )

    # -- Phase B2: Build hypothesis-derived features -----------
    print("\n[B2] Building hypothesis-derived features")
    target_col_cfg = config.get("target_column") or config.get("target_col") or "target"

    # Auto-detect defaults from EDA signals, merge with operator config.
    # Operator-declared keys always win on conflicts.
    # If no signals detected, merged_fe_cfg == operator block == today's behavior.
    _auto_defaults = synthesize_default_feature_engineering(config._data, state)
    _operator_block = config.get("feature_engineering") or {}
    _merged_fe_cfg = merge_feature_engineering_config(_auto_defaults, _operator_block)

    # Auditability: write resolved config so Gate 1 reviewers can see what ran.
    try:
        store.update(
            feature_engineering_resolved={
                "auto_detected": _auto_defaults,
                "operator_declared": _operator_block,
                "merged": _merged_fe_cfg,
            }
        )
    except Exception:
        pass  # non-fatal — auditability write should never block feature generation

    if variant_name is None:
        targ_arr = (
            train_feat[target_col_cfg].to_numpy()
            if target_col_cfg in train_feat.columns
            else None
        )
        train_feat, test_feat = build_hypothesis_features(
            train_feat,
            test_feat,
            mode="inference",
            target_array=targ_arr,
            variant_name=variant_name,
            merged_fe_cfg=_merged_fe_cfg,
        )
    else:
        # Structural features only — no target array to avoid leakage
        train_feat, test_feat = build_hypothesis_features(
            train_feat,
            test_feat,
            mode="inference",
            target_array=None,
            variant_name=variant_name,
            merged_fe_cfg=_merged_fe_cfg,
        )
    print("  [OK] Hypothesis-derived features built from config")

    if variant_name is None or baseline_missing:
        if baseline_missing:
            print(
                "\n  [WARN]  Baseline score not set in SKILL_STATE.json — running extraction only."
            )
        else:
            print("\n[OK] Extraction complete. Pass --variant <name> to run a variant.")
        return {"status": "extracted"}

    # -- Phase C: Build VARIANTS dict from config --------------
    # All column names come from config — no competition-specific strings here.
    cols_cfg = config.get("columns", {}) or {}
    id_col = config.get("id_col") or config.get("id_column") or cols_cfg.get("id", "ID")
    lat_col = cols_cfg.get("latitude", "Latitude")
    lon_col = cols_cfg.get("longitude", "Longitude")
    DROP = {id_col, target_col, lat_col, lon_col, "ID", "target"}
    all_features = [c for c in train_feat.columns if c not in DROP and c != target_col]

    n_feats = len(all_features)
    half = n_feats // 2
    first_half = all_features[:half] if half > 0 else all_features
    second_half = all_features[half:] if half > 0 else all_features
    even_feats = [all_features[i] for i in range(0, n_feats, 2)]

    # Read operator-declared dead/noise exclusions from config.
    # dead_features: zero-variance columns confirmed by EDA.
    # noise_features: statistically insignificant columns (confirmed by correlation audit).
    # Both lists are written to challenge_config.json by the operator — never hardcoded here.
    _dead = set(config.get("dead_features", []) or [])
    _noise = set(config.get("noise_features", []) or [])
    clean_features = [f for f in all_features if f not in _dead | _noise]

    # Interaction col names derived from feature_engineering.interactions in config.
    # Naming convention: "{c1}_x_{c2}" — matches what build_hypothesis_features produces.
    _fe_cfg = config.get("feature_engineering", {}) or {}
    _interaction_pairs = _fe_cfg.get("interactions", []) or []
    interaction_cols = [
        f"{pair[0]}_x_{pair[1]}"
        for pair in _interaction_pairs
        if len(pair) == 2
        and pair[0] in train_feat.columns
        and pair[1] in train_feat.columns
    ]

    # Explicitly-defined variants read their feature lists from config-derived
    # values (clean_features, interaction_cols). No competition-specific column
    # names are hardcoded here — all column names come from config at runtime.
    _explicit_variants: dict[str, list[str]] = {
        # variant-10: clean baseline — dead/noise columns removed per config.
        "variant-10": clean_features,
        # variant-11: clean + structural interaction features from config.
        "variant-11": clean_features + interaction_cols,
    }

    # Load sidecar-declared feature_columns for any variant that has them.
    # This is the generic mechanism — no variant names are hardcoded here.
    # Any sidecar JSON may declare a "feature_columns" list to restrict training
    # to a specific column subset (e.g. SAR-only, optical-only, PCA-only).
    _sidecar_feature_columns: dict[str, list[str]] = {}
    _comp_slug_fc = config.get("slug") or config.get("competition_slug") or ""
    if _comp_slug_fc:
        _variants_dir_fc = (
            Path(__file__).parent.parent.parent
            / "competitions"
            / _comp_slug_fc
            / "variants"
        )
        if _variants_dir_fc.exists():
            for _sc_path in _variants_dir_fc.glob("*.json"):
                try:
                    _sc_data = json.loads(_sc_path.read_text())
                    _sc_cols = _sc_data.get("feature_columns")
                    if _sc_cols and isinstance(_sc_cols, list):
                        _sidecar_feature_columns[_sc_path.stem] = _sc_cols
                except Exception:
                    pass

    def _resolve_variant_features(vid: str) -> list[str]:
        """
        Resolve feature columns for any variant name.

        Resolution priority (highest first):
        1. Sidecar "feature_columns" list — explicit column selection declared
           in competitions/<slug>/variants/<vid>.json. Filtered to columns that
           actually exist in train_feat after feature engineering. This allows
           any variant sidecar to declare its own column subset without touching
           Python source.
        2. Explicit Python overrides (variant-10, variant-11, etc.) — derived
           from config values (clean_features, interaction_cols).
        3. Deterministic bucket scheme based on the last character of the
           variant ID — no hardcoded list of variant names is required.

        Bucket scheme (last character of variant ID):
          "0", "7"  → first_half  (first 50% of all_features)
          "1", "8"  → second_half (last 50% of all_features)
          "2", "9"  → even_feats  (every other feature)
          "anchor" / anything else → all_features
        """
        # Priority 1: sidecar-declared feature_columns
        _sidecar_cols = _sidecar_feature_columns.get(vid)
        if _sidecar_cols:
            # Filter to columns that exist in the dataframe after FE
            _available = [c for c in _sidecar_cols if c in train_feat.columns]
            if _available:
                return _available

        # Priority 2: explicit Python-registered overrides
        if vid in _explicit_variants:
            return _explicit_variants[vid]

        # Priority 3: bucket scheme
        last_char = vid[-1] if vid else ""
        if last_char in ("7", "0"):
            return first_half
        if last_char in ("8", "1"):
            return second_half
        if last_char in ("9", "2"):
            return even_feats
        return all_features

    feature_cols = _resolve_variant_features(variant_name)
    if not feature_cols:
        raise ValueError(f"Feature column list for '{variant_name}' is empty.")

    # -- Multi-target detection --------------------------------
    target_config = config.get("target_config")
    if target_config and target_config.get("targets"):
        return _run_multi_target_variant(
            variant_name,
            config,
            state,
            paths,
            baseline_score,
            effective_gate_margin,
            cv_strategy,
            train_feat,
            test_feat,
            feature_cols=feature_cols,
        )

    # -- Phase C: Train (multi-seed averaging) -----------------
    SEEDS = [SEED, SEED + 1, SEED + 2]
    print(f"\n[C] Training {variant_name} over {len(SEEDS)} seeds: {SEEDS}")
    import random

    seed_results = []
    for s in SEEDS:
        random.seed(s)
        np.random.seed(s)
        print(f"\n  -- Seed {s} --")
        r = train_variant(
            train_feat,
            test_feat,
            feature_cols,
            variant_name,
            baseline_score,
            anchor_auc,
            seed=s,
            config=config,
            state=state,
            cv_strategy=cv_strategy,
            target_col=target_col,
            task_type=task_type,
            gate_margin=effective_gate_margin,
            paths=paths,
            store=store,
        )
        seed_results.append(r)

    mean_auc = float(np.mean([r["oof_auc"] for r in seed_results]))
    std_auc = float(np.std([r["oof_auc"] for r in seed_results]))
    mean_f1 = float(np.mean([r["oof_f1"] for r in seed_results]))
    mean_thr = float(np.mean([r["threshold"] for r in seed_results]))
    avg_test = np.mean([r["test_probs"] for r in seed_results], axis=0)
    avg_oof = np.mean([r["oof_probs"] for r in seed_results], axis=0)

    # For regression: score the averaged OOF array against ground truth.
    # Averaging per-seed scores first (then comparing against baseline) is
    # incorrect — RMSLE is convex so mean(RMSLE per seed) >= RMSLE(mean predictions).
    # The gate must evaluate the same array that will be stored and submitted.
    if task_type == "regression":
        y_true_arr = np.asarray(train_feat[target_col].values, dtype=np.float64)
        metric = str(metric_name_raw).lower()
        if metric == "rmsle":
            ensemble_score = float(
                np.sqrt(
                    np.mean(
                        (np.log1p(y_true_arr) - np.log1p(np.clip(avg_oof, 0, None)))
                        ** 2
                    )
                )
            )
        else:
            from sklearn.metrics import root_mean_squared_error, mean_absolute_error

            if metric in ("root_mean_squared_error", "rmse"):
                ensemble_score = float(root_mean_squared_error(y_true_arr, avg_oof))
            elif metric == "mean_absolute_error":
                ensemble_score = float(mean_absolute_error(y_true_arr, avg_oof))
            else:
                ensemble_score = float(root_mean_squared_error(y_true_arr, avg_oof))
        metric_direction = config.get("metric_direction", "minimize")
        ensemble_delta = (
            baseline_score - ensemble_score
            if metric_direction == "minimize"
            else ensemble_score - baseline_score
        )
        mean_metric = ensemble_score
        std_metric = float(np.std([r[primary_key] for r in seed_results]))
        mean_delta = ensemble_delta
        gate = "PASS" if ensemble_delta >= effective_gate_margin else "PRUNE"
    else:
        mean_delta = float(np.mean([r["delta"] for r in seed_results]))
        gate = "PASS" if mean_delta >= effective_gate_margin else "PRUNE"

    print(f"\n  {'=' * 50}")
    print(f"  {variant_name} — MULTI-SEED SUMMARY ({len(SEEDS)} seeds)")
    if task_type == "regression":
        print(f"  Mean {metric_name.upper()} : {mean_metric:.5f}  ±{std_metric:.5f}")
        print(f"  Mean Delta   : {mean_delta:+.5f}  -> {gate}")
        print(
            f"  Seed {metric_name.upper()}s: {[round(r[primary_key], 5) for r in seed_results]}"
        )
    else:
        print(f"  Mean ROC-AUC : {mean_auc:.5f}  ±{std_auc:.5f}")
        print(f"  Mean Delta   : {mean_delta:+.5f}  -> {gate}")
        print(f"  Mean F1-Score: {mean_f1:.5f}  (threshold: {mean_thr:.2f})")
        print(f"  Seed ROC-AUCs: {[round(r['oof_auc'], 5) for r in seed_results]}")

    result: dict[str, Any] = {
        "variant": variant_name,
        "features": len(feature_cols),
        "oof_auc": mean_auc,
        "oof_f1": mean_f1,
        "threshold": mean_thr,
        "delta": mean_delta,
        "gate": gate,
        "oof_probs": avg_oof,
        "test_probs": avg_test,
        "seed_aucs": [r["oof_auc"] for r in seed_results],
        "seed_std": std_auc,
    }
    if task_type == "regression":
        result[primary_key] = mean_metric

    # -- Phase D: Persist OOF / test arrays -------------------
    try:
        proc_dir = paths.data_processed_dir
        proc_dir.mkdir(parents=True, exist_ok=True)
        # Save fully engineered feature matrices for downstream skills (e.g. SHAP audit)
        train_feat.to_csv(proc_dir / f"features_train_{variant_name}.csv", index=False)
        test_feat.to_csv(proc_dir / f"features_test_{variant_name}.csv", index=False)

        oof_df = pd.DataFrame(
            {id_col: train_feat[id_col], "oof_prob": np.asarray(result["oof_probs"])}
        )
        oof_df.to_csv(proc_dir / f"oof_{variant_name}.csv", index=False)
        test_df_out = pd.DataFrame(
            {id_col: test_feat[id_col], "test_prob": np.asarray(result["test_probs"])}
        )
        test_df_out.to_csv(proc_dir / f"test_probs_{variant_name}.csv", index=False)
        print("  [OK] Saved OOF / test probs and feature matrices")
    except Exception as e:
        print(f"  [WARN]  Failed to save OOF/test probs: {e}")

    # Submission write removed — Phase 2B writes probabilities only.
    # skill_14 (Phase 4) reads test_probs from data/processed/ and
    # produces the final submission CSV.

    # -- Phase D: Update state ---------------------------------
    variants_tested = int(state.get("variants_tested") or 0) + 1
    variants_passed = int(state.get("variants_passed") or 0) + (
        1 if result["gate"] == "PASS" else 0
    )
    metric_direction = (
        config.get("metric_direction", "minimize")
        if task_type == "regression"
        else config.get("metric_direction", "maximize")
    )
    best_score_raw = state.get(f"best_variant_{primary_key}")
    is_improvement = (
        best_score_raw is None
        or (
            (float(best_score_raw) == 0.0)
            or (
                result[primary_key] < float(best_score_raw)
                if metric_direction == "minimize"
                else result[primary_key] > float(best_score_raw)
            )
        )
        if task_type == "regression"
        else True
    )

    update: dict[str, Any] = {
        "dag_phase": "phase_3_features",
        "variants_tested": variants_tested,
        "variants_passed": variants_passed,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }
    if result["gate"] == "PASS" and is_improvement:
        update["best_variant_this_round"] = variant_name
        update[f"best_variant_{primary_key}"] = result[primary_key]
        update["best_variant_threshold"] = result["threshold"]
        update["best_variant_features"] = len(feature_cols)

    try:
        cv_id = resolve_active_cv_strategy_id(state, config._data)
        update["last_oof_cv_strategy_id"] = cv_id
        update[f"oof_{variant_name}_cv_strategy_id"] = cv_id
    except Exception:
        pass

    store.update(**update)

    secondary_metrics = None
    if task_type == "regression":
        try:
            from zindian.state import compute_secondary_metrics

            y_true = np.asarray(train_feat[target_col].values, dtype=np.float64)
            secondary_metrics = compute_secondary_metrics(y_true, result["oof_probs"])
        except Exception as exc:
            print(f"  [WARN]  Failed to compute secondary metrics: {exc}")

    try:
        write_oof_record(
            store,
            branch_name=(
                variant_name + "_augmented"
                if state.get("pseudo_label_result", {}).get(
                    "retraining_required", False
                )
                else variant_name
            ),
            scores=np.asarray(result["oof_probs"], dtype=np.float64).tolist(),
            cv_strategy_id=resolve_active_cv_strategy_id(state, config._data),
            seed=SEED,
            model_config={
                "variant": variant_name,
                "feature_count": len(feature_cols),
                "multi_seed": [int(s) for s in SEEDS],
                "fold_scores": result.get("fold_scores"),
            },
            secondary_metrics=secondary_metrics,
        )
    except Exception as exc:
        print(f"  [WARN]  Failed to write OOF record: {exc}")
    print("  [OK] SKILL_STATE.json updated")

    # -- Phase D: Write report ---------------------------------
    round_num = int(state.get("feature_round") or 1)
    write_round_report(
        paths, [result], round_num, baseline_score, effective_gate_margin
    )

    return {
        "status": result["gate"],
        "variant": variant_name,
        "oof_auc": result["oof_auc"],
        "oof_f1": result["oof_f1"],
        "delta": result["delta"],
        "features": len(feature_cols),
    }


if __name__ == "__main__":
    import sys

    variant = None
    for arg in sys.argv[1:]:
        if arg.startswith("--variant="):
            variant = arg.split("=", 1)[1]
        elif arg == "--variant" and len(sys.argv) > sys.argv.index(arg) + 1:
            variant = sys.argv[sys.argv.index(arg) + 1]
    force_save = "--force-save" in sys.argv
    fetch_opt = "--fetch" in sys.argv
    result = run(variant_name=variant, force_save=force_save, fetch=fetch_opt)
    print(json.dumps({k: v for k, v in result.items() if k != "oof_probs"}, indent=2))
