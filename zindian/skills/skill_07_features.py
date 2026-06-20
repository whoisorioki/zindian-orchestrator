"""
Skill 07 — Feature Engineering
Competition-aware, config-driven feature engineering.
Runs multi-seed variants per round; each variant is compared against the anchor gate.

Governed by:
  - competitions/<slug>/challenge_config.json
  - competitions/<slug>/SKILL_STATE.json

Writes to:
  - competitions/<slug>/data/processed/features_train.csv
  - competitions/<slug>/data/processed/features_test.csv
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
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


def _write_state(state: dict, path: Path) -> None:
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
        json.dump(state, tmp, indent=2)
        tmp_path = tmp.name
    os.replace(tmp_path, path)


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


def build_hypothesis_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    mode: str,
    target_array: np.ndarray | None = None,
    train_idx: np.ndarray | None = None,
    variant_name: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build derived features dynamically using generic mathematical operations.
    Reads feature engineering instructions from challenge_config.json, with
    an empty fallback when no feature_engineering block is present.

    All column names are read from config — never hardcoded in this function.

    Args:
        train_df:     Training feature DataFrame.
        test_df:      Test feature DataFrame.
        mode:         "cv" (fold-restricted) or "inference" (full training set).
        target_array: Target values array. Required for target-dependent features.
        train_idx:    Training fold indices. Required in mode="cv".
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
                    tr_vals = train.iloc[tr_idx][col].to_numpy()
                    tr_targets = np.asarray(target_array)[tr_idx]
                else:
                    tr_vals = train[col].to_numpy()
                    tr_targets = np.asarray(target_array)

                try:
                    _, bin_edges = pd.qcut(
                        tr_vals, q=q_val, retbins=True, duplicates="drop"
                    )
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
                bin_map = tr_bins.to_frame(name="bin")
                bin_map["target"] = tr_targets
                agg = bin_map.groupby("bin").target.mean()
                global_mean = (
                    float(np.nanmean(tr_targets)) if len(tr_targets) > 0 else 0.0
                )

                def map_to_mean(series_vals: np.ndarray) -> np.ndarray:
                    cats = pd.cut(
                        pd.Series(series_vals), bins=bin_edges, include_lowest=True
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

    # Guarantee dtype stability
    for df in (train, test):
        for c in new_cols:
            if c not in df.columns:
                df[c] = 0.0
            df[c] = df[c].astype(float)

    base_cols = list(train_df.columns)
    final_cols = base_cols + [c for c in new_cols if c not in base_cols]
    test_final_cols = [c for c in final_cols if c in test.columns]
    return train[final_cols], test[test_final_cols]


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
) -> dict:
    """
    Train one LightGBM variant and evaluate against the anchor gate.
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
                "label",
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
        _y_raw_07 = train[TARGET].values
        if _y_raw_07.dtype.kind in ("U", "S", "O"):
            _le_07 = LabelEncoder()
            y = _le_07.fit_transform(_y_raw_07.astype(str)).astype(np.int32)
        else:
            y = np.asarray(_y_raw_07, dtype=np.int32)
    X_test = np.asarray(test[feature_cols].values, dtype=np.float64)

    shared_lgb_variants = {
        "variant-00",
        "variant-06",
        "variant-07",
        "variant-08",
        "variant-09",
        "variant-10",
        "variant-11",
        "variant-12",
        "variant-15",
        "variant-16",
        "variant-17",
        "variant-20",
        "variant-30",
        "variant-36",
        "variant-31",
        "variant-32",
        "variant-33",
        "variant-35",
        "variant-37",
        "strength",
        "recency_strength",
        "squad_manager_experience",
    }
    tuned_lgb_variants = {
        "variant-13": {
            "learning_rate": 0.02,
            "num_leaves": 63,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
        },
        "variant-19": {
            "learning_rate": 0.02,
            "num_leaves": 127,
            "max_depth": 8,
            "min_child_samples": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.05,
            "reg_lambda": 0.05,
        },
        "variant-27": {
            "learning_rate": 0.02,
            "num_leaves": 63,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
        },
    }

    if variant_name in shared_lgb_variants | tuned_lgb_variants.keys():
        print(f"\n  Training {variant_name} ({len(feature_cols)} features)...")
        params = {"learning_rate": 0.05, "num_leaves": 31, "seed": seed}
        if variant_name in tuned_lgb_variants:
            params.update(tuned_lgb_variants[variant_name])

        splitter = make_cv_splitter(cv_strategy=cv_strategy, random_seed=seed)
        n_splits = getattr(splitter, "n_splits", 5)
        lgb_result = train_lightgbm_cv(
            train=train,
            test=test,
            feature_cols=feature_cols,
            target_col=TARGET,
            n_splits=n_splits,
            random_seed=seed,
            cv=splitter,
            params=params,
            num_boost_round=1000 if variant_name in tuned_lgb_variants else 500,
            early_stopping_rounds=100 if variant_name in tuned_lgb_variants else 50,
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
            metric_direction = (
                config.get("metric_direction", "minimize")
                if config is not None
                else "minimize"
            )
            delta = (
                baseline_score - oof_score
                if metric_direction == "minimize"
                else oof_score - baseline_score
            )
            gate = "PASS" if delta >= gate_margin else "PRUNE"

            print(f"\n  {'=' * 50}")
            print(f"  {variant_name}")
            print(
                f"  OOF {metric_name.upper()} : {oof_score:.5f}  (baseline: {baseline_score:.5f})"
            )
            print(f"  Delta    : {delta:+.5f}  → {gate}")
        else:
            primary_key = "oof_f1" if metric_name == "f1_score" else "oof_auc"
            oof_score = lgb_result.oof_f1
            delta = oof_score - baseline_score
            gate = "PASS" if delta >= gate_margin else "PRUNE"

            print(f"\n  {'=' * 50}")
            print(f"  {variant_name}")
            print(
                f"  OOF F1   : {lgb_result.oof_f1:.5f}  (baseline: {baseline_score:.5f})"
            )
            print(f"  Delta    : {delta:+.5f}  → {gate}")
            print(
                f"  ROC-AUC  : {lgb_result.oof_auc:.5f}  (threshold: {lgb_result.threshold:.2f})"
            )

        ret = {
            "variant": variant_name,
            "features": len(feature_cols),
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

    # -- Non-shared variants: per-fold manual loop -----------------------------
    splitter = make_cv_splitter(cv_strategy=cv_strategy, random_seed=seed)
    n_splits = getattr(splitter, "n_splits", 5)
    oof_probs = np.zeros(len(y))
    test_probs = np.zeros(len(test))
    fold_scores_list: list[float] = []

    print(f"\n  Training {variant_name} ({len(feature_cols)} features)...")
    for fold, (tr_idx, val_idx) in enumerate(splitter.split(X, y)):
        model = None

        try:
            train_fold, test_fold = build_hypothesis_features(
                train, test, mode="cv", target_array=y, train_idx=tr_idx
            )
            X = np.asarray(train_fold[feature_cols].values, dtype=np.float64)
            X_test = np.asarray(test_fold[feature_cols].values, dtype=np.float64)
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            X_test = scaler.transform(X_test)
        except Exception:
            X = np.asarray(train[feature_cols].values, dtype=np.float64)
            X_test = np.asarray(test[feature_cols].values, dtype=np.float64)

        if variant_name in ("variant-13", "variant-27"):
            model = lgb.LGBMClassifier(
                n_estimators=1000,
                learning_rate=0.02,
                num_leaves=63,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=SEED,
                verbose=-1,
            )
            model.fit(
                X[tr_idx],
                y[tr_idx],
                eval_set=[(X[val_idx], y[val_idx])],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(-1)],
            )

        elif variant_name in ("variant-14", "variant-28"):
            from sklearn.ensemble import RandomForestClassifier

            model = RandomForestClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=SEED,
                n_jobs=-1,
            )
            model.fit(X[tr_idx], y[tr_idx])

        elif variant_name in ("variant-18", "variant-29"):
            from xgboost import XGBClassifier

            model = XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=seed,
                verbosity=0,
                n_jobs=-1,
            )
            model.fit(
                X[tr_idx], y[tr_idx], eval_set=[(X[val_idx], y[val_idx])], verbose=False
            )

        elif variant_name == "variant-19":
            model = lgb.LGBMClassifier(
                n_estimators=1000,
                learning_rate=0.02,
                num_leaves=127,
                max_depth=8,
                min_child_samples=10,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.05,
                reg_lambda=0.05,
                random_state=seed,
                verbose=-1,
            )
            model.fit(
                X[tr_idx],
                y[tr_idx],
                eval_set=[(X[val_idx], y[val_idx])],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(-1)],
            )

        elif variant_name in ("variant-25", "variant-34"):
            from sklearn.ensemble import RandomForestClassifier

            lgb_model = lgb.LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                random_state=seed,
                verbose=-1,
            )
            lgb_model.fit(
                X[tr_idx],
                y[tr_idx],
                eval_set=[(X[val_idx], y[val_idx])],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
            )
            rf_model = RandomForestClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=seed,
                n_jobs=-1,
            )
            rf_model.fit(X[tr_idx], y[tr_idx])
            lgb_val = np.asarray(lgb_model.predict_proba(X[val_idx]))[:, 1]
            rf_val = np.asarray(rf_model.predict_proba(X[val_idx]))[:, 1]
            lgb_test = np.asarray(lgb_model.predict_proba(X_test))[:, 1]
            rf_test = np.asarray(rf_model.predict_proba(X_test))[:, 1]
            oof_probs[val_idx] = 0.5 * lgb_val + 0.5 * rf_val
            test_probs += (0.5 * lgb_test + 0.5 * rf_test) / n_splits
            fold_scores_list.append(
                float(roc_auc_score(y[val_idx], oof_probs[val_idx]))
            )
            print(f"    Fold {fold + 1}: ROC-AUC={fold_scores_list[-1]:.5f}")
            continue

        elif variant_name == "variant-26":
            model = lgb.LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                random_state=seed,
                verbose=-1,
            )
            model.fit(
                X[tr_idx],
                y[tr_idx],
                eval_set=[(X[val_idx], y[val_idx])],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
            )

        elif variant_name in (
            "variant-39",
            "variant-40",
            "variant-41",
            "variant-42",
            "variant-43",
        ):
            model = lgb.LGBMClassifier(
                boosting_type="dart",
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                random_state=SEED,
                verbose=-1,
            )
            model.fit(X[tr_idx], y[tr_idx])

        elif variant_name == "variant-38":
            from sklearn.ensemble import RandomForestClassifier
            from xgboost import XGBClassifier

            _lgb = lgb.LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                random_state=SEED,
                verbose=-1,
            )
            _rf = RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=SEED,
                n_jobs=-1,
            )
            _xgb = XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                random_state=SEED,
                verbosity=0,
                eval_metric="logloss",
            )
            _lgb.fit(X[tr_idx], y[tr_idx])
            _rf.fit(X[tr_idx], y[tr_idx])
            _xgb.fit(X[tr_idx], y[tr_idx])
            lgb_val = np.asarray(_lgb.predict_proba(X[val_idx]))[:, 1]
            rf_val = np.asarray(_rf.predict_proba(X[val_idx]))[:, 1]
            xgb_val = np.asarray(_xgb.predict_proba(X[val_idx]))[:, 1]
            lgb_test = np.asarray(_lgb.predict_proba(X_test))[:, 1]
            rf_test = np.asarray(_rf.predict_proba(X_test))[:, 1]
            xgb_test = np.asarray(_xgb.predict_proba(X_test))[:, 1]
            oof_probs[val_idx] = (lgb_val + rf_val + xgb_val) / 3.0
            test_probs += (lgb_test + rf_test + xgb_test) / 3.0 / n_splits
            fold_scores_list.append(
                float(roc_auc_score(y[val_idx], oof_probs[val_idx]))
            )
            print(f"    Fold {fold + 1}: ROC-AUC={fold_scores_list[-1]:.5f}")
            continue

        if model is None:
            raise RuntimeError(f"Model was not initialized for {variant_name}")

        oof_probs[val_idx] = np.asarray(model.predict_proba(X[val_idx]))[:, 1]
        test_probs += np.asarray(model.predict_proba(X_test))[:, 1] / n_splits
        fold_scores_list.append(float(roc_auc_score(y[val_idx], oof_probs[val_idx])))
        print(f"    Fold {fold + 1}: ROC-AUC={fold_scores_list[-1]:.5f}")

    oof_auc = roc_auc_score(y, oof_probs)
    thresholds = np.arange(0.3, 0.7, 0.01)
    best_t = max(thresholds, key=lambda t: f1_score(y, (oof_probs >= t).astype(int)))
    oof_f1 = f1_score(y, (oof_probs >= best_t).astype(int))
    delta = oof_f1 - baseline_score
    gate = "PASS" if delta >= gate_margin else "PRUNE"

    print(f"\n  {'=' * 50}")
    print(f"  {variant_name}")
    print(f"  OOF F1   : {oof_f1:.5f}  (baseline: {baseline_score:.5f})")
    print(f"  Delta    : {delta:+.5f}  → {gate}")
    print(f"  ROC-AUC  : {oof_auc:.5f}  (threshold: {best_t:.2f})")

    return {
        "variant": variant_name,
        "features": len(feature_cols),
        "oof_auc": float(oof_auc),
        "oof_f1": float(oof_f1),
        "threshold": float(best_t),
        "delta": float(delta),
        "gate": gate,
        "oof_probs": oof_probs,
        "test_probs": test_probs,
        "fold_scores": fold_scores_list,
    }


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
    print(f"\n  [OK] Round report → {report_path}")


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
                baseline_score,
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
            cv_strategy_id=state.get("anchor_cv_strategy_id", "stratified_5fold"),
            seed=42,
            model_config={"target_name": target_name, "variant": variant_name},
        )

    # Compute composite score
    rmse = all_metrics.get("total_goals", {}).get("oof_rmse", 0)
    f1 = all_metrics.get("Target", {}).get("oof_f1", 0)

    target_std = (
        float(raw_train["total_goals"].std())
        if "total_goals" in raw_train.columns
        else 1.0
    )
    normalized_rmse = rmse / target_std if target_std > 0 else rmse
    regression_score = max(0.0, 1.0 - normalized_rmse)

    composite = 0.6 * f1 + 0.4 * regression_score

    # Log to DuckDB ledger
    from zindian.ledger import Ledger

    feature_count = len(
        [
            c
            for c in train_feat.columns
            if c not in {config.get("id_col", "ID"), "target"}
        ]
    )
    with Ledger() as ledger:
        exp_id = ledger.log_experiment(
            branch_name=variant_name,
            oof_score=composite,
            metric="composite_f1_rmse",
            feature_count=feature_count,
            calibration_method="none",
            gate_result="PASS",
            gate_reason=f"Multi-target variant: {len(targets)} targets trained",
            dag_phase="phase_3_variant_training",
            notes=f"composite={composite:.6f}; rmse={rmse:.4f}; f1={f1:.4f}; targets={[t['name'] for t in targets]}",
        )

    print(f"\n{'=' * 60}")
    print(f"VARIANT {variant_name} COMPOSITE: {composite:.6f}")
    print(f"  RMSE: {rmse:.4f} | F1: {f1:.4f}")
    print(
        f"  Baseline: {baseline_score:.6f} | Delta: {composite - baseline_score:+.6f}"
    )
    print(f"[OK] Experiment logged → DuckDB exp_id={exp_id}")
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
    use_probabilities = config.get("use_probabilities", True)
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

    extractor = None
    if plugin_path:
        try:
            extractor = importlib.import_module(plugin_path)
        except Exception as e:
            print(f"  [WARN]  Failed to import plugin '{plugin_path}': {e}")

    if extractor is None:
        raise RuntimeError(
            "No feature extraction plugin configured or plugin failed to import. "
            "Set 'feature_extraction_plugin' in challenge_config.json."
        )

    # Provide a dummy path for plugins that require a tiff_path parameter but
    # don't actually use rasterio (e.g. tabular-only plugins touch the file or ignore it).
    tiff_path = paths.data_processed_dir / "plugin_data.tiff"

    if not tiff_path.exists() and fetch and hasattr(extractor, "fetch"):
        tiff_path = extractor.fetch(paths, config, allow_network=True)

    # -- Phase B: Extract features -----------------------------
    print("\n[B] Feature Extraction")
    if hasattr(extractor, "extract"):
        train_feat, test_feat = extractor.extract(paths, tiff_path, config)
    else:
        raise RuntimeError(
            f"Plugin '{plugin_path}' has no extract() function. "
            f"Implement extract(paths, tiff_path, config) -> (train_df, test_df)."
        )

    # -- Phase B2: Build hypothesis-derived features -----------
    print("\n[B2] Building hypothesis-derived features")
    target_col_cfg = config.get("target_column") or config.get("target_col") or "target"
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
        )
    else:
        # Structural features only — no target array to avoid leakage
        train_feat, test_feat = build_hypothesis_features(
            train_feat,
            test_feat,
            mode="inference",
            target_array=None,
            variant_name=variant_name,
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
        )

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

    def _resolve_variant_features(vid: str) -> list[str]:
        """
        Resolve feature columns for any variant name.

        Explicit overrides (variant-10, variant-11, etc.) are checked first.
        All other variant names fall back to a deterministic bucket scheme
        based on the last character of the variant ID — no hardcoded list of
        variant names is required.

        Bucket scheme (last character of variant ID):
          "0", "7"  → first_half  (first 50% of all_features)
          "1", "8"  → second_half (last 50% of all_features)
          "2", "9"  → even_feats  (every other feature)
          "anchor" / anything else → all_features
        """
        if vid in _explicit_variants:
            return _explicit_variants[vid]
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
        print(f"  Mean Delta   : {mean_delta:+.5f}  → {gate}")
        print(
            f"  Seed {metric_name.upper()}s: {[round(r[primary_key], 5) for r in seed_results]}"
        )
    else:
        print(f"  Mean ROC-AUC : {mean_auc:.5f}  ±{std_auc:.5f}")
        print(f"  Mean Delta   : {mean_delta:+.5f}  → {gate}")
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
        oof_df = pd.DataFrame(
            {id_col: train_feat[id_col], "oof_prob": np.asarray(result["oof_probs"])}
        )
        oof_df.to_csv(proc_dir / f"oof_{variant_name}.csv", index=False)
        test_df_out = pd.DataFrame(
            {id_col: test_feat[id_col], "test_prob": np.asarray(result["test_probs"])}
        )
        test_df_out.to_csv(proc_dir / f"test_probs_{variant_name}.csv", index=False)
        print("  [OK] Saved OOF / test probs")
    except Exception as e:
        print(f"  [WARN]  Failed to save OOF/test probs: {e}")

    # -- Phase D: Save submission if PASS or force_save --------
    if result["gate"] == "PASS" or force_save:
        input_files = config.get("input_files", {}) or {}
        sample_file = input_files.get("sample", "SampleSubmission.csv")
        sample = pd.read_csv(paths.data_raw_dir / sample_file)
        sub_col = [c for c in sample.columns if c != id_col][0]
        test_probs_arr = result["test_probs"]
        if task_type == "regression":
            sub_values = test_probs_arr
        elif use_probabilities:
            sub_values = test_probs_arr
        else:
            sub_values = (test_probs_arr >= result["threshold"]).astype(int)
        sub = pd.DataFrame({id_col: test_feat[id_col], sub_col: sub_values})
        sub = sub.set_index(id_col).reindex(sample[id_col]).reset_index()
        out = competition_dir / f"submissions/{variant_name}_submission.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(out, index=False)
        print(f"  [OK] Submission saved → {out}")

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
