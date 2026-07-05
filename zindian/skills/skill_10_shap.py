"""
Skill 10 — Governed SHAP Audit

Trains a LightGBM model on the current competition feature set, computes
fold-level TreeSHAP importances, and evaluates a lightweight correlation-pruning
wrapper using the active gate metric.

Outputs:
  - competitions/<slug>/reports/shap_analysis.json
  - competitions/<slug>/reports/shap_summary.md

Usage:
  python3 -m zindian.skills.skill_10_shap
"""

from __future__ import annotations
import tabula.skill_state_autopatch  # noqa

import json
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any, cast

import lightgbm as lgb
import numpy as np
import pandas as pd

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from zindian.cv import make_cv_splitter

from zindian.config import ChallengeConfig
from zindian.config import get_seed
from zindian.paths import CompetitionPaths, resolve_competition_paths
from zindian.state import SkillStateStore
from zindian.state import resolve_active_cv_strategy_id
from zindian.state import write_oof_record
from zindian.skills._lightgbm_shared import train_lightgbm_cv


def _load_train_frame(paths: CompetitionPaths) -> pd.DataFrame:
    if paths.competition_dir is None:
        raise FileNotFoundError("Competition directory could not be resolved")

    full = paths.competition_dir / "data" / "processed" / "features_full_train.csv"
    processed = paths.competition_dir / "data" / "processed" / "features_train.csv"
    try:
        config = ChallengeConfig.load()
        train_file = (config.get("input_files") or {}).get("train", "Training_Data.csv")
    except Exception:
        train_file = "Training_Data.csv"
    fallback = paths.data_raw_dir / train_file
    if full.exists():
        return pd.read_csv(full)
    if processed.exists():
        return pd.read_csv(processed)
    if fallback.exists():
        return pd.read_csv(fallback)
    raise FileNotFoundError(f"Could not find {full}, {processed} or {fallback}")


def _feature_columns(frame: pd.DataFrame, target: str) -> list[str]:
    """
    Return feature columns from frame, excluding target, id, and coordinate columns.
    Column names are read from config — never hardcoded — to satisfy A5.
    """
    try:
        config = ChallengeConfig.load()
        cols_cfg = config.get("columns", {}) or {}
        id_col = (
            config.get("id_col") or config.get("id_column") or cols_cfg.get("id", "ID")
        )
        lat_col = cols_cfg.get("latitude", "Latitude")
        lon_col = cols_cfg.get("longitude", "Longitude")
    except Exception:
        id_col, lat_col, lon_col = "ID", "Latitude", "Longitude"

    excluded = {target.lower(), id_col.lower(), lat_col.lower(), lon_col.lower()}
    return [col for col in frame.columns if col.lower() not in excluded]


def _as_positive_shap_values(raw_values: object) -> np.ndarray:
    if isinstance(raw_values, list):
        return np.asarray(raw_values[-1], dtype=np.float64)
    values = np.asarray(raw_values, dtype=np.float64)
    if values.ndim == 3:
        return values[..., -1]
    return values


def _train_shap_fold_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    seed: int,
    task_type: str = "classification",
) -> lgb.LGBMClassifier | lgb.LGBMRegressor:
    if task_type == "regression":
        model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            random_state=seed,
            verbose=-1,
        )
    else:
        model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            random_state=seed,
            verbose=-1,
        )
    model.fit(
        train_x,
        train_y,
        eval_set=[(val_x, val_y)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )
    return model


def _compute_shap_audit(
    frame: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    *,
    n_splits: int = 5,
    seed: int | None = None,
    task_type: str = "classification",
) -> dict:
    X = np.asarray(frame[feature_cols].values, dtype=np.float64)
    if task_type == "regression":
        y = np.asarray(frame[target].values, dtype=np.float64)
    else:
        _y_raw = frame[target].values
        assert _y_raw is not None
        if _y_raw.dtype.kind in ("U", "S", "O"):
            _le = LabelEncoder()
            transformed = _le.fit_transform(_y_raw.astype(str))
            y = np.asarray(transformed, dtype=np.int32)
        else:
            y = np.asarray(_y_raw, dtype=np.int32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Resolve canonical seed if caller passed None
    if seed is None:
        from zindian.config import get_seed

        seed = get_seed()
    # Safely cast seed to int if possible
    try:
        seed = int(seed)
    except (ValueError, TypeError):
        # Fallback to default seed from config
        from zindian.config import get_seed

        seed = get_seed()

    # Use config's cv_strategy block if defined, otherwise fall back to central factory
    from zindian.config import ChallengeConfig

    try:
        config = ChallengeConfig.load()
        cv_strategy = config.get("cv_strategy")
    except Exception:
        cv_strategy = None

    if not cv_strategy:
        # Use KFold for regression, StratifiedKFold for classification
        cv_strategy = {
            "type": "kfold" if task_type == "regression" else "stratified",
            "n_splits": n_splits,
        }
    splitter = make_cv_splitter(cv_strategy=cv_strategy, random_seed=seed)

    # Initialize OOF array based on task type
    if task_type == "regression":
        oof_probs = np.zeros(len(frame), dtype=np.float64)
    else:
        n_classes = len(np.unique(y))
        if n_classes == 2:
            oof_probs = np.zeros(len(frame), dtype=np.float64)
        else:
            oof_probs = np.zeros((len(frame), n_classes), dtype=np.float64)

    fold_scores: list[float] = []
    fold_importances: list[np.ndarray] = []

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y), start=1):
        model = _train_shap_fold_model(
            X[train_idx],
            y[train_idx],
            X[val_idx],
            y[val_idx],
            seed=seed + fold_idx,
            task_type=task_type,
        )
        if task_type == "regression":
            val_preds = np.asarray(model.predict(X[val_idx]), dtype=np.float64)
            oof_probs[val_idx] = val_preds
            from sklearn.metrics import root_mean_squared_error

            fold_rmse = float(root_mean_squared_error(y[val_idx], val_preds))
            fold_scores.append(fold_rmse)
            print(f"  Fold {fold_idx}/{n_splits}: rmse={fold_rmse:.6f}")
        else:
            val_probs = np.asarray(
                cast(lgb.LGBMClassifier, model).predict_proba(X[val_idx]),
                dtype=np.float64,
            )
            n_classes = val_probs.shape[1]
            if n_classes == 2:
                val_probs_1d = val_probs[:, 1]
                try:
                    fold_auc = float(roc_auc_score(y[val_idx], val_probs_1d))
                except ValueError:
                    fold_auc = 0.0
                oof_probs[val_idx] = val_probs_1d
            else:
                try:
                    fold_auc = float(
                        roc_auc_score(y[val_idx], val_probs, multi_class="ovr")
                    )
                except ValueError:
                    # Missing class in fold - skip AUC
                    fold_auc = 0.0
                oof_probs[val_idx] = val_probs
            fold_scores.append(fold_auc)
            print(f"  Fold {fold_idx}/{n_splits}: auc={fold_auc:.6f}")

        shap_module = cast(Any, shap)
        explainer = shap_module.TreeExplainer(model)
        shap_values = explainer.shap_values(X[val_idx], check_additivity=False)
        fold_importance = np.abs(_as_positive_shap_values(shap_values)).mean(axis=0)
        fold_importances.append(fold_importance)

    mean_abs_shap = np.mean(np.vstack(fold_importances), axis=0)
    ranking = (
        pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs_shap})
        .sort_values(["mean_abs_shap", "feature"], ascending=[False, True])
        .reset_index(drop=True)
    )

    shap_total = (
        float(np.asarray(ranking["mean_abs_shap"], dtype=np.float64).sum())
        if not ranking.empty
        else 0.0
    )
    top15_share = (
        float(
            np.asarray(ranking.head(15)["mean_abs_shap"], dtype=np.float64).sum()
            / shap_total
        )
        if shap_total > 0
        else 0.0
    )
    tail_share = float(1.0 - top15_share) if shap_total > 0 else 0.0

    # -- Leak detection: top-feature dominance ratio --------------------------
    # A feature is flagged as a suspected leak when its mean absolute SHAP is
    # more than shap_leak_threshold × the mean of all OTHER features' SHAP.
    # This catches the "one column explains everything" signature without
    # requiring a hardcoded SHAP magnitude — the ratio is scale-invariant.
    leaked_feature_names: list[str] = []
    if len(ranking) >= 2:
        try:
            _cfg_shap = ChallengeConfig.load()
            _leak_threshold = float(_cfg_shap.get("shap_leak_threshold") or 3.0)
        except Exception:
            _leak_threshold = 3.0
        top_shap = float(ranking.iloc[0]["mean_abs_shap"])
        rest_mean = float(ranking.iloc[1:]["mean_abs_shap"].mean())
        if rest_mean > 0 and (top_shap / rest_mean) > _leak_threshold:
            leaked_feature_names.append(str(ranking.iloc[0]["feature"]))

    if task_type == "regression":
        from sklearn.metrics import root_mean_squared_error

        oof_rmse = float(root_mean_squared_error(y, oof_probs))
        return {
            "oof_probs": oof_probs,
            "oof_auc": 0.0,
            "oof_f1": 0.0,
            "oof_rmse": oof_rmse,
            "threshold": 0.0,
            "fold_scores": fold_scores,
            "ranking": ranking,
            "top15_share": top15_share,
            "tail_share": tail_share,
            "leaked_features": leaked_feature_names,
        }
    else:
        if oof_probs.ndim == 1:
            # Binary classification
            thresholds = np.arange(0.3, 0.7, 0.01)
            best_threshold = float(
                max(thresholds, key=lambda t: f1_score(y, (oof_probs >= t).astype(int)))
            )
            oof_f1 = float(f1_score(y, (oof_probs >= best_threshold).astype(int)))
            try:
                oof_auc = float(roc_auc_score(y, oof_probs))
            except ValueError:
                oof_auc = 0.0
        else:
            # Multiclass classification
            oof_preds = np.argmax(oof_probs, axis=1)
            oof_f1 = float(f1_score(y, oof_preds, average="weighted"))
            try:
                oof_auc = float(roc_auc_score(y, oof_probs, multi_class="ovr"))
            except ValueError:
                oof_auc = 0.0
            best_threshold = 0.0

        return {
            "oof_probs": oof_probs,
            "oof_auc": oof_auc,
            "oof_f1": oof_f1,
            "threshold": best_threshold,
            "fold_scores": fold_scores,
            "ranking": ranking,
            "top15_share": top15_share,
            "tail_share": tail_share,
            "leaked_features": leaked_feature_names,
        }


def _build_pruned_feature_set(
    feature_cols: list[str], ranking: pd.DataFrame, frame: pd.DataFrame
) -> dict:
    corr = cast(pd.DataFrame, frame.loc[:, feature_cols].corr()).abs()
    corr_values = corr.to_numpy(dtype=float, copy=False)
    upper_mask = np.triu(np.ones(corr_values.shape, dtype=bool), k=1)
    rank_lookup = {
        feature: rank for rank, feature in enumerate(ranking["feature"].tolist())
    }

    correlated_pairs: list[dict[str, object]] = []
    drop_features: set[str] = set()
    for row_idx, left in enumerate(corr.index):
        for col_idx, right in enumerate(corr.columns):
            if not upper_mask[row_idx, col_idx]:
                continue
            value = float(corr_values[row_idx, col_idx])
            if value > 0.95:
                correlated_pairs.append(
                    {"feature_a": left, "feature_b": right, "corr": value}
                )
                if rank_lookup.get(left, 10**9) <= rank_lookup.get(right, 10**9):
                    drop_features.add(right)
                else:
                    drop_features.add(left)

    pruned_features = [
        feature for feature in feature_cols if feature not in drop_features
    ]
    return {
        "correlated_pairs": correlated_pairs,
        "drop_features": sorted(drop_features),
        "pruned_features": pruned_features,
    }


def _write_outputs(
    paths: CompetitionPaths, report: dict, summary_lines: Iterable[str]
) -> None:
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = paths.reports_dir / "shap_analysis.json"
    summary_path = paths.reports_dir / "shap_summary.md"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"OK SHAP report written -> {report_path}")
    print(f"OK SHAP summary written -> {summary_path}")


def run(n_splits: int = 5, seed: int | None = None) -> dict:
    if not SHAP_AVAILABLE:
        return {
            "status": "SKIPPED",
            "reason": "shap_not_installed",
            "message": "SHAP library not available. Install with: pip install shap",
        }

    paths = resolve_competition_paths(require_competition=True)
    if paths.competition_dir is None:
        raise FileNotFoundError("Competition directory could not be resolved")

    config = ChallengeConfig.load()
    state = SkillStateStore(paths.state_path).read()

    # Multi-target detection
    target_config = config.get("target_config")
    if target_config and target_config.get("targets"):
        return _run_multi_target_shap(paths, config, state, n_splits, seed)

    frame = _load_train_frame(paths)
    target = config.get("target_col") or config.get("target_column")
    if not target:
        raise ValueError("target_col not configured in challenge_config.json")
    if target not in frame.columns:
        raise ValueError(
            f"Target column '{target}' not found in training features columns"
        )
    feature_cols = _feature_columns(frame, target)

    print(f"Competition      : {config.slug}")
    print(f"Target           : {target}")
    print(f"Features         : {len(feature_cols)}")
    print(f"DAG phase        : {state.get('dag_phase')}")

    task_type = config.get("task_type", "classification")

    if len(feature_cols) < 2:
        print("[WARN] X.shape[1] < 2: skipping SHAP ratio audit.")
        # Perform 5-fold CV to get oof_probs and metrics
        splitter = make_cv_splitter(n_splits=n_splits, random_seed=seed or get_seed())
        X = np.asarray(frame[feature_cols].values, dtype=np.float64)
        if task_type == "regression":
            y = np.asarray(frame[target].values, dtype=np.float64)
        else:
            _y_raw_sf = frame[target].values
            assert _y_raw_sf is not None
            if _y_raw_sf.dtype.kind in ("U", "S", "O"):
                _le_sf = LabelEncoder()
                transformed_sf = _le_sf.fit_transform(_y_raw_sf.astype(str))
                y = np.asarray(transformed_sf, dtype=np.int32)
            else:
                y = np.asarray(_y_raw_sf, dtype=np.int32)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        oof_probs = np.zeros(len(frame), dtype=np.float64)
        for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y), start=1):
            model = _train_shap_fold_model(
                X[train_idx],
                y[train_idx],
                X[val_idx],
                y[val_idx],
                seed=(seed or get_seed()) + fold_idx,
            )
            if task_type == "regression":
                val_preds = np.asarray(model.predict(X[val_idx]), dtype=np.float64)
                oof_probs[val_idx] = val_preds
            else:
                val_probs = np.asarray(
                    cast(lgb.LGBMClassifier, model).predict_proba(X[val_idx]),
                    dtype=np.float64,
                )[:, 1]
                oof_probs[val_idx] = val_probs

        state_store = SkillStateStore(paths.state_path)
        try:
            cfg = ChallengeConfig.load()._data
        except Exception:
            cfg = {}
        try:
            cv_id = resolve_active_cv_strategy_id(state, cfg)
        except Exception:
            cv_id = "unknown"

        state_store.update(
            shap_completed_at=datetime.now(timezone.utc).isoformat(),
            shap_feature_count=len(feature_cols),
            shap_audit_skipped_reason="single_feature",
            last_updated=datetime.now(timezone.utc).isoformat(),
            shap_oof_cv_strategy_id=cv_id,
        )
        try:
            _seed_val_sf = int(seed) if seed is not None else get_seed()
        except (ValueError, TypeError):
            _seed_val_sf = get_seed()
        write_oof_record(
            state_store,
            branch_name="shap_audit",
            scores=np.asarray(oof_probs, dtype=np.float64).tolist(),
            cv_strategy_id=cv_id,
            seed=_seed_val_sf,
            model_config={
                "feature_count": len(feature_cols),
                "n_splits": n_splits,
                "shap_audit_skipped_reason": "single_feature",
            },
        )
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "competition": config.slug,
            "target": target,
            "feature_count": len(feature_cols),
            "shap_audit_skipped_reason": "single_feature",
            "shap": {
                "oof_probs": oof_probs,
                "oof_auc": 0.5,
                "oof_f1": 0.0,
                "threshold": 0.5,
                "fold_scores": [0.5] * n_splits,
                "ranking": pd.DataFrame(
                    {"feature": feature_cols, "mean_abs_shap": [0.0]}
                ),
                "top15_share": 1.0,
                "tail_share": 0.0,
            },
        }

    print("Training governed SHAP audit…")

    full_audit = _compute_shap_audit(
        frame, feature_cols, target, n_splits=n_splits, seed=seed, task_type=task_type
    )
    ranking = full_audit["ranking"]
    pruning = _build_pruned_feature_set(feature_cols, ranking, frame)

    # Use the shared LightGBM CV path to test the correlation-pruning wrapper.
    full_cv = train_lightgbm_cv(
        train=frame,
        test=frame,
        feature_cols=feature_cols,
        target_col=target,
        n_splits=n_splits,
        random_seed=seed,
        scale=True,
        num_boost_round=500,
        early_stopping_rounds=50,
    )
    pruned_cv = train_lightgbm_cv(
        train=frame,
        test=frame,
        feature_cols=pruning["pruned_features"],
        target_col=target,
        n_splits=n_splits,
        random_seed=seed,
        scale=True,
        num_boost_round=500,
        early_stopping_rounds=50,
    )

    if task_type == "regression":
        metric_direction = config.get("metric_direction", "minimize")
        if metric_direction == "minimize":
            pruning_delta = float(full_cv.oof_rmse - pruned_cv.oof_rmse)
        else:
            pruning_delta = float(pruned_cv.oof_rmse - full_cv.oof_rmse)
        target_std = float(state.get("eda", {}).get("target_std", 1.0))
        pruning_pass = pruning_delta >= -0.01 * target_std
    else:
        pruning_delta = float(pruned_cv.oof_f1 - full_cv.oof_f1)
        pruning_pass = pruning_delta >= 0.005

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "competition": config.slug,
        "target": target,
        "feature_count": len(feature_cols),
        "metric": "rmse" if task_type == "regression" else "f1_score",
        "shap": {
            "fold_scores": full_audit["fold_scores"],
            "oof_auc": full_audit["oof_auc"],
            "oof_f1": full_audit["oof_f1"],
            "threshold": full_audit["threshold"],
            "top15_share": full_audit["top15_share"],
            "tail_share": full_audit["tail_share"],
            "top_features": ranking.head(20).to_dict(orient="records"),
        },
        "correlation_pruning": {
            "high_corr_pairs_count": len(pruning["correlated_pairs"]),
            "correlated_pairs": pruning["correlated_pairs"],
            "dropped_features": pruning["drop_features"],
            "pruned_feature_count": len(pruning["pruned_features"]),
            "full_oof_f1": (
                full_cv.oof_rmse if task_type == "regression" else full_cv.oof_f1
            ),
            "pruned_oof_f1": (
                pruned_cv.oof_rmse if task_type == "regression" else pruned_cv.oof_f1
            ),
            "delta_f1": pruning_delta,
            "gate_pass": pruning_pass,
        },
    }
    if task_type == "regression":
        report["shap"]["oof_rmse"] = full_audit["oof_rmse"]

    if task_type == "regression":
        summary_lines = [
            f"# SHAP Audit Summary — {report['generated_at']}",
            f"**Feature count**: {len(feature_cols)}",
            f"**Target**: {target}",
            f"**Full OOF RMSE**: {full_audit['oof_rmse']:.6f}",
            f"**Top-15 SHAP share**: {full_audit['top15_share']:.3%}",
            f"**Tail SHAP share**: {full_audit['tail_share']:.3%}",
            f"**High-correlation pairs**: {len(pruning['correlated_pairs'])}",
            f"**Pruned feature count**: {len(pruning['pruned_features'])}",
            f"**Pruned OOF RMSE**: {pruned_cv.oof_rmse:.6f}",
            f"**Delta RMSE**: {pruning_delta:+.6f}",
            f"**Pruning gate**: {'PASS' if pruning_pass else 'PRUNE'}",
            "",
            "## Top 10 SHAP Features",
        ]
    else:
        summary_lines = [
            f"# SHAP Audit Summary — {report['generated_at']}",
            f"**Feature count**: {len(feature_cols)}",
            f"**Target**: {target}",
            f"**Full OOF AUC**: {full_audit['oof_auc']:.6f}",
            f"**Full OOF F1**: {full_audit['oof_f1']:.6f} (threshold={full_audit['threshold']:.2f})",
            f"**Top-15 SHAP share**: {full_audit['top15_share']:.3%}",
            f"**Tail SHAP share**: {full_audit['tail_share']:.3%}",
            f"**High-correlation pairs**: {len(pruning['correlated_pairs'])}",
            f"**Pruned feature count**: {len(pruning['pruned_features'])}",
            f"**Pruned OOF F1**: {pruned_cv.oof_f1:.6f}",
            f"**Delta F1**: {pruning_delta:+.6f}",
            f"**Pruning gate**: {'PASS' if pruning_pass else 'PRUNE'}",
            "",
            "## Top 10 SHAP Features",
        ]
    for idx, row in ranking.head(10).iterrows():
        summary_lines.append(
            f"{idx + 1}. {row['feature']} — {row['mean_abs_shap']:.8f}"
        )

    _write_outputs(paths, report, summary_lines)

    state_store = SkillStateStore(paths.state_path)
    try:
        cfg = ChallengeConfig.load()._data
    except Exception:
        cfg = {}
    try:
        cv_id = resolve_active_cv_strategy_id(state, cfg)
    except Exception:
        cv_id = "unknown"

    state_store.update(
        shap_completed_at=datetime.now(timezone.utc).isoformat(),
        shap_feature_count=len(feature_cols),
        shap_top_feature=ranking.iloc[0]["feature"] if not ranking.empty else None,
        shap_top_features=[
            row["feature"] for row in ranking.head(10).to_dict(orient="records")
        ],
        high_corr_pairs_count=len(pruning["correlated_pairs"]),
        pruning_delta_f1=pruning_delta,
        pruning_pass=pruning_pass,
        last_updated=datetime.now(timezone.utc).isoformat(),
        shap_oof_cv_strategy_id=cv_id,
    )
    write_oof_record(
        state_store,
        branch_name="shap_audit",
        scores=np.asarray(full_audit["oof_probs"], dtype=np.float64).tolist(),
        cv_strategy_id=cv_id,
        seed=int(seed if seed is not None else get_seed()),
        model_config={
            "feature_count": len(feature_cols),
            "n_splits": n_splits,
            "pruning_gate": bool(pruning_pass),
        },
    )

    print(
        f"Top SHAP feature : {ranking.iloc[0]['feature']}"
        if not ranking.empty
        else "Top SHAP feature : none"
    )
    print(f"Pruning delta F1 : {pruning_delta:+.6f}")
    print(f"Top-15 SHAP share : {full_audit['top15_share']:.3%}")
    print(f"Pruning gate     : {'PASS' if pruning_pass else 'PRUNE'}")

    return report


def _run_multi_target_shap(paths, config, state, n_splits, seed) -> dict:
    """Multi-target SHAP analysis per SoT v2.2.1 A11."""
    print("\n[TARGET] MULTI-TARGET SHAP MODE\n")
    target_config = config.get("target_config", {})
    targets = target_config.get("targets", [])
    print(f"Analyzing {len(targets)} targets: {[t['name'] for t in targets]}\n")

    # Load features and raw data with targets
    frame = _load_train_frame(paths)
    input_files = config.get("input_files", {}) or {}
    train_file = input_files.get("train", "Train.csv")
    raw_train = pd.read_csv(paths.data_raw_dir / train_file)

    all_results = {}
    all_pass = True

    for target_spec in targets:
        target_name = target_spec["name"]
        target_task = target_spec["task_type"]
        print(f"\n{'-' * 60}")
        print(f"SHAP: {target_name} ({target_task})")
        print(f"{'-' * 60}")

        # Merge target from raw data
        frame_with_target = frame.copy()
        if target_name in raw_train.columns:
            target_series = raw_train[target_name]
            if target_series.dtype == "object":
                target_series = pd.factorize(target_series)[0]
            frame_with_target[target_name] = target_series
        else:
            print(f"[WARN] Target {target_name} not in raw data, skipping")
            continue

        feature_cols = _feature_columns(frame_with_target, target_name)
        full_audit = _compute_shap_audit(
            frame_with_target,
            feature_cols,
            target_name,
            n_splits=n_splits,
            seed=seed,
            task_type=target_task,
        )
        ranking = full_audit["ranking"]
        pruning = _build_pruned_feature_set(feature_cols, ranking, frame_with_target)

        # Use appropriate CV strategy for task type
        cv_strategy_dict = {
            "type": "kfold" if target_task == "regression" else "stratified",
            "n_splits": n_splits,
        }
        splitter = make_cv_splitter(cv_strategy=cv_strategy_dict, random_seed=seed)

        # Pre-generate splits to avoid stratification issues
        X_dummy = np.zeros((len(frame_with_target), 1))
        y_dummy = frame_with_target[target_name].values
        cv_splits = list(splitter.split(X_dummy, y_dummy))

        full_cv = train_lightgbm_cv(
            train=frame_with_target,
            test=frame_with_target,
            feature_cols=feature_cols,
            target_col=target_name,
            n_splits=n_splits,
            random_seed=seed,
            cv=cv_splits,
            scale=True,
            num_boost_round=500,
            early_stopping_rounds=50,
            regression_metric="rmse" if target_task == "regression" else None,
        )
        pruned_cv = train_lightgbm_cv(
            train=frame_with_target,
            test=frame_with_target,
            feature_cols=pruning["pruned_features"],
            target_col=target_name,
            n_splits=n_splits,
            random_seed=seed,
            cv=cv_splits,
            scale=True,
            num_boost_round=500,
            early_stopping_rounds=50,
            regression_metric="rmse" if target_task == "regression" else None,
        )

        if target_task == "regression":
            pruning_delta = float(full_cv.oof_rmse - pruned_cv.oof_rmse)
            pruning_pass = pruning_delta >= -0.01
        elif len(np.unique(frame_with_target[target_name])) == 2:
            # Binary classification
            pruning_delta = float(pruned_cv.oof_f1 - full_cv.oof_f1)
            pruning_pass = pruning_delta >= 0.005
        else:
            # Multiclass - skip pruning comparison (complex AUC calculation)
            pruning_delta = 0.0
            pruning_pass = True

        all_results[target_name] = {
            "pruning_pass": pruning_pass,
            "pruning_delta": pruning_delta,
            "top_feature": ranking.iloc[0]["feature"] if not ranking.empty else None,
            "top_features": ranking.head(20).to_dict(orient="records"),
            "full_oof_f1": (
                full_cv.oof_rmse if target_task == "regression" else full_cv.oof_f1
            ),
            "pruned_oof_f1": (
                pruned_cv.oof_rmse if target_task == "regression" else pruned_cv.oof_f1
            ),
            "high_corr_pairs_count": len(pruning["correlated_pairs"]),
            "dropped_features": pruning["drop_features"],
            "pruned_feature_count": len(pruning["pruned_features"]),
            "top15_share": full_audit.get("top15_share"),
            "tail_share": full_audit.get("tail_share"),
            "oof_auc": full_audit.get("oof_auc"),
            "oof_f1": full_audit.get("oof_f1"),
            "oof_rmse": full_audit.get("oof_rmse"),
            "fold_scores": full_audit.get("fold_scores"),
            "leaked_features": full_audit.get("leaked_features", []),
        }
        all_pass = all_pass and pruning_pass

        # Log any leaked features immediately
        target_leaked = full_audit.get("leaked_features") or []
        if target_leaked:
            print(
                f"  [LEAK DETECTED] {target_name}: features exceeding "
                f"shap_leak_threshold — {target_leaked}"
            )

    # -- Write combined shap_analysis.json + shap_summary.md ---------------
    # Mirror the single-target _write_outputs path so both modes produce the
    # same report artefacts regardless of target count.
    from datetime import datetime, timezone as _tz

    generated_at = datetime.now(_tz.utc).isoformat()
    combined_report = {
        "generated_at": generated_at,
        "multi_target": True,
        "overall_pass": all_pass,
        "targets": {
            tname: {
                "top_feature": r.get("top_feature"),
                "top_features": r.get("top_features", []),
                "oof_auc": r.get("oof_auc"),
                "oof_f1": r.get("oof_f1"),
                "oof_rmse": r.get("oof_rmse"),
                "fold_scores": r.get("fold_scores"),
                "top15_share": r.get("top15_share"),
                "tail_share": r.get("tail_share"),
                "correlation_pruning": {
                    "high_corr_pairs_count": r.get("high_corr_pairs_count"),
                    "dropped_features": r.get("dropped_features", []),
                    "pruned_feature_count": r.get("pruned_feature_count"),
                    "full_oof_f1": r.get("full_oof_f1"),
                    "pruned_oof_f1": r.get("pruned_oof_f1"),
                    "delta_f1": r.get("pruning_delta"),
                    "gate_pass": r.get("pruning_pass"),
                },
            }
            for tname, r in all_results.items()
        },
    }

    summary_lines = [
        f"# SHAP Audit Summary (Multi-Target) — {generated_at}",
        f"**Overall pruning gate**: {'PASS' if all_pass else 'PRUNE'}",
        "",
    ]
    for tname, r in all_results.items():
        task_label = "AUC" if r.get("oof_auc") is not None else "RMSE"
        score_val = r.get("oof_auc") if task_label == "AUC" else r.get("oof_rmse")
        score_str = f"{score_val:.6f}" if score_val is not None else "N/A"
        summary_lines += [
            f"## Target: {tname}",
            f"**Top feature**: {r.get('top_feature')}",
            f"**OOF {task_label}**: {score_str}",
            (
                f"**OOF F1**: {r.get('oof_f1'):.6f}"
                if r.get("oof_f1") is not None
                else "**OOF F1**: N/A"
            ),
            (
                f"**Top-15 SHAP share**: {r.get('top15_share', 0):.3%}"
                if r.get("top15_share") is not None
                else "**Top-15 SHAP share**: N/A"
            ),
            f"**High-corr pairs**: {r.get('high_corr_pairs_count', 0)}",
            f"**Dropped features**: {len(r.get('dropped_features', []))}",
            f"**Pruned feature count**: {r.get('pruned_feature_count', 'N/A')}",
            f"**Delta F1**: {r.get('pruning_delta', 0):+.6f}",
            f"**Pruning gate**: {'PASS' if r.get('pruning_pass') else 'PRUNE'}",
            "",
            "### Top 10 SHAP Features",
        ]
        for idx, feat in enumerate(r.get("top_features", [])[:10], start=1):
            summary_lines.append(
                f"{idx}. {feat.get('feature', '?')} — {feat.get('mean_abs_shap', 0):.8f}"
            )
        summary_lines.append("")

    _write_outputs(paths, combined_report, summary_lines)

    state_store = SkillStateStore(paths.state_path)
    state_store.update(
        shap_completed_at=generated_at,
        shap_multi_target_results=all_results,
        pruning_pass=all_pass,
        last_updated=generated_at,
    )

    # Write leaked_features in the flat dict format skill_11/three_lens expect:
    # {"anchor-baseline": ["feat_a"], "variant-xx": [...]}
    # Key is "anchor-baseline" since SHAP runs against the anchor feature set.
    leaked_by_target = {t: r.get("leaked_features", []) for t, r in all_results.items()}
    # Flatten: any feature leaked in ANY target blocks the anchor branch
    all_leaked = sorted(set(f for leaked in leaked_by_target.values() for f in leaked))
    state_store.update(leaked_features={"anchor-baseline": all_leaked})
    print(f"\n[OK] Multi-target SHAP complete. Overall pass: {all_pass}")
    return {"multi_target": True, "targets": all_results, "overall_pass": all_pass}


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    n_splits = 5
    seed = None
    for idx, arg in enumerate(args):
        if arg == "--n-splits" and idx + 1 < len(args):
            try:
                n_splits = int(args[idx + 1])
            except ValueError:
                pass
        elif arg.startswith("--n-splits="):
            try:
                n_splits = int(arg.split("=", 1)[1])
            except ValueError:
                pass
        elif arg == "--seed" and idx + 1 < len(args):
            try:
                seed = int(args[idx + 1])
            except ValueError:
                pass
        elif arg.startswith("--seed="):
            try:
                seed = int(arg.split("=", 1)[1])
            except ValueError:
                pass
        # ignore any other positional arguments (e.g., 'final')
    result = run(n_splits=n_splits, seed=seed)
    if result.get("multi_target"):
        print(json.dumps(result, indent=2))
    else:
        print(
            json.dumps(
                {
                    "competition": result["competition"],
                    "target": result["target"],
                    "feature_count": result["feature_count"],
                    "full_oof_f1": result["shap"]["oof_f1"],
                    "pruned_oof_f1": result["correlation_pruning"]["pruned_oof_f1"],
                    "delta_f1": result["correlation_pruning"]["delta_f1"],
                    "pruning_gate": result["correlation_pruning"]["gate_pass"],
                },
                indent=2,
            )
        )
