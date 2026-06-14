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

import json
from collections.abc import Iterable
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import f1_score, roc_auc_score
from zindian.cv import make_cv_splitter
from sklearn.preprocessing import StandardScaler

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
    fallback = paths.data_raw_dir / "Training_Data.csv"
    if full.exists():
        return pd.read_csv(full)
    if processed.exists():
        return pd.read_csv(processed)
    if fallback.exists():
        return pd.read_csv(fallback)
    raise FileNotFoundError(f"Could not find {full}, {processed} or {fallback}")


def _detect_target(config: ChallengeConfig, frame: pd.DataFrame) -> str:
    for key in ("target_column", "target", "label", "output_column"):
        value = config.get(key)
        if isinstance(value, str) and value and value in frame.columns:
            return value
    for candidate in ("target", "label", "target_col", "y"):
        if candidate in frame.columns:
            return candidate
    for column in frame.columns:
        series = frame[column].dropna()
        if not series.empty and series.isin([0, 1]).all() and series.nunique() <= 2:
            return column
    raise ValueError("Could not infer target column")


def _feature_columns(frame: pd.DataFrame, target: str) -> list[str]:
    excluded = {"id", "latitude", "longitude", target.lower()}
    return [column for column in frame.columns if column.lower() not in excluded]


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
) -> lgb.LGBMClassifier | lgb.LGBMRegressor:
    try:
        from zindian.config import ChallengeConfig
        config = ChallengeConfig.load()
        task_type = config.get("task_type", "classification")
    except Exception:
        task_type = "classification"

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
        y = np.asarray(frame[target].values, dtype=np.int32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Resolve canonical seed if caller passed None
    if seed is None:
        from zindian.config import get_seed

        seed = get_seed()
    seed = int(seed)

    splitter = make_cv_splitter(n_splits=n_splits, random_seed=seed)
    oof_probs = np.zeros(len(frame), dtype=np.float64)
    fold_aucs: list[float] = []
    fold_importances: list[np.ndarray] = []

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y), start=1):
        model = _train_shap_fold_model(
            X[train_idx],
            y[train_idx],
            X[val_idx],
            y[val_idx],
            seed=seed + fold_idx,
        )
        if task_type == "regression":
            val_preds = np.asarray(model.predict(X[val_idx]), dtype=np.float64)
            oof_probs[val_idx] = val_preds
            from sklearn.metrics import root_mean_squared_error
            fold_rmse = float(root_mean_squared_error(y[val_idx], val_preds))
            fold_aucs.append(fold_rmse)
            print(f"  Fold {fold_idx}/{n_splits}: rmse={fold_rmse:.6f}")
        else:
            val_probs = np.asarray(model.predict_proba(X[val_idx]), dtype=np.float64)[:, 1]
            oof_probs[val_idx] = val_probs
            fold_auc = float(roc_auc_score(y[val_idx], val_probs))
            fold_aucs.append(fold_auc)
            print(f"  Fold {fold_idx}/{n_splits}: auc={fold_auc:.6f}")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X[val_idx], check_additivity=False)
        fold_importance = np.abs(_as_positive_shap_values(shap_values)).mean(axis=0)
        fold_importances.append(fold_importance)

    mean_abs_shap = np.mean(np.vstack(fold_importances), axis=0)
    ranking = (
        pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs_shap})
        .sort_values(["mean_abs_shap", "feature"], ascending=[False, True])
        .reset_index(drop=True)
    )

    shap_total = float(ranking["mean_abs_shap"].sum()) if not ranking.empty else 0.0
    top15_share = (
        float(ranking.head(15)["mean_abs_shap"].sum() / shap_total)
        if shap_total > 0
        else 0.0
    )
    tail_share = float(1.0 - top15_share) if shap_total > 0 else 0.0

    if task_type == "regression":
        from sklearn.metrics import root_mean_squared_error
        oof_rmse = float(root_mean_squared_error(y, oof_probs))
        return {
            "oof_probs": oof_probs,
            "oof_auc": 0.0,
            "oof_f1": 0.0,
            "oof_rmse": oof_rmse,
            "threshold": 0.0,
            "fold_aucs": fold_aucs,
            "ranking": ranking,
            "top15_share": top15_share,
            "tail_share": tail_share,
        }
    else:
        thresholds = np.arange(0.3, 0.7, 0.01)
        best_threshold = float(
            max(thresholds, key=lambda t: f1_score(y, (oof_probs >= t).astype(int)))
        )
        oof_f1 = float(f1_score(y, (oof_probs >= best_threshold).astype(int)))
        oof_auc = float(roc_auc_score(y, oof_probs))

        return {
            "oof_probs": oof_probs,
            "oof_auc": oof_auc,
            "oof_f1": oof_f1,
            "threshold": best_threshold,
            "fold_aucs": fold_aucs,
            "ranking": ranking,
            "top15_share": top15_share,
            "tail_share": tail_share,
        }


def _build_pruned_feature_set(
    feature_cols: list[str], ranking: pd.DataFrame, frame: pd.DataFrame
) -> dict:
    corr = frame[feature_cols].corr().abs()
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
    print(f"✅ SHAP report written → {report_path}")
    print(f"✅ SHAP summary written → {summary_path}")


def run(n_splits: int = 5, seed: int | None = None) -> dict:
    paths = resolve_competition_paths(require_competition=True)
    if paths.competition_dir is None:
        raise FileNotFoundError("Competition directory could not be resolved")

    config = ChallengeConfig.load()
    state = SkillStateStore(paths.state_path).read()
    frame = _load_train_frame(paths)
    target = _detect_target(config, frame)
    feature_cols = _feature_columns(frame, target)

    print(f"Competition      : {config.slug}")
    print(f"Target           : {target}")
    print(f"Features         : {len(feature_cols)}")
    print(f"DAG phase        : {state.get('dag_phase')}")

    task_type = config.get("task_type", "classification")

    if len(feature_cols) < 2:
        print("⚠️ X.shape[1] < 2: skipping SHAP ratio audit.")
        # Perform 5-fold CV to get oof_probs and metrics
        splitter = make_cv_splitter(n_splits=n_splits, random_seed=seed or get_seed())
        X = np.asarray(frame[feature_cols].values, dtype=np.float64)
        if task_type == "regression":
            y = np.asarray(frame[target].values, dtype=np.float64)
        else:
            y = np.asarray(frame[target].values, dtype=np.int32)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        oof_probs = np.zeros(len(frame), dtype=np.float64)
        for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y), start=1):
            model = _train_shap_fold_model(
                X[train_idx], y[train_idx], X[val_idx], y[val_idx], seed=(seed or get_seed()) + fold_idx
            )
            if task_type == "regression":
                val_preds = np.asarray(model.predict(X[val_idx]), dtype=np.float64)
                oof_probs[val_idx] = val_preds
            else:
                val_probs = np.asarray(model.predict_proba(X[val_idx]), dtype=np.float64)[:, 1]
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
        write_oof_record(
            state_store,
            branch_name="shap_audit",
            scores=np.asarray(oof_probs, dtype=np.float64).tolist(),
            cv_strategy_id=cv_id,
            seed=int(seed if seed is not None else get_seed()),
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
                "fold_aucs": [0.5] * n_splits,
                "ranking": pd.DataFrame({"feature": feature_cols, "mean_abs_shap": [0.0]}),
                "top15_share": 1.0,
                "tail_share": 0.0,
            }
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
            "fold_aucs": full_audit["fold_aucs"],
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
            "full_oof_f1": full_cv.oof_rmse if task_type == "regression" else full_cv.oof_f1,
            "pruned_oof_f1": pruned_cv.oof_rmse if task_type == "regression" else pruned_cv.oof_f1,
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


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    n_splits = 5
    seed = None
    for idx, arg in enumerate(args):
        if arg == "--n-splits" and idx + 1 < len(args):
            n_splits = int(args[idx + 1])
        elif arg.startswith("--n-splits="):
            n_splits = int(arg.split("=", 1)[1])
        elif arg == "--seed" and idx + 1 < len(args):
            seed = int(args[idx + 1])
        elif arg.startswith("--seed="):
            seed = int(arg.split("=", 1)[1])

    result = run(n_splits=n_splits, seed=seed)
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
