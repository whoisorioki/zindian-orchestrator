"""
Skill 08 — Anchor Baseline
Train LightGBM baseline on base features only.
Features excluded per policy filters in challenge_config.json.
Lock first confirmed anchor artifacts and create git branch.
Must run after Skill 07 feature engineering completes.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

# KFold usage is delegated to the central CV factory / shared trainer
from zindian.cv import make_cv_splitter

from zindian.config import ChallengeConfig
from zindian.config import get_seed
from zindian.paths import resolve_competition_paths
from zindian.state import (
    SkillStateStore,
    resolve_active_cv_strategy_id,
    write_oof_record,
)
from zindian.ledger import Ledger
from zindian.skills._lightgbm_shared import train_lightgbm_cv

# ── Data ───────────────────────────────────────────────────────────────────────


def load_data(
    paths, config: ChallengeConfig
) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """
    Load training and test data.
    Returns (train, test, training_target_col, submission_col).
    These are intentionally different — do not conflate them.
    """
    # Load from processed features — base features live here
    train = pd.read_csv(paths.data_processed_dir / "features_train.csv")
    test = pd.read_csv(paths.data_processed_dir / "features_test.csv")

    # Training target must be initialized during intake and read dynamically.
    training_target_col = config.get("target_col") or config.get("target_column")
    if not training_target_col:
        # Fallback inference for legacy/minimal configs: target is any train-only column
        # excluding common coordinate/id columns. This keeps the module config-driven
        # while remaining robust for old fixtures.
        cols_cfg = config.get("columns", {}) or {}
        id_col = (
            config.get("id_col") or config.get("id_column") or cols_cfg.get("id", "ID")
        )
        lat_col = cols_cfg.get("latitude", "Latitude")
        lon_col = cols_cfg.get("longitude", "Longitude")
        candidate_cols = [
            c
            for c in train.columns
            if c not in test.columns and c not in {id_col, lat_col, lon_col}
        ]
        if len(candidate_cols) == 1:
            training_target_col = candidate_cols[0]
        else:
            raise RuntimeError("target_col not initialized in challenge_config.json")

    # Submission column: what Zindi expects in the CSV header
    input_files = config.get("input_files", {}) or {}
    sample_file = input_files.get("sample", "SampleSubmission.csv")
    cols_cfg = config.get("columns", {}) or {}
    id_col = config.get("id_col") or config.get("id_column") or cols_cfg.get("id", "ID")
    sample = pd.read_csv(paths.data_raw_dir / sample_file)
    submission_col = [c for c in sample.columns if c != id_col][0]

    return train, test, training_target_col, submission_col


# ── Training ───────────────────────────────────────────────────────────────────


def compute_oof_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: ChallengeConfig,
    target_col: str,
    state: dict | None = None,
    n_splits: int = 5,
    random_seed: int | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    float,
    list[tuple[np.ndarray, np.ndarray]],
    list[str],
    str,
    int,
    list[float],
]:
    """
    Train LightGBM with KFold cross-validation on base features only.
    Returns (oof_preds, test_preds, oof_logloss, oof_auc, oof_f1, best_threshold).
    Computes F1 using optimal threshold (challenge metric).
    """
    if random_seed is None:
        random_seed = get_seed()

    cols_cfg = config.get("columns", {}) or {}
    id_col = config.get("id_col") or config.get("id_column") or cols_cfg.get("id", "ID")

    # Resolve active CV strategy from state override first, then config.
    state = state or {}
    override_active = state.get("cv_strategy_override", {}).get("active", False)
    if override_active:
        active_strategy = state.get("cv_strategy_override", {}).get("override_strategy")
    else:
        active_strategy = config.get("cv_strategy", {}).get("type")
    if not active_strategy:
        active_strategy = "stratified"

    # Load policy blocked columns from config
    policy_blocked = config.get("policy_filters", []) or []
    excluded_cols = {id_col, target_col}
    for col in policy_blocked:
        if col is not None:
            excluded_cols.add(str(col))

    feature_cols = [c for c in train.columns if c not in excluded_cols]
    feature_count = len(feature_cols)

    # Option [C] challenge intercept: allow anchor_challenge to override params/model family.
    anchor_challenge = state.get("anchor_challenge", {})
    model_family = "lightgbm"
    model_params = {"learning_rate": 0.05, "num_leaves": 31, "seed": random_seed}
    if anchor_challenge.get("active", False):
        model_family = str(
            anchor_challenge.get("model_family")
            or anchor_challenge.get("framework")
            or "lightgbm"
        )
        model_params.update(
            anchor_challenge.get("params", {})
            or anchor_challenge.get("hyperparams", {})
            or {}
        )
        n_splits = int(anchor_challenge.get("n_splits") or n_splits)

    if model_family != "lightgbm":
        raise RuntimeError(
            f"Unsupported anchor_challenge model_family '{model_family}' in skill_08; supported: 'lightgbm'"
        )

    import random

    random.seed(random_seed)
    np.random.seed(random_seed)

    task_type = str(config.get("task_type", "classification")).lower()
    if task_type == "regression":
        y = np.asarray(train[target_col].values, dtype=np.float64)
    else:
        y = np.asarray(train[target_col].values, dtype=np.int32)
    groups = None
    group_col = (config.get("cv_strategy", {}) or {}).get("group_column")
    if group_col and group_col in train.columns:
        groups = np.asarray(train[group_col].values)

    splitter = make_cv_splitter(
        cv_strategy={"type": active_strategy, "n_splits": n_splits},
        random_seed=random_seed,
    )
    X_dummy = np.zeros((len(train), 1), dtype=np.float64)
    if groups is not None:
        split_iter = list(splitter.split(X_dummy, y, groups))
    else:
        split_iter = list(splitter.split(X_dummy, y))

    # Resolve regression metric for target transformation lifecycle (SoT v2.2)
    regression_metric = config.get("metric") if task_type == "regression" else None

    result = train_lightgbm_cv(
        train=train,
        test=test,
        feature_cols=feature_cols,
        target_col=target_col,
        n_splits=n_splits,
        random_seed=random_seed,
        cv=split_iter,
        params=model_params,
        num_boost_round=500,
        early_stopping_rounds=50,
        scale=True,
        regression_metric=regression_metric,
    )

    if task_type == "regression":
        oof_rmse = result.oof_rmse
        print(f"\nOOF RMSE     : {oof_rmse:.6f}")
        oof_logloss = oof_rmse
    else:
        from sklearn.metrics import log_loss

        try:
            oof_logloss = float(log_loss(y, result.oof_probs))
        except Exception:
            oof_logloss = 0.0
        print(f"\nOOF Log Loss : {oof_logloss:.6f}")
        print(f"OOF AUC      : {result.oof_auc:.6f}")
        print(f"OOF F1       : {result.oof_f1:.6f} (threshold={result.threshold:.2f})")

    cv_strategy_id = resolve_active_cv_strategy_id(state, config._data)
    return (
        result.oof_probs,
        result.test_probs,
        oof_logloss,
        result.oof_auc,
        result.oof_f1,
        result.threshold,
        split_iter,
        feature_cols,
        cv_strategy_id,
        feature_count,
        [float(score) for score in result.fold_aucs],
    )


# ── Git ────────────────────────────────────────────────────────────────────────


def create_git_branch(branch_name: str = "anchor-baseline") -> None:
    """Create and switch to the anchor git branch."""
    import subprocess

    try:
        subprocess.run(
            ["git", "checkout", "-b", branch_name],
            check=True,
            capture_output=True,
        )
        print(f"✅ Git branch created: {branch_name}")
    except subprocess.CalledProcessError:
        try:
            subprocess.run(
                ["git", "checkout", branch_name],
                check=True,
                capture_output=True,
            )
            print(f"✓  Switched to existing branch: {branch_name}")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Git branch operation failed: {e.stderr.decode()}")


# ── Save Submission CSV ────────────────────────────────────────────────────────


def next_submission_path(paths, suffix: str = "anchor") -> Path:
    """Return the next numbered submission path for this competition."""
    state = SkillStateStore(paths.state_path).read()
    highest_state_count = max(
        int(state.get("submissions_used_today") or 0),
        int(state.get("submissions_used_total") or 0),
    )

    highest_file_count = 0
    pattern = re.compile(r"^sub_(\d{3})_.*\.csv$")
    if paths.submissions_dir.exists():
        for path in paths.submissions_dir.glob("sub_*.csv"):
            match = pattern.match(path.name)
            if match:
                highest_file_count = max(highest_file_count, int(match.group(1)))

    next_num = max(highest_state_count, highest_file_count) + 1
    return paths.submissions_dir / f"sub_{next_num:03d}_{suffix}.csv"


def save_submission(
    test_ids: np.ndarray,
    predictions: np.ndarray,
    submission_col: str,
    output_path: Path,
) -> None:
    """Save predictions in Zindi submission format (probabilities or hard labels per config)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sub_df = pd.DataFrame(
        {
            "ID": np.asarray(test_ids),
            submission_col: np.asarray(predictions),
        }
    )
    sub_df.to_csv(output_path, index=False)
    print(f"✅ Submission CSV saved → {output_path}")


# ── Entry Point ────────────────────────────────────────────────────────────────


def run(
    *,
    n_splits: int = 5,
    random_seed: int | None = None,
    submit: bool = False,
) -> dict:
    """
    Skill 08 — Anchor Baseline.

    Args:
        n_splits:    KFold splits (default 5).
        random_seed: Global random seed. Defaults to the canonical config seed.
        submit:      Deprecated for this skill role. Skill 08 does not submit.
                 Submission is handled by `skill_16_submit`.

    Returns:
        dict with status, oof_logloss, oof_auc, submission path, git branch,
        submitted flag, and submission result if submitted.
    """
    print(f"\n{'=' * 60}")
    print("SKILL 08 — Anchor Baseline (LightGBM · Base features only)")
    print(f"{'=' * 60}\n")

    paths = resolve_competition_paths()
    config = ChallengeConfig.load()
    state_store = SkillStateStore(paths.state_path)
    state = state_store.read()

    if random_seed is None:
        random_seed = get_seed()
    import random

    random.seed(random_seed)
    np.random.seed(random_seed)

    print(f"Competition      : {config.slug}")
    print(f"Metric           : {config.metric}")
    print(f"Use probabilities: {config.use_probabilities}")

    # ── Load data ──────────────────────────────────────────────
    train, test, training_target_col, submission_col = load_data(paths, config)
    print("\nData loaded:")
    print(f"  Train          : {train.shape}")
    print(f"  Test           : {test.shape}")
    print(f"  Training target: {training_target_col}")
    print(f"  Submission col : {submission_col}")

    # ── Train ──────────────────────────────────────────────────
    print("\nTraining LightGBM anchor baseline…")
    try:
        result = compute_oof_predictions(
            train,
            test,
            config,
            training_target_col,
            state=state,
            n_splits=n_splits,
            random_seed=random_seed,
        )
        if len(result) >= 11:
            (
                oof_preds,
                test_preds,
                oof_logloss,
                oof_auc,
                oof_f1,
                best_t,
                split_iter,
                feature_cols,
                cv_strategy_id,
                feature_count,
                fold_scores_list,
            ) = result
        else:
            (
                oof_preds,
                test_preds,
                oof_logloss,
                oof_auc,
                oof_f1,
                best_t,
                split_iter,
                feature_cols,
                cv_strategy_id,
                feature_count,
            ) = result[:10]
            fold_scores_list = [0.0] * n_splits
    except (TypeError, ValueError):
        # Backward compatibility for monkeypatched tests expecting the old
        # compute_oof_predictions signature.
        compat_result = cast(
            tuple[Any, ...],
            compute_oof_predictions(
                train,
                test,
                config,
                training_target_col,
                n_splits=n_splits,
                random_seed=random_seed,
            ),
        )
        if len(compat_result) >= 10:
            (
                oof_preds,
                test_preds,
                oof_logloss,
                oof_auc,
                oof_f1,
                best_t,
                split_iter,
                feature_cols,
                cv_strategy_id,
                feature_count,
            ) = compat_result[:10]
            fold_scores_list = (
                compat_result[10] if len(compat_result) >= 11 else [0.0] * n_splits
            )
        elif len(compat_result) >= 6:
            oof_preds, test_preds, oof_logloss, oof_auc, oof_f1, best_t = compat_result[
                :6
            ]
            # Reconstruct required metadata in compatibility path.
            cols_cfg = config.get("columns", {}) or {}
            id_col = (
                config.get("id_col")
                or config.get("id_column")
                or cols_cfg.get("id", "ID")
            )
            lat_col = cols_cfg.get("latitude", "Latitude")
            lon_col = cols_cfg.get("longitude", "Longitude")
            feature_cols = [
                c
                for c in train.columns
                if c not in (id_col, lat_col, lon_col, training_target_col)
            ]
            feature_count = len(feature_cols)
            cv_strategy_id = resolve_active_cv_strategy_id(state, config._data)

            # Legacy/test compatibility path: build a deterministic index mapping
            # without requiring valid stratification/group constraints.
            idx = np.arange(len(train), dtype=int)
            split_iter = [(idx, idx)]
            fold_scores_list = [0.0] * n_splits
        else:
            raise RuntimeError(
                "compute_oof_predictions returned an unexpected tuple shape"
            )

    # ── Save OOF predictions ───────────────────────────────────
    oof_path = paths.data_raw_dir / "oof_anchor.csv"
    # Safe index tracking: use explicit CV validation indices to map OOF predictions.
    oof_index = np.full(len(train), -1, dtype=int)
    for _, val_idx in split_iter:
        oof_index[np.asarray(val_idx, dtype=int)] = np.asarray(val_idx, dtype=int)
    cols_cfg = config.get("columns", {}) or {}
    id_col = config.get("id_col") or config.get("id_column") or cols_cfg.get("id", "ID")
    pd.DataFrame(
        {
            id_col: np.asarray(train[id_col].values),
            "row_index": oof_index,
            "Predicted": oof_preds,
            training_target_col: np.asarray(train[training_target_col].values),
        }
    ).to_csv(oof_path, index=False)
    print(f"✅ OOF predictions saved → {oof_path}")

    # ── Save submission CSV ────────────────────────────────────
    sub_path = next_submission_path(paths)
    input_files = config.get("input_files", {}) or {}
    input_files.get("sample", "SampleSubmission.csv")
    # Probability-aware output format from config
    task_type = str(config.get("task_type", "classification")).lower()
    if task_type == "regression" or config.get("use_probabilities", True):
        predictions_to_save = np.asarray(test_preds, dtype=np.float64)
    else:
        print(f"\nApplying optimal F1 threshold: {best_t:.2f}")
        predictions_to_save = (
            np.asarray(test_preds, dtype=np.float64) >= best_t
        ).astype(int)

    save_submission(
        np.asarray(test[id_col].values), predictions_to_save, submission_col, sub_path
    )

    # ── Log to DuckDB ledger ───────────────────────────────────
    ledger = Ledger()
    exp_id = ledger.log_experiment(
        branch_name="anchor-baseline",
        # Classification metrics are tracked in SKILL_STATE and notes; avoid writing F1 into RMSE field.
        oof_rmse=None,
        feature_count=feature_count,
        calibration_method="none",
        gate_result="PASS",
        gate_reason="Initial anchor baseline — Base features only. Metric: F1 Score (threshold={:.2f}), AUC={:.4f}".format(
            best_t, oof_auc
        ),
        dag_phase="phase_2_anchor_confirmed",
        notes=f"oof_f1={oof_f1:.6f}; oof_auc={oof_auc:.6f}; cv_strategy_id={cv_strategy_id}",
    )
    ledger.close()
    print(f"✅ Experiment logged → DuckDB exp_id={exp_id}")

    # ── Update SKILL_STATE.json ────────────────────────────────
    retraining_active = bool(
        state.get("pseudo_label_result", {}).get("retraining_required", False)
    )
    if retraining_active:
        f1_key = "anchor_oof_f1_augmented"
        auc_key = "anchor_oof_auc_augmented"
        score_key = "anchor_oof_score_augmented"
    else:
        f1_key = "anchor_oof_f1"
        auc_key = "anchor_oof_auc"
        score_key = "anchor_oof_score"

    task_type = str(config.get("task_type", "classification")).lower()
    metric_name = str(config.get("metric", "f1")).lower()
    # SoT v2.2 metric_map: covers all canonical metric name variants
    # including the explicit root_mean_squared_error and mean_absolute_error
    # strings defined in the Regression Target Transformation Lifecycle.
    metric_map = {
        "auc": oof_auc,
        "f1": oof_f1,
        "f1_score": oof_f1,
        "rmse": oof_logloss if task_type == "regression" else oof_f1,
        "root_mean_squared_error": oof_logloss if task_type == "regression" else oof_f1,
        "mae": oof_logloss if task_type == "regression" else oof_f1,
        "mean_absolute_error": oof_logloss if task_type == "regression" else oof_f1,
        "logloss": oof_logloss,
        "log_loss": oof_logloss,
        "rmsle": oof_logloss if task_type == "regression" else oof_f1,
    }
    anchor_oof_score = metric_map.get(metric_name, oof_f1)

    secondary_metrics = None
    if task_type == "regression":
        from zindian.state import compute_secondary_metrics

        y_true = np.asarray(train[training_target_col].values, dtype=np.float64)
        secondary_metrics = compute_secondary_metrics(y_true, oof_preds)

    updates = {
        score_key: anchor_oof_score,
        "anchor_git_branch": "anchor-baseline",
        "anchor_cv_strategy_id": cv_strategy_id,
        "dag_phase": "phase_2_anchor_confirmed",
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }
    # Deprecated: anchor_oof_f1 and anchor_oof_auc are preserved for backward compatibility
    updates[f1_key] = oof_f1
    updates[auc_key] = oof_auc

    state_store.update(**updates)

    branch_name = (
        "anchor-baseline_augmented" if retraining_active else "anchor-baseline"
    )
    write_oof_record(
        state_store,
        branch_name=branch_name,
        scores=np.asarray(oof_preds, dtype=np.float64).tolist(),
        cv_strategy_id=cv_strategy_id,
        seed=int(random_seed if random_seed is not None else get_seed()),
        model_config={
            "feature_count": feature_count,
            "n_splits": n_splits,
            "threshold": float(best_t),
            "active_strategy": (
                state.get("cv_strategy_override", {}).get("override_strategy")
                if state.get("cv_strategy_override", {}).get("active", False)
                else config.get("cv_strategy", {}).get("type")
            ),
            "fold_scores": fold_scores_list,
        },
        secondary_metrics=secondary_metrics,
    )
    print(
        f"✅ SKILL_STATE.json updated: score={anchor_oof_score:.6f}  f1={oof_f1:.6f}  auc={oof_auc:.6f}  threshold={best_t:.2f}"
    )

    # ── Create git branch ──────────────────────────────────────
    create_git_branch("anchor-baseline")

    # ── Submission responsibility boundary ─────────────────────
    # Skill 08 is single-role: train and persist anchor artifacts only.
    # Submission is delegated to skill_16_submit.
    submission_result = None
    if submit:
        print(
            "⚠️  submit=True ignored in skill_08_anchor. Use skill_16_submit for submissions."
        )

    print(f"""
{"=" * 60}
Submission NOT triggered (single-role boundary: skill_16_submit owns submissions).
OOF Log Loss : {oof_logloss:.6f}
OOF AUC      : {oof_auc:.6f}

To submit when ready (via orchestrator flow):
  run skill_16_submit after gates and inference formatting
{"=" * 60}""")

    return {
        "status": "OK",
        "oof_logloss": oof_logloss,
        "oof_auc": oof_auc,
        "submission_path": str(sub_path),
        "git_branch": state_store.read().get("current_git_branch", "anchor-baseline"),
        "n_features": feature_count,
        "submitted": submission_result is not None,
        "submission_result": submission_result,
        "message": "Anchor baseline trained and locked (submission delegated to skill_16_submit)",
    }


if __name__ == "__main__":
    import sys

    submit_flag = "--submit" in sys.argv
    result = run(submit=submit_flag)
    printable = {k: v for k, v in result.items() if k != "submission_result"}
    print(json.dumps(printable, indent=2))
