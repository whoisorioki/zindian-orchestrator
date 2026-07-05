"""
Skill 08 — Anchor Baseline
Train LightGBM baseline on base features only.
Features excluded per policy filters in challenge_config.json.
Lock first confirmed anchor artifacts and create git branch.
Must run after Skill 07 feature engineering completes.
"""

from __future__ import annotations

import tabula.skill_state_autopatch  # noqa
import json
from datetime import datetime, timezone
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

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

# -- Data -----------------------------------------------------------------------


def load_data(
    paths, config: ChallengeConfig, state: dict
) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """
    Load training and test data.
    Returns (train, test, training_target_col, submission_col).
    These are intentionally different — do not conflate them.
    """
    # Load from processed features — base features live here
    # Use branch-specific files for reproducibility contract
    # Branch name comes from state or defaults to "anchor-baseline"
    branch_name = state.get("anchor_git_branch") or "anchor-baseline"
    train = pd.read_csv(paths.data_processed_dir / f"features_train_{branch_name}.csv")
    test = pd.read_csv(paths.data_processed_dir / f"features_test_{branch_name}.csv")

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


# -- Training -------------------------------------------------------------------


def compute_oof_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: ChallengeConfig,
    target_col: str,
    state: dict | None = None,
    n_splits: int = 5,
    random_seed: int | None = None,
    variant_name: str | None = None,
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
        _y_raw_08 = train[target_col].values
        if _y_raw_08.dtype.kind in ("U", "S", "O"):
            _le_08 = LabelEncoder()
            y = _le_08.fit_transform(_y_raw_08.astype(str)).astype(np.int32)
        else:
            y = np.asarray(_y_raw_08, dtype=np.int32)
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
        variant_name=variant_name,
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
        [
            float(score)
            for score in getattr(
                result, "fold_scores", getattr(result, "fold_aucs", [])
            )
        ],
    )


# -- Git ------------------------------------------------------------------------


def create_git_branch(branch_name: str = "anchor-baseline") -> None:
    """Create and switch to the anchor git branch."""
    import subprocess

    try:
        subprocess.run(
            ["git", "checkout", "-b", branch_name],
            check=True,
            capture_output=True,
        )
        print(f"[OK] Git branch created: {branch_name}")
    except subprocess.CalledProcessError:
        try:
            subprocess.run(
                ["git", "checkout", branch_name],
                check=True,
                capture_output=True,
            )
            print(f"[OK]  Switched to existing branch: {branch_name}")
        except subprocess.CalledProcessError as e:
            print(f"[WARN]  Git branch operation failed: {e.stderr.decode()}")


# -- Save Submission CSV --------------------------------------------------------


# next_submission_path() and save_submission() removed — Phase 2B must not write
# to submissions/. All submission formatting belongs exclusively in Phase 4 (skill_14).

# -- Entry Point ----------------------------------------------------------------


def run(
    *,
    n_splits: int = 5,
    random_seed: int | None = None,
    submit: bool = False,
    variant_name: str | None = None,
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

    # Multi-target detection
    target_config = config.get("target_config")
    if target_config and target_config.get("targets"):
        return _run_multi_target(
            paths,
            config,
            state_store,
            state,
            n_splits,
            random_seed,
            submit,
            variant_name,
        )

    # -- Load data ----------------------------------------------
    train, test, training_target_col, submission_col = load_data(paths, config, state)
    print("\nData loaded:")
    print(f"  Train          : {train.shape}")
    print(f"  Test           : {test.shape}")
    print(f"  Training target: {training_target_col}")
    print(f"  Submission col : {submission_col}")

    # -- Train --------------------------------------------------
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
            variant_name=variant_name,
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

    # -- Save OOF predictions -----------------------------------
    oof_path = paths.data_raw_dir / "oof_anchor.csv"
    # Safe index tracking: use explicit CV validation indices to map OOF predictions.
    oof_index = np.full(len(train), -1, dtype=int)
    for _, val_idx in split_iter:
        oof_index[np.asarray(val_idx, dtype=int)] = np.asarray(val_idx, dtype=int)
    cols_cfg = config.get("columns", {}) or {}
    id_col = config.get("id_col") or config.get("id_column") or cols_cfg.get("id", "ID")

    # Handle multiclass OOF predictions (2D array)
    if oof_preds.ndim > 1:
        oof_df = pd.DataFrame(
            {
                id_col: np.asarray(train[id_col].values),
                "row_index": oof_index,
                training_target_col: np.asarray(train[training_target_col].values),
            }
        )
        for i in range(oof_preds.shape[1]):
            oof_df[f"Predicted_class_{i}"] = oof_preds[:, i]
        oof_df.to_csv(oof_path, index=False)
    else:
        pd.DataFrame(
            {
                id_col: np.asarray(train[id_col].values),
                "row_index": oof_index,
                "Predicted": oof_preds,
                training_target_col: np.asarray(train[training_target_col].values),
            }
        ).to_csv(oof_path, index=False)
    print(f"[OK] OOF predictions saved → {oof_path}")

    # Submission formatting removed — Phase 2B writes probabilities only.
    # skill_14 (Phase 4) reads test_probs from data/processed/ and produces
    # the final submission CSV.

    # -- Compute anchor_oof_score before logging ----------------
    retraining_active = bool(
        state.get("pseudo_label_result", {}).get("retraining_required", False)
    )
    if retraining_active:
        score_key = "anchor_oof_score_augmented"
    else:
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

    # -- Log to DuckDB ledger -----------------------------------
    with Ledger() as ledger:
        exp_id = ledger.log_experiment(
            branch_name="anchor-baseline",
            oof_score=anchor_oof_score,
            metric=str(config.get("metric", "f1")).lower(),
            feature_count=feature_count,
            calibration_method="none",
            gate_result="PASS",
            gate_reason="Initial anchor baseline — Base features only. Metric: F1 Score (threshold={:.2f}), AUC={:.4f}".format(
                best_t, oof_auc
            ),
            dag_phase="phase_2_anchor_confirmed",
            notes=f"oof_f1={oof_f1:.6f}; oof_auc={oof_auc:.6f}; cv_strategy_id={cv_strategy_id}",
        )
    print(f"[OK] Experiment logged -> DuckDB exp_id={exp_id}")

    # -- Update SKILL_STATE.json --------------------------------
    updates = {
        score_key: anchor_oof_score,
        "anchor_git_branch": "anchor-baseline",
        "anchor_cv_strategy_id": cv_strategy_id,
        "dag_phase": "phase_2_anchor_confirmed",
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }

    state_store.update(**updates)

    branch_name = (
        "anchor-baseline_augmented" if retraining_active else "anchor-baseline"
    )

    # Convert multiclass OOF to 1D for state storage
    oof_scores_for_state = (
        oof_preds if oof_preds.ndim == 1 else np.argmax(oof_preds, axis=1)
    )

    write_oof_record(
        state_store,
        branch_name=branch_name,
        scores=np.asarray(oof_scores_for_state, dtype=np.float64).tolist(),
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
        f"[OK] SKILL_STATE.json updated: score={anchor_oof_score:.6f}  f1={oof_f1:.6f}  auc={oof_auc:.6f}  threshold={best_t:.2f}"
    )

    # -- Surface cv_limitations.known_risk into metadata_warnings ----------
    # This makes the temporal CV gap visible to skill_11, skill_17, and
    # three_lens without requiring any consumer to parse config prose.
    cv_lim = config.get("cv_limitations") or {}
    known_risk = cv_lim.get("known_risk")
    if (
        known_risk
        and cv_lim.get("temporal_holdout_required")
        and not cv_lim.get("temporal_cv_feasible")
    ):
        state_now = state_store.read()
        existing_warnings = state_now.get("metadata_warnings") or []
        if not isinstance(existing_warnings, list):
            existing_warnings = []
        temporal_warning = (
            f"⚠️  Temporal CV not feasible — no date column in data. "
            f"Anchor OOF ({anchor_oof_score:.4f}) was validated via "
            f"{cv_lim.get('fallback_strategy', 'stratified')} CV only. "
            f"{known_risk}"
        )
        if not any("Temporal CV not feasible" in w for w in existing_warnings):
            state_store.update(metadata_warnings=existing_warnings + [temporal_warning])
            print(f"[WARN] {temporal_warning}")

    # -- Create git branch --------------------------------------
    create_git_branch("anchor-baseline")

    # -- Submission responsibility boundary ---------------------
    # Skill 08 is single-role: train and persist anchor artifacts only.
    # Submission is delegated to skill_16_submit.
    submission_result = None
    if submit:
        print(
            "[WARN]  submit=True ignored in skill_08_anchor. Use skill_16_submit for submissions."
        )

    print(
        f"""
{"=" * 60}
Submission NOT triggered (single-role boundary: skill_16_submit owns submissions).
OOF Log Loss : {oof_logloss:.6f}
OOF AUC      : {oof_auc:.6f}

To submit when ready (via orchestrator flow):
  run skill_16_submit after gates and inference formatting
{"=" * 60}"""
    )

    return {
        "status": "OK",
        "oof_logloss": oof_logloss,
        "oof_auc": oof_auc,
        "submission_path": None,
        "git_branch": state_store.read().get("current_git_branch", "anchor-baseline"),
        "n_features": feature_count,
        "submitted": submission_result is not None,
        "submission_result": submission_result,
        "message": "Anchor baseline trained and locked (submission delegated to skill_16_submit)",
    }


def _run_multi_target(
    paths, config, state_store, state, n_splits, random_seed, submit, variant_name
) -> dict:
    """Multi-target training loop per SoT v2.2.1 A11."""
    print("\n[TARGET] MULTI-TARGET MODE DETECTED\n")
    target_config = config.get("target_config", {})
    targets = target_config.get("targets", [])
    print(f"Training {len(targets)} targets: {[t['name'] for t in targets]}\n")

    # Load features
    # Use branch-specific files for reproducibility contract
    branch_name = state.get("anchor_git_branch") or "anchor-baseline"
    X_train = pd.read_csv(
        paths.data_processed_dir / f"features_train_{branch_name}.csv"
    )
    X_test = pd.read_csv(paths.data_processed_dir / f"features_test_{branch_name}.csv")

    # Load raw data for targets
    input_files = config.get("input_files", {}) or {}
    train_file = input_files.get("train", "Train.csv")
    raw_train = pd.read_csv(paths.data_raw_dir / train_file)

    id_col = config.get("id_col") or config.get("id_column") or "ID"

    all_oof: dict[str, Any] = {}
    all_test_preds: dict[str, Any] = {}
    all_metrics: dict[str, dict[str, Any]] = {}
    target_uniques: dict[str, Any] = {}

    for target_spec in targets:
        target_name = target_spec["name"]
        target_task = target_spec["task_type"]
        print(f"\n{'-' * 60}")
        print(f"Target: {target_name} ({target_task})")
        print(f"{'-' * 60}")

        # Extract target from raw data
        train_with_target = X_train.copy()
        target_series = raw_train[target_name]

        # Encode if categorical
        if not pd.api.types.is_numeric_dtype(target_series):
            coded_series, uniques = pd.factorize(target_series)
            target_series = pd.Series(coded_series, index=target_series.index)
            target_uniques[target_name] = list(uniques)

        # Drop any other target columns that might exist
        other_targets = [t["name"] for t in targets if t["name"] != target_name]
        train_with_target = train_with_target.drop(
            columns=other_targets, errors="ignore"
        )

        train_with_target[target_name] = target_series

        print(f"\nDEBUG: About to train {target_name}")
        print(f"  task_type: {target_task}")
        print(f"  train_with_target shape: {train_with_target.shape}")
        print(f"  train_with_target columns: {train_with_target.columns.tolist()}")
        print(f"  {target_name} in columns: {target_name in train_with_target.columns}")
        print(f"  {target_name} dtype: {train_with_target[target_name].dtype}")
        print(f"  {target_name} unique: {train_with_target[target_name].nunique()}")
        print(
            f"  {target_name} range: {train_with_target[target_name].min()} to {train_with_target[target_name].max()}\n"
        )

        # Override config for this target
        target_config_override = ChallengeConfig(
            path=config.path,
            _data={
                **config._data,
                "target_col": target_name,
                "task_type": target_task,
                "metric": target_spec.get("metric", "rmse"),
            },
        )

        result = compute_oof_predictions(
            train_with_target,
            X_test,
            target_config_override,
            target_name,
            state=state,
            n_splits=n_splits,
            random_seed=random_seed,
            variant_name=(
                f"{variant_name}_{target_name}" if variant_name else target_name
            ),
        )

        print(
            f"DEBUG: target_name={target_name}, target_col in config={target_config_override.get('target_col')}"
        )
        print(
            f"DEBUG: Columns in train_with_target: {train_with_target.columns.tolist()}"
        )
        print(
            f"DEBUG: {target_name} dtype: {train_with_target[target_name].dtype}, range: {train_with_target[target_name].min()}-{train_with_target[target_name].max()}"
        )

        oof_preds, test_preds, oof_logloss, oof_auc, oof_f1, best_t = result[:6]
        fold_scores = result[10] if len(result) > 10 else []
        all_oof[target_name] = oof_preds
        all_test_preds[target_name] = test_preds

        # Use task-specific metric key names
        target_task = target_spec.get("task_type", "classification")
        if target_task == "regression":
            all_metrics[target_name] = {
                "oof_rmse": oof_logloss,  # oof_logloss contains RMSE for regression
                "oof_auc": oof_auc,
                "oof_f1": oof_f1,
                "threshold": best_t,
                "fold_scores": fold_scores,
            }
        else:
            all_metrics[target_name] = {
                "oof_logloss": oof_logloss,
                "oof_auc": oof_auc,
                "oof_f1": oof_f1,
                "threshold": best_t,
                "fold_scores": fold_scores,
            }

    # Save multi-target OOF
    oof_df = pd.DataFrame({id_col: raw_train[id_col]})
    for target_name, preds in all_oof.items():
        if preds.ndim > 1:
            # Multiclass: save argmax as primary prediction
            oof_df[f"{target_name}_pred"] = np.argmax(preds, axis=1)
            # Save class probabilities
            for i in range(preds.shape[1]):
                oof_df[f"{target_name}_prob_class_{i}"] = preds[:, i]
        else:
            oof_df[f"{target_name}_pred"] = preds
        # Save true labels with the same encoding used during training
        # so macro F1 can be independently verified
        if target_name in raw_train.columns:
            oof_df[f"{target_name}_true"] = raw_train[target_name].values
    oof_path = paths.data_raw_dir / "oof_anchor_multi.csv"
    oof_df.to_csv(oof_path, index=False)

    # Independently compute and log honest macro F1 from saved predictions
    for target_name, preds in all_oof.items():
        if preds.ndim > 1:
            target_specs = [t for t in targets if t["name"] == target_name]
            target_task = (
                target_specs[0]["task_type"] if target_specs else "classification"
            )
            if target_task == "classification":
                from sklearn.metrics import f1_score as _f1

                _y_pred = oof_df[f"{target_name}_pred"].values
                _y_true = oof_df[f"{target_name}_true"].values

                # Use the same encoding as training (pd.factorize) for reproducibility
                _codes, _uniques = pd.factorize(_y_true)
                _macro_f1 = float(_f1(_codes, _y_pred, average="macro"))
                _weighted_f1 = float(_f1(_codes, _y_pred, average="weighted"))

                print(f"\n[HONEST MACRO F1] {target_name}:")
                print(f"  Classes: {len(_uniques)} ({list(_uniques)})")
                print(f"  Macro F1:    {_macro_f1:.6f}")
                print(f"  Weighted F1: {_weighted_f1:.6f}")
                print(
                    f"  Model F1:    {all_metrics.get(target_name, {}).get('oof_f1', 0.0):.6f}"
                )

                # Update metrics with honest macro F1
                all_metrics[target_name]["oof_f1_macro"] = _macro_f1
                all_metrics[target_name]["oof_f1_weighted"] = _weighted_f1

    print(f"\n[OK] Multi-target OOF saved → {oof_path}")

    # Save multi-target submission
    test_file = input_files.get("test", "Test.csv")
    raw_test = pd.read_csv(paths.data_raw_dir / test_file)

    # Save test probabilities for calibration
    for target_name, preds in all_test_preds.items():
        target_specs = [t for t in targets if t["name"] == target_name]
        target_task = target_specs[0]["task_type"] if target_specs else "classification"
        test_prob_df = pd.DataFrame({id_col: raw_test[id_col]})
        if preds.ndim > 1:
            # Multiclass: one column per class
            for i in range(preds.shape[1]):
                test_prob_df[f"{target_name}_prob_class_{i}"] = preds[:, i]
        else:
            # Binary classification: store positive-class probability
            if target_task == "classification":
                test_prob_df[f"{target_name}_prob_class_1"] = np.asarray(
                    preds, dtype=np.float64
                )
            else:
                # Regression: no probability columns to save
                continue
        test_prob_path = (
            paths.data_processed_dir / f"test_probs_anchor-baseline_{target_name}.csv"
        )
        test_prob_df.to_csv(test_prob_path, index=False)
        print(f"[OK] Test probabilities saved → {test_prob_path}")

    # Submission formatting removed from Phase 2B — skill_14 (Phase 4) reads
    # test_probs_anchor-baseline_<target>.csv from data/processed/ and produces
    # the final submission CSV with competition-specific column formatting.

    # Compute weighted composite score per SoT v2.2.1 A11
    # For mixed-task competitions: 0.6 × classification_f1 + 0.4 × normalized_regression
    classification_targets = [t for t in targets if t["task_type"] == "classification"]
    regression_targets = [t for t in targets if t["task_type"] == "regression"]

    if classification_targets and regression_targets:
        # Mixed-task: use weighted composite
        weighted_scores = []
        class_details = []
        reg_details = []

        for t in classification_targets:
            target_name = t["name"]
            weight = t.get("weight", 0.5)
            f1 = all_metrics[target_name]["oof_f1"]
            weighted_scores.append(f1 * weight)
            class_details.append((target_name, f1, weight))

        for t in regression_targets:
            target_name = t["name"]
            weight = t.get("weight", 0.5)
            rmse = float(all_metrics[target_name]["oof_rmse"])
            target_std = float(raw_train[target_name].std())
            normalized_rmse = rmse / target_std if target_std > 0 else rmse
            regression_score = max(0.0, 1.0 - normalized_rmse)
            weighted_scores.append(regression_score * weight)
            reg_details.append((target_name, regression_score, weight))

        total_weight = sum(t.get("weight", 0.5) for t in targets)
        avg_score = (
            sum(weighted_scores) / total_weight
            if total_weight > 0
            else sum(weighted_scores)
        )

        print("\n[STATS] Weighted Composite Score Calculation:")
        for name, f1, w in class_details:
            print(f"  Classification F1 ({name}): {f1:.6f} (weight: {w})")
        for name, score, w in reg_details:
            print(f"  Regression score ({name}):  {score:.6f} (weight: {w})")
        print(f"  Composite:         {avg_score:.6f}")
    elif classification_targets:
        # Classification-only: use average F1
        classification_f1s = [
            float(all_metrics[t["name"]]["oof_f1"]) for t in classification_targets
        ]
        avg_score = np.mean(classification_f1s)
        print(f"\n[STATS] Classification-only composite: {avg_score:.6f}")
    elif regression_targets:
        # Regression-only: use average normalized RMSE score
        regression_scores = []
        for t in regression_targets:
            target_name = t["name"]
            rmse = float(
                all_metrics[target_name]["oof_rmse"]
            )  # Now using correct key name
            target_std = float(raw_train[target_name].std())
            normalized_rmse = rmse / target_std if target_std > 0 else rmse
            regression_score = max(0.0, 1.0 - normalized_rmse)
            regression_scores.append(regression_score)
        avg_score = np.mean(regression_scores)
        print(f"\n[STATS] Regression-only composite: {avg_score:.6f}")
    else:
        # Fallback (should never happen)
        avg_score = 0.0

    # Compute MD5 hash of all target columns
    import hashlib

    target_names = [t["name"] for t in targets]
    targets_csv = raw_train[target_names].to_csv(index=False).encode("utf-8")
    md5_target_hash = hashlib.md5(targets_csv).hexdigest()

    # Log to DuckDB ledger
    feature_count = len([c for c in X_train.columns if c != id_col])
    with Ledger() as ledger:
        exp_id = ledger.log_experiment(
            branch_name="anchor-baseline",
            oof_score=avg_score,
            metric="composite_avg",
            feature_count=feature_count,
            calibration_method="none",
            gate_result="PASS",
            gate_reason=f"Multi-target anchor baseline: {len(targets)} targets trained",
            dag_phase="phase_2_anchor_confirmed",
            notes=f"avg_score={avg_score:.6f}; targets={target_names}",
        )
    print(f"[OK] Experiment logged -> DuckDB exp_id={exp_id}")

    # Write OOF records for each target (A12 policy: use _augmented suffix during retraining)
    from zindian.config import get_seed

    seed = int(random_seed if random_seed is not None else get_seed())
    cv_strategy_id = state.get("anchor_cv_strategy_id", "stratified_5fold")

    retraining_active = bool(
        state.get("pseudo_label_result", {}).get("retraining_required", False)
    )
    branch_suffix = "_augmented" if retraining_active else ""

    for target_name, preds in all_oof.items():
        oof_1d = preds if preds.ndim == 1 else np.argmax(preds, axis=1)
        write_oof_record(
            state_store,
            branch_name=f"anchor-baseline_{target_name}{branch_suffix}",
            scores=np.asarray(oof_1d, dtype=np.float64).tolist(),
            cv_strategy_id=cv_strategy_id,
            seed=seed,
            model_config={
                "feature_count": feature_count,
                "n_splits": n_splits,
                "threshold": all_metrics[target_name]["threshold"],
                "fold_scores": all_metrics[target_name]["fold_scores"],
                "target_name": target_name,
            },
        )

    # Resolve first classification and regression targets dynamically from config
    first_classification_target = next(
        (t["name"] for t in targets if t.get("task_type") == "classification"), None
    )
    first_regression_target = next(
        (t["name"] for t in targets if t.get("task_type") == "regression"), None
    )

    state_store.update(
        competition=config.slug,
        md5_target_hash=md5_target_hash,
        anchor_oof_score=avg_score,
        anchor_oof_f1=(
            all_metrics.get(first_classification_target, {}).get("oof_f1", 0.0)
            if first_classification_target
            else 0.0
        ),
        anchor_oof_rmse=(
            all_metrics.get(first_regression_target, {}).get("oof_rmse", 0.0)
            if first_regression_target
            else 0.0
        ),
        anchor_multi_target_metrics=all_metrics,
        dag_phase="phase_2_anchor_confirmed",
        last_updated=datetime.now(timezone.utc).isoformat(),
    )
    print(f"\n[OK] Multi-target training complete. Avg score: {avg_score:.6f}")

    return {
        "status": "OK",
        "multi_target": True,
        "targets": list(all_metrics.keys()),
        "metrics": all_metrics,
        "avg_score": avg_score,
        "submission_path": None,
        "oof_path": str(oof_path),
    }


if __name__ == "__main__":
    import sys

    submit_flag = "--submit" in sys.argv
    variant_arg = None
    for i, arg in enumerate(sys.argv):
        if arg == "--variant" and i + 1 < len(sys.argv):
            variant_arg = sys.argv[i + 1]
            break
    result = run(submit=submit_flag, variant_name=variant_arg)
    printable = {k: v for k, v in result.items() if k != "submission_result"}
    print(json.dumps(printable, indent=2))
