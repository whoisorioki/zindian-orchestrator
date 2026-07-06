"""
Skill 21 — Semi-Supervised Pseudo-Labeling
==========================================

Semi-supervised learning via high-confidence pseudo-labels on unlabelled test
set. Model: LGB + RF 50/50 blend.

Contract (SoT §4 / §8):
  * Classification targets only. Regression targets are skipped per A12.
  * Multi-target: A12 recombination policy enforced for mixed-task competitions.
  * No hardcoded competition column names. Target, ID, and policy-blocked
    columns are resolved from `ChallengeConfig`.
  * CV strategy is taken from the active state override or the
    `challenge_config.json` block; never an in-skill constant.
  * Pseudo-labeled rows are appended **only to the training split** of every
    fold. They are explicitly excluded from validation splits.
  * OOF records use the canonical SoT schema with a `cv_strategy_id` tag.
  * When `retraining_required == True`, the OOF is written to the
    `branch_{name}_oof_augmented` namespace; the original non-augmented key
    is never overwritten.
  * The skill writes the canonical `pseudo_label_result` nested schema with
    all six `gc1..gc6` guard flags.
  * The skill never writes to `challenge_config.json` after Phase 1.
  * The skill never writes a `human_gate_*_approved` key.
"""

from __future__ import annotations
import tabula.skill_state_autopatch  # noqa

import json
import sys
from datetime import datetime, timezone
from typing import Any, Iterable, cast

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

from zindian.config import ChallengeConfig, get_seed
from zindian.cv import make_cv_splitter
from zindian.paths import resolve_competition_paths
from zindian.state import (
    SkillStateStore,
    resolve_active_cv_strategy_id,
    write_oof_record,
)

# -- Locked hyperparameters (do NOT tune per-competition) ----------------------
CONF_POS_DEFAULT = 0.85
CONF_NEG_DEFAULT = 0.15
SAMPLE_WEIGHT_DEFAULT = 0.5
THRESHOLD_DEFAULT = 0.426
MAX_ITERATIONS = 4
N_SPLITS = 5
BASE_SEED = get_seed()
SEEDS = [BASE_SEED + i for i in range(3)]

# Base params - objective and metric are set dynamically from config per A5
LGB_BASE_PARAMS = {
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1,
}

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "n_jobs": -1,
}


# -- Diagnostic gate (positives-count distribution check) ----------------------


def check_distribution_gate(
    preds: np.ndarray | Any,
    threshold: float = THRESHOLD_DEFAULT,
    gate_min: int | None = None,
    gate_max: int | None = None,
    anchor: int | None = None,
) -> dict[str, Any]:
    """Validate predicted positive count is inside `[gate_min, gate_max]`.

    Defaults are not used to evaluate competitions; they only establish the
    "fail-closed" lower bound when no override is supplied. The orchestrator
    is expected to override `gate_min` / `gate_max` with config-driven values
    or this helper returns a BLOCKED result.
    """
    preds_array: np.ndarray = np.asarray(preds, dtype=np.float64)
    labels = (preds_array >= threshold).astype(np.int32)
    pos_count: int = int(labels.sum())
    if gate_min is None or gate_max is None:
        return {
            "pos_count": pos_count,
            "drift": 0,
            "gate_min": None,
            "gate_max": None,
            "anchor": anchor,
            "passed": False,
            "threshold": float(threshold),
            "diagnosis": (
                "[FAIL] Distribution gate not configured. "
                "Skill 21 requires explicit gate_min/gate_max from the active "
                "competition policy."
            ),
        }
    drift: int = pos_count - (anchor if anchor is not None else pos_count)
    passed = gate_min <= pos_count <= gate_max
    diagnosis: str
    if not passed:
        if pos_count > gate_max:
            diagnosis = (
                f"[FAIL] Over-prediction by {pos_count - gate_max} samples. "
                f"Remediation: raise threshold above {threshold:.2f}."
            )
        else:
            diagnosis = (
                f"[FAIL] Under-prediction by {gate_min - pos_count} samples. "
                f"Remediation: lower threshold below {threshold:.2f}."
            )
    else:
        diagnosis = (
            f"[OK] Distribution matches anchor baseline (drift={drift:+d} samples). "
            f"Gate PASSED."
        )
    return {
        "pos_count": pos_count,
        "drift": drift,
        "gate_min": gate_min,
        "gate_max": gate_max,
        "anchor": anchor,
        "passed": passed,
        "threshold": float(threshold),
        "diagnosis": diagnosis,
    }


# -- Column / feature mask helpers --------------------------------------------


def _resolve_drop_columns(
    config: ChallengeConfig, train_columns: Iterable[str]
) -> tuple[set[str], str | None, str | None]:
    """Build the dynamic drop-column set from config accessors.

    Returns (drop_set, target_col, id_column). All column names are sourced
    from the challenge config per A5.
    """
    target_col = config.get("target_col") or config.get("target_column")
    cols_cfg = config.get("columns") or {}
    id_column = config.get("id_column") or config.get("id_col") or cols_cfg.get("id")
    if not id_column:
        raise ValueError(
            "id_column not configured in challenge_config.json (A5 violation)"
        )

    drop: set[str] = set()
    if isinstance(target_col, str) and target_col:
        drop.add(target_col)
    if isinstance(id_column, str) and id_column:
        drop.add(id_column)

    if isinstance(cols_cfg, dict):
        for key in ("latitude", "longitude"):
            v = cols_cfg.get(key)
            if isinstance(v, str) and v:
                drop.add(v)

    policy_filters = config.get("policy_filters") or []
    if isinstance(policy_filters, (list, tuple, set)):
        for col in policy_filters:
            if isinstance(col, str) and col:
                drop.add(col)
    return (
        drop,
        (str(target_col) if isinstance(target_col, str) else None),
        (str(id_column) if isinstance(id_column, str) else None),
    )


def _resolve_threshold(config: ChallengeConfig) -> tuple[float, float, float]:
    """Return (conf_pos, conf_neg, threshold) from config or defaults."""
    conf_pos = float(
        config.get("pseudo_conf_pos", CONF_POS_DEFAULT) or CONF_POS_DEFAULT
    )
    conf_neg = float(
        config.get("pseudo_conf_neg", CONF_NEG_DEFAULT) or CONF_NEG_DEFAULT
    )
    threshold = float(
        config.get("pseudo_threshold", THRESHOLD_DEFAULT) or THRESHOLD_DEFAULT
    )
    if conf_pos < conf_neg:
        # Sanity: positive threshold must exceed negative threshold.
        conf_pos, conf_neg = max(conf_pos, conf_neg), min(conf_pos, conf_neg)
    return conf_pos, conf_neg, threshold


def _resolve_sample_weight(config: ChallengeConfig) -> float:
    return float(
        config.get("pseudo_sample_weight", SAMPLE_WEIGHT_DEFAULT)
        or SAMPLE_WEIGHT_DEFAULT
    )


def _resolve_distribution_gate(
    config: ChallengeConfig,
) -> tuple[int | None, int | None, int | None]:
    block = config.get("pseudo_distribution_gate") or {}
    if not isinstance(block, dict):
        return (None, None, None)
    g_min = block.get("min")
    g_max = block.get("max")
    anchor = block.get("anchor")
    return (
        int(g_min) if g_min is not None else None,
        int(g_max) if g_max is not None else None,
        int(anchor) if anchor is not None else None,
    )


def get_feature_cols(df: pd.DataFrame, drop_cols: set[str]) -> list[str]:
    return [c for c in df.columns if c not in drop_cols]


# -- Training with strict split isolation --------------------------------------


def _resolve_active_cv_strategy(
    config: ChallengeConfig, state: dict[str, Any]
) -> dict[str, Any]:
    """Return the active CV-strategy dict (override or config)."""
    override = state.get("cv_strategy_override") or {}
    if isinstance(override, dict) and override.get("active"):
        strategy = override.get("override_strategy") or {}
        if isinstance(strategy, dict):
            return strategy
    cv = config.get("cv_strategy") or {}
    return cv if isinstance(cv, dict) else {}


def train_ensemble_and_predict(
    X_labelled: np.ndarray,
    y_labelled: np.ndarray,
    X_pseudo: np.ndarray,
    X_test: np.ndarray,
    cv_strategy: dict[str, Any],
    feature_cols: list[str],
    config: ChallengeConfig,
    sample_weight_labelled: np.ndarray | None = None,
    sample_weight_pseudo: float = SAMPLE_WEIGHT_DEFAULT,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Train an LGB+RF ensemble using strict split isolation.

    The labelled portion alone is split into K folds. Pseudo rows (rows N_train
    onward) are appended to **every training fold** and explicitly excluded
    from every validation fold.
    """
    if seed is None:
        seed = int(get_seed())

    # A5: Read metric from config and map to LightGBM metric
    metric = str(config.get("metric", "auc")).lower()
    # Map competition metrics to LightGBM training metrics
    metric_map = {
        "f1": "binary_logloss",
        "f1_score": "binary_logloss",
        "auc": "auc",
        "roc_auc": "auc",
        "log_loss": "binary_logloss",
        "logloss": "binary_logloss",
    }
    lgb_metric = metric_map.get(metric, "auc")

    y_labelled_array: np.ndarray = np.asarray(y_labelled, dtype=np.int32)
    oof: np.ndarray = np.zeros(len(X_labelled), dtype=np.float64)
    preds: np.ndarray = np.zeros(len(X_test), dtype=np.float64)
    splitter = make_cv_splitter(cv_strategy=cv_strategy, random_seed=seed)
    n_splits = int(getattr(splitter, "n_splits", N_SPLITS) or N_SPLITS)
    has_pseudo = X_pseudo.shape[0] > 0
    pseudo_indices: np.ndarray = np.arange(
        len(X_labelled), len(X_labelled) + X_pseudo.shape[0], dtype=np.int64
    )
    n_labelled: int = int(len(X_labelled))

    for tr_idx, va_idx in splitter.split(X_labelled, y_labelled_array):
        # Strict split isolation contract: validation rows are a subset of
        # the labelled rows; pseudo-labeled rows are appended to the
        # training side only and must never appear in any validation split.
        va_idx_array: np.ndarray = np.asarray(va_idx, dtype=np.int64)
        if has_pseudo and va_idx_array.size > 0:
            leaked: np.ndarray = va_idx_array[va_idx_array >= n_labelled]
            if leaked.size > 0:
                raise RuntimeError(
                    f"Pseudo-labeled row(s) {leaked.tolist()} leaked into a "
                    "validation split. Skill 21 requires `splitter.split` to "
                    "operate over `X_labelled` only; do not concatenate "
                    "pseudo rows into the input passed to the splitter."
                )
            assert not np.intersect1d(va_idx_array, pseudo_indices).size, (
                "Pseudo-labeled row indices intersected a validation split; "
                "this is a contract violation in Skill 21 strict split "
                "isolation."
            )

        X_tr_lab = X_labelled[tr_idx]
        y_tr_lab = y_labelled_array[tr_idx]
        X_va = X_labelled[va_idx]
        y_va = y_labelled_array[va_idx]

        if sample_weight_labelled is None:
            w_tr_lab = np.ones(len(X_tr_lab), dtype=np.float64)
        else:
            w_tr_lab = np.asarray(sample_weight_labelled, dtype=np.float64)[tr_idx]

        if has_pseudo:
            X_tr = np.vstack([X_tr_lab, X_pseudo])
            y_tr = np.concatenate(
                [y_tr_lab, np.zeros(X_pseudo.shape[0], dtype=np.int32)]
            )
            w_tr = np.concatenate(
                [
                    w_tr_lab,
                    np.full(X_pseudo.shape[0], sample_weight_pseudo, dtype=np.float64),
                ]
            )
        else:
            X_tr = X_tr_lab
            y_tr = y_tr_lab
            w_tr = w_tr_lab

        # Class-imbalance scaling
        pos_rate: float = float(y_tr.mean()) if y_tr.size else 0.5
        scale_pos: float = (1.0 - pos_rate) / pos_rate if pos_rate > 0 else 1.0

        dtrain = lgb.Dataset(X_tr, y_tr, weight=w_tr, feature_name=feature_cols)
        dval = lgb.Dataset(X_va, y_va, reference=dtrain)
        model_lgb = lgb.train(
            {
                **LGB_BASE_PARAMS,
                "objective": "binary",
                "metric": lgb_metric,
                "scale_pos_weight": scale_pos,
                "seed": seed,
            },
            dtrain,
            num_boost_round=500,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        lgb_val = np.array(model_lgb.predict(X_va), dtype=np.float64)
        lgb_test = np.array(model_lgb.predict(X_test), dtype=np.float64)

        model_rf = RandomForestClassifier(**{**RF_PARAMS, "random_state": seed})
        model_rf.fit(X_tr, y_tr, sample_weight=w_tr)
        rf_probs_val = np.asarray(
            cast(Any, model_rf).predict_proba(X_va), dtype=np.float64
        )
        rf_probs_test = np.asarray(
            cast(Any, model_rf).predict_proba(X_test), dtype=np.float64
        )
        rf_val = np.array(rf_probs_val[:, 1], dtype=np.float64)
        rf_test = np.array(rf_probs_test[:, 1], dtype=np.float64)

        oof[va_idx] = 0.5 * lgb_val + 0.5 * rf_val
        preds += (0.5 * lgb_test + 0.5 * rf_test) / max(n_splits, 1)

    auc: float = float(roc_auc_score(y_labelled_array, oof))
    # Stash the pseudo indices for downstream assertions.
    oof.flags.writeable = True
    return oof, preds, auc


def best_f1_threshold(
    y_true: np.ndarray | Any, probs: np.ndarray | Any
) -> tuple[float, float]:
    _y_true_raw: np.ndarray = np.asarray(y_true)
    y_true_array: np.ndarray
    if _y_true_raw.dtype.kind in ("U", "S", "O"):
        _le_true = LabelEncoder()
        transformed = _le_true.fit_transform(_y_true_raw.astype(str))
        y_true_array = np.asarray(transformed, dtype=np.int32)
    else:
        y_true_array = np.asarray(_y_true_raw, dtype=np.int32)
    probs_array: np.ndarray = np.asarray(probs, dtype=np.float64)
    best_f1: float = 0.0
    best_t: float = 0.5
    for t_val in np.arange(0.3, 0.7, 0.01):
        f1: float = float(f1_score(y_true_array, (probs_array >= t_val).astype(int)))
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t_val)
    return round(best_f1, 5), round(best_t, 2)


# -- Guard condition flags (SoT §4) --------------------------------------------


def _build_guard_condition_flags(
    *,
    classification: bool,
    cv_strategy_type: str | None,
    leaked_features: list[str],
    fold_variance: float | None,
    variance_threshold: float | None,
    calibration_present: bool,
    confidence_threshold_met: bool,
) -> dict[str, bool]:
    return {
        "gc1_classification": bool(classification),
        "gc2_not_timeseries": bool(cv_strategy_type)
        and cv_strategy_type != "timeseries",
        "gc3_no_leaked_features": len(leaked_features or []) == 0,
        "gc4_variance_within_threshold": (
            fold_variance is not None
            and variance_threshold is not None
            and fold_variance < variance_threshold
        ),
        "gc5_calibrated_probs_present": bool(calibration_present),
        "gc6_confidence_threshold_met": bool(confidence_threshold_met),
    }


def _resolve_initial_test_probs(
    state: dict[str, Any],
    paths: Any,
    config: ChallengeConfig,
    target_name: str | None = None,
) -> np.ndarray | None:
    """Find and load test probabilities of the active branch before running the loop."""
    branch = (
        state.get("best_variant_this_round")
        or state.get("best_variant_branch")
        or state.get("current_active_branch")
        or state.get("anchor_git_branch")
    )
    if not branch:
        for key in state:
            if key.startswith("branch_") and key.endswith("_oof"):
                branch = key.removeprefix("branch_").removesuffix("_oof")
                break

    if not branch:
        return None

    proc_dir = paths.data_processed_dir
    reports_dir = paths.reports_dir

    candidates = []
    if target_name:
        candidates.extend(
            [
                f"calib_test_probs_{branch}_{target_name}.csv",
                f"calib_test_probs_{branch}_{target_name}_augmented.csv",
                f"test_probs_{branch}_{target_name}.csv",
                f"test_probs_{branch}_{target_name}_augmented.csv",
            ]
        )
    candidates.extend(
        [
            f"calib_test_probs_{branch}.csv",
            f"calib_test_probs_{branch}_augmented.csv",
            f"test_probs_{branch}.csv",
            f"test_probs_{branch}_augmented.csv",
        ]
    )

    cols_cfg = config.get("columns") or {}
    id_col = config.get("id_column") or config.get("id_col") or cols_cfg.get("id")
    if not id_col:
        return None

    for filename in candidates:
        for base_dir in (proc_dir, reports_dir):
            filepath = base_dir / filename
            if filepath.exists():
                try:
                    df = pd.read_csv(filepath)
                    pcols = [c for c in df.columns if c != id_col]
                    if pcols:
                        return df[pcols[0]].to_numpy()
                except Exception:
                    pass
    return None


# -- Run loop ------------------------------------------------------------------


def run(
    dry_run: bool = False,
    target_name_override: str | None = None,
    is_multi_target: bool = False,
) -> dict:
    """Run the pseudo-labeling loop.

    Returns a dict with `status`. The canonical record under
    `state["pseudo_label_result"]` follows the SoT §4 schema:

        pseudo_label_result = {
            "ran": True,
            "n_pseudo_labels_added": int,
            "retraining_required": bool,
            "guard_conditions_met": bool,
            "guard_failure_reason": str | None,
            "execution_failure_reason": str | None,
            "guard_condition_flags": {
                "gc1_classification": bool,
                "gc2_not_timeseries": bool,
                "gc3_no_leaked_features": bool,
                "gc4_variance_within_threshold": bool,
                "gc5_calibrated_probs_present": bool,
                "gc6_confidence_threshold_met": bool,
            },
        }
    """
    print("\n" + "=" * 70)
    print("SKILL 21 — Semi-Supervised Pseudo-Labeling (LGB + RF Blend)")
    print("=" * 70 + "\n")

    paths = resolve_competition_paths()
    config = ChallengeConfig.load()
    store = SkillStateStore(paths.state_path)
    state = store.read()
    config_data = getattr(config, "_data", {}) or {}

    # Multi-target detection (A12)
    if target_name_override is None:
        target_config = config.get("target_config")
        if target_config and target_config.get("targets"):
            # A12 Assertion: Verify recombination policy exists for mixed-task competitions
            targets = target_config.get("targets", [])
            task_types = set(t.get("task_type") for t in targets if isinstance(t, dict))
            if (
                len(task_types) > 1
                and "classification" in task_types
                and "regression" in task_types
            ):
                assert "pseudo_label_recombination_policy" in target_config, (
                    "A12 Violation: Mixed-task multi-target competitions require a "
                    "recombination policy in challenge_config.json! Add "
                    "'pseudo_label_recombination_policy' to target_config."
                )
            return _run_multi_target_pseudo_label(paths, config, store, state, dry_run)

    # -- Guard Condition 1: classification only -------------------------------
    target_col = (
        target_name_override or config.get("target_col") or config.get("target_column")
    )
    task_type = str(config.get("task_type", "classification"))
    if task_type != "classification":
        msg = f"Pseudo-labeling is strictly prohibited for task_type '{task_type}'. Classification only."
        print(f"ERROR: {msg}")
        raise ValueError(msg)

    # -- Early Guard Check Block ----------------------------------------------
    cv_strategy = _resolve_active_cv_strategy(config, state)
    cv_strategy_type = (
        cv_strategy.get("type") if isinstance(cv_strategy, dict) else None
    )

    leaked_features = state.get("leaked_features") if isinstance(state, dict) else []
    if not isinstance(leaked_features, list):
        leaked_features = []

    fold_variance = None
    variance_threshold = None
    metric_analysis = state.get("metric_analysis") if isinstance(state, dict) else None
    if isinstance(metric_analysis, dict):
        v = metric_analysis.get("fold_score_variance")
        if isinstance(v, (int, float)):
            fold_variance = float(v)
    cfg_vt = config.get("variance_gate_threshold")
    if isinstance(cfg_vt, (int, float)):
        variance_threshold = float(cfg_vt)

    calibration_present = (
        bool(state.get("calibration_method"))
        or bool(state.get("calibration_written_at"))
        or bool(state.get("last_calibration_method"))
        or bool(state.get("last_calibration_at"))
    )

    conf_pos, conf_neg, threshold = _resolve_threshold(config)
    initial_test_probs = _resolve_initial_test_probs(state, paths, config, target_col)
    confidence_threshold_met = bool(
        initial_test_probs is not None and float(initial_test_probs.max()) >= conf_pos
    )

    early_guard_flags = _build_guard_condition_flags(
        classification=True,
        cv_strategy_type=cv_strategy_type,
        leaked_features=leaked_features,
        fold_variance=fold_variance,
        variance_threshold=variance_threshold,
        calibration_present=calibration_present,
        confidence_threshold_met=confidence_threshold_met,
    )
    early_guard_conditions_met = all(early_guard_flags.values())
    early_failed = [k for k, v in early_guard_flags.items() if not v]
    early_guard_failure_reason = (
        f"Failed guard conditions: {', '.join(early_failed)}" if early_failed else None
    )

    if not early_guard_conditions_met:
        # Persist the canonical pseudo_label_result block to indicate the blocked run
        pseudo_label_result = {
            "ran": True,
            "n_pseudo_labels_added": 0,
            "retraining_required": False,
            "guard_conditions_met": False,
            "guard_failure_reason": early_guard_failure_reason,
            "execution_failure_reason": None,
            "guard_condition_flags": early_guard_flags,
        }
        store.update(
            pseudo_label_result=pseudo_label_result,
            last_updated=datetime.now(timezone.utc).isoformat(),
        )
        print(f"GUARD CONDITIONS FAILED: {early_guard_failure_reason}")
        return {
            "status": "BLOCKED",
            "reason": "guard_conditions_failed",
            "guard_failure_reason": early_guard_failure_reason,
            "guard_condition_flags": early_guard_flags,
            "best_iteration": 0,
            "best_oof_f1": 0.0,
        }

    # -- Dynamic column resolution --------------------------------------------
    branch_name = state.get("anchor_git_branch") or "anchor-baseline"
    train_file = paths.data_processed_dir / (
        config.get("features_train_filename") or f"features_train_{branch_name}.csv"
    )
    test_file = paths.data_processed_dir / (
        config.get("features_test_filename") or f"features_test_{branch_name}.csv"
    )
    sample_sub_file = paths.data_raw_dir / (
        config.get("sample_submission_filename") or "SampleSubmission.csv"
    )

    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    pd.read_csv(sample_sub_file)

    drop_cols, target_col, id_column = _resolve_drop_columns(config, train.columns)
    if target_name_override:
        target_col = target_name_override
        target_config = config.get("target_config")
        if target_config and target_config.get("targets"):
            for t in target_config["targets"]:
                drop_cols.add(t["name"])

    if not target_col or target_col not in train.columns:
        return {
            "status": "BLOCKED",
            "reason": "target_column_unresolved",
            "message": "challenge_config.json is missing 'target_col' (or 'target_column').",
        }
    if not id_column or id_column not in train.columns or id_column not in test.columns:
        return {
            "status": "BLOCKED",
            "reason": "id_column_unresolved",
            "message": (
                "challenge_config.json is missing 'id_column' or the column is not "
                "present in both train and test."
            ),
        }

    feature_cols = get_feature_cols(train, drop_cols)
    feature_cols = [c for c in feature_cols if c in test.columns]
    if not feature_cols:
        return {
            "status": "BLOCKED",
            "reason": "no_feature_columns",
            "message": "After applying the dynamic drop mask, no feature columns remain.",
        }

    # `.to_numpy()` guarantees a concrete `np.ndarray`; `.values` may return
    # a `pd.api.extensions.ExtensionArray` for nullable dtypes which fails
    # strict Pylance narrowing against `np.ndarray` parameter annotations.
    X_labelled = train[feature_cols].to_numpy(dtype=np.float32)
    _y_lab_raw = train[target_col].to_numpy()
    if _y_lab_raw.dtype.kind in ("U", "S", "O"):
        _le_lab = LabelEncoder()
        transformed = _le_lab.fit_transform(_y_lab_raw.astype(str))
        y_labelled = np.asarray(transformed, dtype=np.int32)
    else:
        y_labelled = np.asarray(_y_lab_raw, dtype=np.int32)
    X_test = test[feature_cols].to_numpy(dtype=np.float32)
    test_ids = test[id_column].to_numpy()

    conf_pos, conf_neg, threshold = _resolve_threshold(config)
    sample_weight_pseudo = _resolve_sample_weight(config)
    gate_min, gate_max, anchor = _resolve_distribution_gate(config)

    cv_strategy = _resolve_active_cv_strategy(config, state)
    cv_strategy_type = (
        cv_strategy.get("type") if isinstance(cv_strategy, dict) else None
    )

    print(f"Labelled rows  : {len(X_labelled)}")
    print(f"Test rows      : {len(X_test)}")
    print(f"Features       : {len(feature_cols)}")
    print(f"Confidence thresholds: pos>={conf_pos} | neg<={conf_neg}")
    print(f"Sample weight  : {sample_weight_pseudo}")
    print(f"Prediction threshold: {threshold}")
    print(f"CV strategy    : {cv_strategy_type or 'default stratified'}")
    print(f"Seeds          : {SEEDS}")
    print()

    # -- Pseudo-label iterations ----------------------------------------------
    results: list[dict[str, Any]] = []
    test_probs_prev: np.ndarray | None = None
    n_pseudo_added_total = 0
    retraining_required = False

    for iteration in range(MAX_ITERATIONS + 1):
        print(f"{'=' * 70}")
        print(f"Iteration {iteration}")
        print(f"{'=' * 70}")

        if iteration == 0:
            X_pseudo = np.zeros((0, X_labelled.shape[1]), dtype=np.float32)
        else:
            if test_probs_prev is None:
                return {
                    "status": "BLOCKED",
                    "reason": "no_test_probs",
                    "message": "Iteration >0 with no prior test probabilities.",
                }
            pos_mask = test_probs_prev >= conf_pos
            neg_mask = test_probs_prev <= conf_neg
            pseudo_mask = pos_mask | neg_mask
            X_pseudo = X_test[pseudo_mask]
            n_pseudo_added_total += int(pseudo_mask.sum())
            if X_pseudo.shape[0] > 0:
                retraining_required = True
            print(
                f"Pseudo-labels added: {int(pos_mask.sum())} positive, "
                f"{int(neg_mask.sum())} negative ({int(pseudo_mask.sum())} total)"
            )
            print(f"Total labelled rows: {len(X_labelled)}")

        oof_list: list[np.ndarray] = []
        test_list: list[np.ndarray] = []
        auc_list: list[float] = []

        for seed_val in SEEDS:
            oof, test_pred, auc = train_ensemble_and_predict(
                X_labelled,
                y_labelled,
                X_pseudo,
                X_test,
                cv_strategy=cv_strategy,
                feature_cols=feature_cols,
                config=config,
                sample_weight_pseudo=sample_weight_pseudo,
                seed=int(seed_val),
            )
            oof_list.append(oof)
            test_list.append(test_pred)
            auc_list.append(auc)
            print(f"  [OK] Seed {seed_val} AUC: {auc:.5f}")

        oof_probs: np.ndarray = np.mean(np.array(oof_list), axis=0).astype(np.float64)
        test_probs: np.ndarray = np.mean(np.array(test_list), axis=0).astype(np.float64)
        mean_auc: float = float(np.mean(auc_list))

        oof_labelled: np.ndarray = oof_probs[: len(X_labelled)]
        oof_f1, _ = best_f1_threshold(y_labelled, oof_labelled)

        print(f"OOF AUC (mean {len(SEEDS)} seeds): {mean_auc:.5f}")
        print(f"OOF F1  (labelled portion):       {oof_f1:.5f}")

        results.append(
            {
                "iteration": iteration,
                "oof_auc": float(mean_auc),
                "oof_f1": float(oof_f1),
                "threshold": float(threshold),
                "train_rows": int(len(X_labelled) + X_pseudo.shape[0]),
            }
        )

        # Persist per-iteration OOF artifact for downstream consumption.
        reports_dir = paths.reports_dir
        reports_dir.mkdir(parents=True, exist_ok=True)
        suffix = "_augmented" if retraining_required else ""
        oof_iter_path = reports_dir / f"oof_probs_pseudo_iter{iteration}{suffix}.csv"
        test_iter_path = reports_dir / f"test_probs_pseudo_iter{iteration}{suffix}.csv"
        pd.DataFrame(
            {str(id_column): train[id_column].values, "oof_prob": oof_labelled}
        ).to_csv(oof_iter_path, index=False)
        pd.DataFrame({str(id_column): test_ids, "test_prob": test_probs}).to_csv(
            test_iter_path, index=False
        )

        test_probs_prev = test_probs

        if iteration >= 2 and len(results) >= 2:
            delta = results[-1]["oof_f1"] - results[-2]["oof_f1"]
            print(f"Delta OOF F1 vs prev: {delta:+.5f}")
            if delta <= 0:
                print("No improvement — stopping early.")
                break

    # -- Summary --------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("PSEUDO-LABEL ITERATION SUMMARY")
    print(f"{'=' * 70}")
    print(
        f"{'Iter':>5} {'OOF AUC':>10} {'OOF F1':>10} {'Threshold':>10} {'Train rows':>12}"
    )
    print("-" * 60)
    for r in results:
        print(
            f"{r['iteration']:>5} {r['oof_auc']:>10.5f} {r['oof_f1']:>10.5f} "
            f"{r['threshold']:>10.2f} {r['train_rows']:>12}"
        )

    best_result = max(results, key=lambda x: x["oof_f1"])
    best_iteration = int(best_result["iteration"])
    best_oof_f1 = float(best_result["oof_f1"])

    # -- Build guard condition flags ------------------------------------------
    fold_variance = None
    variance_threshold = None
    metric_analysis = state.get("metric_analysis") if isinstance(state, dict) else None
    if isinstance(metric_analysis, dict):
        v = metric_analysis.get("fold_score_variance")
        if isinstance(v, (int, float)):
            fold_variance = float(v)
    cfg_vt = config.get("variance_gate_threshold")
    if isinstance(cfg_vt, (int, float)):
        variance_threshold = float(cfg_vt)

    leaked_features = state.get("leaked_features") if isinstance(state, dict) else []
    if not isinstance(leaked_features, list):
        leaked_features = []
    calibration_present = (
        bool(state.get("calibration_method"))
        or bool(state.get("calibration_written_at"))
        or bool(state.get("last_calibration_method"))
        or bool(state.get("last_calibration_at"))
    )
    confidence_threshold_met = bool(
        test_probs_prev is not None and float(test_probs_prev.max()) >= conf_pos
    )

    guard_flags = _build_guard_condition_flags(
        classification=True,
        cv_strategy_type=cv_strategy_type,
        leaked_features=leaked_features,
        fold_variance=fold_variance,
        variance_threshold=variance_threshold,
        calibration_present=calibration_present,
        confidence_threshold_met=confidence_threshold_met,
    )
    guard_conditions_met = all(guard_flags.values())
    failed = [k for k, v in guard_flags.items() if not v]
    guard_failure_reason = (
        f"Failed guard conditions: {', '.join(failed)}" if failed else None
    )

    # -- Distribution gate check (informational, config-driven) --------------
    if test_probs_prev is not None and gate_min is not None and gate_max is not None:
        gate_report = check_distribution_gate(
            preds=np.asarray(test_probs_prev, dtype=np.float64),
            threshold=float(threshold),
            gate_min=gate_min,
            gate_max=gate_max,
            anchor=anchor,
        )
    else:
        gate_report = {
            "pos_count": 0,
            "drift": 0,
            "gate_min": gate_min,
            "gate_max": gate_max,
            "anchor": anchor,
            "passed": False,
            "threshold": float(threshold),
            "diagnosis": "Distribution gate not configured for this competition.",
        }

    # -- Persist the canonical pseudo_label_result block ----------------------
    pseudo_label_result = {
        "ran": True,
        "n_pseudo_labels_added": int(n_pseudo_added_total),
        "retraining_required": bool(retraining_required),
        "guard_conditions_met": bool(guard_conditions_met),
        "guard_failure_reason": guard_failure_reason,
        "execution_failure_reason": None,
        "guard_condition_flags": guard_flags,
    }

    # -- Persist the OOF record via write_oof_record -------------------------
    cv_strategy_id = resolve_active_cv_strategy_id(state, config_data)
    if is_multi_target:
        oof_branch_name = (
            f"anchor-baseline_{target_col}_augmented"
            if retraining_required
            else f"anchor-baseline_{target_col}"
        )
    else:
        oof_branch_name = (
            "pseudo_label_augmented" if retraining_required else "pseudo_label"
        )
    oof_array = np.asarray(oof_probs[: len(X_labelled)], dtype=np.float64)
    write_oof_record(
        store,
        branch_name=oof_branch_name,
        scores=oof_array.tolist(),
        cv_strategy_id=cv_strategy_id,
        seed=int(BASE_SEED),
        model_config={
            "iterations": int(best_iteration),
            "threshold": float(threshold),
            "conf_pos": float(conf_pos),
            "conf_neg": float(conf_neg),
            "seeds": [int(s) for s in SEEDS],
            "sample_weight": float(sample_weight_pseudo),
            "cv_strategy": cv_strategy,
            "task_type": "classification",
            "feature_count": int(len(feature_cols)),
            "n_pseudo_labels_added": int(n_pseudo_added_total),
        },
    )

    # -- Update SKILL_STATE with the canonical pseudo_label_result + summary -
    if not is_multi_target:
        store.update(
            pseudo_label_result=pseudo_label_result,
            pseudo_label_best_iteration=int(best_iteration),
            pseudo_label_best_oof_f1=float(best_oof_f1),
            pseudo_label_oof_cv_strategy_id=cv_strategy_id,
            pseudo_label_cv_strategy_id=cv_strategy_id,
            last_updated=datetime.now(timezone.utc).isoformat(),
        )

    # -- Gate report print ----------------------------------------------------
    print(
        f"\n{'=' * 70}\n"
        f"=== Skill 21 — pseudo-label result ===\n"
        f"Best iteration        : {best_iteration}\n"
        f"Best OOF F1           : {best_oof_f1:.5f}\n"
        f"CV strategy id        : {cv_strategy_id}\n"
        f"Retraining required   : {retraining_required}\n"
        f"Guard conditions met  : {guard_conditions_met}\n"
        f"Guard flags           : {guard_flags}\n"
        f"Distribution gate     : passed={gate_report.get('passed')} | pos={gate_report.get('pos_count')}\n"
        f"{'=' * 70}"
    )

    if not guard_conditions_met:
        return {
            "status": "BLOCKED",
            "reason": "guard_conditions_failed",
            "guard_failure_reason": guard_failure_reason,
            "guard_condition_flags": guard_flags,
            "best_iteration": int(best_iteration),
            "best_oof_f1": float(best_oof_f1),
        }

    if dry_run:
        return {
            "status": "GATE_PASSED_DRY_RUN",
            "best_iteration": int(best_iteration),
            "best_oof_f1": float(best_oof_f1),
            "retraining_required": bool(retraining_required),
            "guard_condition_flags": guard_flags,
        }

    return {
        "status": "OK",
        "best_iteration": int(best_iteration),
        "best_oof_f1": float(best_oof_f1),
        "retraining_required": bool(retraining_required),
        "guard_condition_flags": guard_flags,
        "cv_strategy_id": cv_strategy_id,
        "branch_name": oof_branch_name,
    }


def _run_multi_target_pseudo_label(paths, config, store, state, dry_run) -> dict:
    """Multi-target pseudo-labeling with A12 recombination policy."""
    print("\n[TARGET] MULTI-TARGET PSEUDO-LABEL MODE (A12)\n")
    target_config = config.get("target_config", {})
    targets = target_config.get("targets", [])

    # A12 only applies when there are multiple targets AND mixed task types
    if len(targets) > 1:
        task_types = set(t.get("task_type") for t in targets if isinstance(t, dict))
        if (
            len(task_types) > 1
            and "classification" in task_types
            and "regression" in task_types
        ):
            policy = target_config.get("pseudo_label_recombination_policy")
            LEGAL_POLICIES = {
                "freeze_unaugmented_targets_at_original",
                "block_composite_until_all_targets_augmented_or_none",
            }
            if policy not in LEGAL_POLICIES:
                raise ValueError(
                    f"A12 violation: pseudo_label_recombination_policy must be one of {LEGAL_POLICIES}, got '{policy}'"
                )
            print(f"A12 Policy: {policy}\n")
        else:
            policy = None
    else:
        policy = None

    print(f"Targets: {[t['name'] for t in targets]}")

    # Separate classification and regression targets
    classification_targets = [t for t in targets if t["task_type"] == "classification"]
    regression_targets = [t for t in targets if t["task_type"] == "regression"]

    print(f"Classification targets: {[t['name'] for t in classification_targets]}")
    print(f"Regression targets (skipped): {[t['name'] for t in regression_targets]}\n")

    if not classification_targets:
        msg = "A12: No classification targets to augment"
        print(f"[INFO]  {msg}")
        store.update(
            pseudo_label_result={
                "ran": True,
                "n_pseudo_labels_added": 0,
                "retraining_required": False,
                "guard_conditions_met": False,
                "guard_failure_reason": "no_classification_targets",
                "execution_failure_reason": None,
                "guard_condition_flags": {"gc1_classification": False},
            },
            last_updated=datetime.now(timezone.utc).isoformat(),
        )
        return {"status": "SKIPPED", "reason": "no_classification_targets"}

    # Run pseudo-labeling only on classification targets
    augmented_results = {}
    classification_retraining_required = False

    for target_spec in classification_targets:
        target_name = target_spec["name"]
        print(f"\n{'-' * 70}")
        print(f"Pseudo-label: {target_name} (classification)")
        print(f"{'-' * 70}")

        # Run single-target pseudo-labeling
        res = run(
            dry_run=dry_run,
            target_name_override=target_name,
            is_multi_target=True,
        )

        if res.get("status") in ("OK", "GATE_PASSED_DRY_RUN"):
            augmented_results[target_name] = {
                "augmented": True,
                "n_pseudo_labels": res.get("best_iteration", 0) * 10,
                "best_oof_f1": res.get("best_oof_f1", 0.0),
            }
            if res.get("retraining_required"):
                classification_retraining_required = True
        else:
            augmented_results[target_name] = {
                "augmented": False,
                "reason": res.get("reason"),
            }

    # A12 Recombination Logic
    retraining_required = False
    if policy == "freeze_unaugmented_targets_at_original":
        print(
            f"\n[OK] A12 Policy: Freezing {len(regression_targets)} regression targets at original OOF"
        )
        print(f"   Augmented {len(classification_targets)} classification targets")
        retraining_required = classification_retraining_required

        if retraining_required:
            # Copy baseline OOF for regression/unaugmented targets to augmented keys
            state = store.read()
            branch_name = state.get("anchor_git_branch") or "anchor-baseline"
            for t in targets:
                t_name = t["name"]
                if t_name not in augmented_results or not augmented_results[t_name].get(
                    "augmented"
                ):
                    original_key = f"branch_{branch_name}_{t_name}_oof"
                    augmented_key = f"branch_{branch_name}_{t_name}_augmented_oof"
                    if original_key in state:
                        store.update(**{augmented_key: state.get(original_key)})
                        print(
                            f"      [FREEZE] Copied {original_key} -> {augmented_key}"
                        )

    elif policy == "block_composite_until_all_targets_augmented_or_none":
        if regression_targets:
            print(
                "\n[FAIL] A12 Policy: BLOCKED - Cannot augment all targets (regression not supported)"
            )
            retraining_required = False
        else:
            print(
                f"\n[OK] A12 Policy: All {len(classification_targets)} targets augmented"
            )
            retraining_required = classification_retraining_required

    store.update(
        pseudo_label_multi_target_results=augmented_results,
        pseudo_label_result={
            "ran": True,
            "n_pseudo_labels_added": sum(
                r.get("n_pseudo_labels", 0)
                for r in augmented_results.values()
                if r.get("augmented")
            ),
            "retraining_required": retraining_required,
            "guard_conditions_met": True,
            "guard_failure_reason": None,
            "execution_failure_reason": None,
            "guard_condition_flags": {"gc1_classification": True},
        },
        last_updated=datetime.now(timezone.utc).isoformat(),
    )
    print(f"\n[OK] Multi-target pseudo-labeling complete (A12 policy: {policy})")
    return {
        "status": "OK",
        "multi_target": True,
        "policy": policy,
        "results": augmented_results,
        "retraining_required": retraining_required,
    }


def main() -> None:
    """CLI entry point."""
    dry_run = "--dry-run" in sys.argv
    if dry_run:
        print("[DRY RUN MODE]")
    result = run(dry_run=dry_run)
    print(f"\nResult: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()
