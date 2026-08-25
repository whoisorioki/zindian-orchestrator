"""
Skill 11 — Branch Gate
Promotes the best passing variant to new anchor.
Blocks if no variant passed the gate this round.
"""

from __future__ import annotations
import tabula.skill_state_autopatch  # noqa

import subprocess
from datetime import datetime, timezone
from typing import Any

import math
import numpy as np

from zindian.config import ChallengeConfig
from zindian.paths import resolve_competition_paths
from zindian.state import SkillStateStore


def _metric_key(config: ChallengeConfig) -> str:
    metric_name = str(config.get("metric", "f1_score"))
    return "f1" if metric_name == "f1_score" else metric_name


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fold_score_variance(state: dict) -> float | None:
    metric_analysis = state.get("metric_analysis", {}) or {}
    if (
        isinstance(metric_analysis, dict)
        and metric_analysis.get("fold_score_variance") is not None
    ):
        return _to_float(metric_analysis.get("fold_score_variance"))

    # Fallback: recompute from eda fold_scores when metric_analysis is absent.
    # MUST apply NB correction, not raw sample variance — per S3 DoD.
    eda = state.get("eda", {}) or {}
    fold_scores = eda.get("fold_scores") if isinstance(eda, dict) else None
    if not fold_scores:
        return None
    return _nb_corrected_variance(fold_scores)


def _effective_thresholds(
    config: ChallengeConfig,
    state: dict,
) -> tuple[float, float, str | None]:
    """
    Return (effective_variance_threshold, effective_gate_margin, warning_message).

    For regression tasks:
      - If metric == "rmsle": raw thresholds are returned unchanged.
        RMSLE is a dimensionless log-ratio — applying target_std normalisation
        would mix log-space units with original-scale units, which is
        mathematically invalid.
      - If target_std == 0.0 (degenerate target): raw thresholds are
        returned and a non-empty warning string is returned to the caller
        for state logging. Pipeline does not halt — the warning is advisory.
      - Otherwise: variance_threshold is scaled by target_std**2 and
        gate_margin is scaled by target_std, making both thresholds
        scale-invariant across competitions with different target magnitudes.

    For classification tasks: raw thresholds are returned (bounded metrics
    need no scale correction).

    This function does NOT write to state. The caller is responsible for
    writing any returned warning_message to SKILL_STATE["metadata_warnings"].
    """
    variance_gate_threshold = float(config.get("variance_gate_threshold", 0.01) or 0.0)
    gate_margin = float(config.get("gate_margin", 0.001) or 0.0)
    task_type = str(config.get("task_type", "classification"))
    metric = str(config.get("metric", ""))

    if task_type != "regression":
        # Classification: bounded metrics — no scale correction needed.
        return variance_gate_threshold, gate_margin, None

    # SoT v2.2 Generalised Regression: explicit routing for each metric family
    SCALE_INVARIANT_METRICS = frozenset({"rmsle", "mase"})
    SCALE_SENSITIVE_METRICS = frozenset(
        {
            "rmse",
            "root_mean_squared_error",
            "mae",
            "mean_absolute_error",
        }
    )

    if metric in SCALE_INVARIANT_METRICS:
        # RMSLE is computed in log-space and is already scale-invariant.
        # Raw thresholds apply with no normalisation.
        return variance_gate_threshold, gate_margin, None

    if metric in SCALE_SENSITIVE_METRICS or (metric not in SCALE_INVARIANT_METRICS):
        # Catch-all: treat unknown regression metrics as scale-sensitive.
        # This prevents silent raw-threshold fallback for future metrics
        # that should be scaled but haven't been added to the set yet.
        eda_block = state.get("eda", {}) or {}
        target_std = float(eda_block.get("target_std") or 0.0)
        if target_std == 0.0:
            std_vals = [
                float(v)
                for k, v in eda_block.items()
                if k.endswith("_std") and isinstance(v, (int, float))
            ]
            if std_vals:
                target_std = std_vals[0]

        if target_std == 0.0:
            # Degenerate target: skill_04 may not have written target_std yet,
            # or the target has zero variance. Fall back to raw thresholds and
            # return a warning for the caller to log to state.
            warning = (
                "Degenerate target_std (0.0) in skill_11_gate: "
                "effective thresholds falling back to raw config values. "
                "Verify skill_04 EDA output has written target_std correctly."
            )
            return variance_gate_threshold, gate_margin, warning

        # Original-scale regression metrics (RMSE, MAE): scale thresholds by
        # target_std to make them magnitude-invariant across competitions.
        effective_variance = variance_gate_threshold * (target_std**2)
        effective_margin = gate_margin * target_std

        # v2.4 1-SE Promotion Margin (S9): effective_margin = max(effective_margin, 1 * SE_oof)
        metric_analysis = state.get("metric_analysis", {}) or {}
        se_oof = float(metric_analysis.get("se_oof") or 0.0)
        if se_oof > 0.0:
            effective_margin = max(effective_margin, 1.0 * se_oof)

        return effective_variance, effective_margin, None

    # Safety fallback (should not be reached if metric sets are comprehensive)
    metric_analysis = state.get("metric_analysis", {}) or {}
    se_oof = float(metric_analysis.get("se_oof") or 0.0)
    if se_oof > 0.0:
        gate_margin = max(gate_margin, 1.0 * se_oof)
    return variance_gate_threshold, gate_margin, None


def _baseline_score(state: dict, metric_key: str) -> tuple[float | None, str]:
    # Safe state access patterns
    retraining_active = state.get("pseudo_label_result", {}) or {}
    retraining_required = False
    if isinstance(retraining_active, dict):
        retraining_required = bool(retraining_active.get("retraining_required", False))

    anchor_challenge = state.get("anchor_challenge", {}) or {}
    challenge_active = False
    if isinstance(anchor_challenge, dict):
        challenge_active = bool(anchor_challenge.get("active", False))

    if retraining_required:
        key = "anchor_oof_score_augmented"
        value = _to_float(state.get(key))
        if value is not None:
            return value, key
        key = f"anchor_oof_{metric_key}_augmented"
        value = _to_float(state.get(key))
        if value is not None:
            return value, key

    if challenge_active:
        key = "anchor_oof_score_challenged"
        value = _to_float(state.get(key))
        if value is not None:
            return value, key
        key = f"anchor_oof_{metric_key}_challenged"
        value = _to_float(state.get(key))
        if value is not None:
            return value, key

    key = "anchor_oof_score"
    value = _to_float(state.get(key))
    if value is not None:
        return value, key
    key = f"anchor_oof_{metric_key}"
    value = _to_float(state.get(key))
    return (value, key) if value is not None else (None, key)


def _nb_corrected_variance(fold_scores: list[float]) -> float | None:
    """Compute NB-corrected variance from fold scores when metric_analysis
    is not available. Uses the equal-fold fallback NB factor (1/K + 1/(K-1))
    since fold_sizes are not available in the fallback path.

    This ensures the fallback branches of _target_fold_variance return
    NB-corrected variance, not raw sample variance — per the S3 DoD
    requirement that ALL variance paths return NB-corrected values.
    """
    try:
        arr = np.asarray(fold_scores, dtype=np.float64)
        K = len(arr)
        if K <= 1:
            return float(np.var(arr, ddof=1)) if K == 1 else 0.0
        var_sample = float(np.var(arr, ddof=1))
        # Equal-fold fallback NB factor: (1/K) + (1/(K-1))
        # This matches _get_nb_factor(K, None) in skill_12_metric.py.
        nb_factor = (1.0 / K) + (1.0 / (K - 1))
        return float(var_sample * nb_factor)
    except Exception:
        return None


def _target_fold_variance(state: dict, target_name: str) -> float | None:
    metric_analysis = state.get("metric_analysis", {}) or {}
    per_target = (
        metric_analysis.get("per_target", {})
        if isinstance(metric_analysis, dict)
        else {}
    )
    target_analysis = (
        per_target.get(target_name, {}) if isinstance(per_target, dict) else {}
    )
    if isinstance(target_analysis, dict):
        # Primary path: read NB-corrected variance directly from metric_analysis
        variance = _to_float(target_analysis.get("fold_score_variance"))
        if variance is not None:
            return variance
        # Fallback 1 (L206): recompute from fold_scores if variance key missing.
        # MUST apply NB correction, not raw sample variance — per S3 DoD.
        fold_scores = target_analysis.get("fold_scores")
        if fold_scores:
            return _nb_corrected_variance(fold_scores)

    multi_metrics = state.get("anchor_multi_target_metrics", {}) or {}
    target_metrics = (
        multi_metrics.get(target_name, {}) if isinstance(multi_metrics, dict) else {}
    )
    fold_scores = (
        target_metrics.get("fold_scores") if isinstance(target_metrics, dict) else None
    )
    if fold_scores:
        # Fallback 2 (L217): recompute from anchor_multi_target_metrics.
        # MUST apply NB correction, not raw sample variance — per S3 DoD.
        return _nb_corrected_variance(fold_scores)
    return None


def _effective_target_weight(
    config: ChallengeConfig, state: dict, target_spec: dict
) -> float:
    target_config = config.get("target_config", {}) or {}
    use_inverse_variance_weighting = bool(
        config.get("use_inverse_variance_weighting", False)
        or target_config.get("use_inverse_variance_weighting", False)
    )
    base_weight = float(target_spec.get("weight", 0.5) or 0.0)
    if not use_inverse_variance_weighting:
        return base_weight

    target_name = str(target_spec.get("name", ""))
    variance = _target_fold_variance(state, target_name)
    if variance is None:
        return base_weight
    return base_weight / (variance + 1e-8)


def _write_failure_diagnosis(store: SkillStateStore, diagnosis: dict) -> None:
    store.update(
        phase_3_gate_diagnosis=diagnosis,
        dag_phase="phase_3_gate_blocked",
        last_updated=datetime.now(timezone.utc).isoformat(),
    )


def _effective_multi_target_std(config: ChallengeConfig, state: dict) -> float:
    """Weighted RMS of the per-target standard deviations over regression
    targets only — [Fix-3] scoped exactly like the SoT `effective_target_std`:

        effective_target_std = sqrt( sum(w_i * sigma_i^2) / sum(w_i) )
    iterated over regression targets only (the 1-SE margin / variance gate are
    regression-scale-sensitive per D0/D4). Classification targets contribute
    nothing. Returns 0.0 when no regression target has a positive std."""
    target_config = config.get("target_config", {}) or {}
    targets = target_config.get("targets", []) or []
    eda = state.get("eda", {}) or {}
    num = 0.0
    den = 0.0
    for t in targets:
        if t.get("task_type") != "regression" or not t.get("name"):
            continue
        weight = float(t.get("weight", 0.5) or 0.0)
        sigma = _to_float(eda.get(f"{t['name']}_std"))
        if sigma is None or sigma <= 0:
            continue
        num += weight * (sigma ** 2)
        den += weight
    if den <= 0:
        return 0.0
    return math.sqrt(num / den)


def _multi_target_effective_thresholds(
    config: ChallengeConfig, state: dict
) -> tuple[float, float, str | None]:
    """Multi-target analog of `_effective_thresholds`.

    Returns (effective_variance_threshold, effective_gate_margin,
    warning_message | None). The caller writes any non-None warning to
    SKILL_STATE["metadata_warnings"]; this function never writes state.

    Regression-bearing composite:
      - effective_variance_threshold = variance_gate_threshold * (target_std**2)
      - effective_gate_margin        = max(gate_margin * target_std, 1 * composite_se_oof)
    where target_std is the regression-only weighted RMS ([Fix-3]) and
    composite_se_oof is the NB-based composite standard error computed by
    skill_12 ([Fix-1] — already carries NO /sqrt(K)).

    Classification-only composite: raw thresholds, no SE floor (D0/D4 — bounded
    metrics need no magnitude corridor).
    """
    variance_gate_threshold = float(config.get("variance_gate_threshold", 0.01) or 0.0)
    gate_margin = float(config.get("gate_margin", 0.001) or 0.0)
    target_config = config.get("target_config", {}) or {}
    targets = target_config.get("targets", []) or []

    regression_targets = [t for t in targets if t.get("task_type") == "regression"]
    if not regression_targets:
        return variance_gate_threshold, gate_margin, None

    effective_target_std = _effective_multi_target_std(config, state)
    if effective_target_std <= 0:
        warning = (
            "Degenerate multi-target effective_target_std in skill_11_gate: "
            "no positive per-target std found in the EDA block. Multi-target "
            "effective thresholds falling back to raw config values with no "
            "composite 1-SE margin."
        )
        return variance_gate_threshold, gate_margin, warning

    effective_variance = variance_gate_threshold * (effective_target_std ** 2)
    effective_margin = gate_margin * effective_target_std

    # [Fix-1] 1-SE promotion margin from the NB-based composite standard error.
    metric_analysis = state.get("metric_analysis", {}) or {}
    composite_se = _to_float(
        metric_analysis.get("composite_se_oof") if isinstance(metric_analysis, dict) else None
    )
    if composite_se is not None and composite_se > 0:
        effective_margin = max(effective_margin, 1.0 * composite_se)

    return effective_variance, effective_margin, None


def run() -> dict:
    print("\n" + "=" * 60)
    print("SKILL 11 — Branch Gate")
    print("=" * 60 + "\n")

    paths = resolve_competition_paths()
    config = ChallengeConfig.load()
    store = SkillStateStore(paths.state_path)
    state = store.read()

    # Multi-target detection
    target_config = config.get("target_config")
    if target_config and target_config.get("targets"):
        return _run_multi_target_gate(config, store, state)

    best_variant = state.get("best_variant_this_round") or state.get(
        "best_variant_branch"
    )
    metric_key = _metric_key(config)
    best_score_value = state.get("best_variant_oof_score")
    if best_score_value is None:
        best_score_value = state.get(f"best_variant_oof_{metric_key}")
    best_score = float(best_score_value or 0.0)
    fold_score_variance = _fold_score_variance(state)
    effective_variance_threshold, effective_gate_margin, threshold_warning = (
        _effective_thresholds(config, state)
    )
    if threshold_warning is not None:
        existing_warnings = state.get("metadata_warnings") or []
        if not isinstance(existing_warnings, list):
            existing_warnings = []
        store.update(metadata_warnings=existing_warnings + [threshold_warning])
    baseline_score, baseline_key = _baseline_score(state, metric_key)
    leaked_features = state.get("leaked_features", []) or []
    human_gate_key = (
        f"human_gate_2_{best_variant}_approved"
        if best_variant
        else "human_gate_2_unknown_approved"
    )
    human_gate_approved = bool(state.get(human_gate_key, False))
    shap_pass = bool(state.get("shap_completed_at")) and (
        bool(state.get("pruning_pass", False))
        or state.get("shap_audit_skipped_reason") == "single_feature"
        or state.get("shap_audit_skipped_reason") == "pca_columns_excluded"
    )
    variants_passed = int(state.get("variants_passed") or 0)
    branch_name = str(best_variant or "unknown")

    task_type = str(config.get("task_type", "classification"))
    direction = str(config.get("metric_direction", "maximize"))

    print(f"[skill_11] checking branch: {branch_name}")
    print(f"[skill_11] baseline key     : {baseline_key}")
    print(f"[skill_11] metric key       : {metric_key}")
    print(f"[skill_11] fold variance    : {fold_score_variance}")
    print(f"[skill_11] leaked features  : {len(leaked_features)}")
    print(f"[skill_11] human gate key   : {human_gate_key}={human_gate_approved}")

    diagnosis = {
        "branch_name": branch_name,
        "best_variant": best_variant,
        "metric_key": metric_key,
        "task_type": task_type,
        "direction": direction,
        "variants_passed": variants_passed,
        "best_score": best_score,
        "baseline_score": baseline_score,
        "fold_score_variance": fold_score_variance,
        "effective_variance_threshold": effective_variance_threshold,
        "effective_gate_margin": effective_gate_margin,
        "leaked_features": leaked_features,
        "shap_pass": shap_pass,
        "human_gate_key": human_gate_key,
        "human_gate_approved": human_gate_approved,
    }

    if variants_passed == 0:
        diagnosis["failure_reason"] = "no_variants_passed"
        _write_failure_diagnosis(store, diagnosis)
        return {
            "status": "BLOCKED",
            "reason": "no variants passed",
            "diagnosis": diagnosis,
        }

    if not branch_name:
        diagnosis["failure_reason"] = "missing_branch_name"
        _write_failure_diagnosis(store, diagnosis)
        return {
            "status": "BLOCKED",
            "reason": "missing branch name",
            "diagnosis": diagnosis,
        }

    if branch_name in {str(item) for item in leaked_features}:
        diagnosis["failure_reason"] = "branch_leaked"
        _write_failure_diagnosis(store, diagnosis)
        return {"status": "BLOCKED", "reason": "branch leaked", "diagnosis": diagnosis}

    if (
        fold_score_variance is None
        or fold_score_variance >= effective_variance_threshold
    ):
        diagnosis["failure_reason"] = "variance_gate_failed"
        _write_failure_diagnosis(store, diagnosis)
        return {
            "status": "BLOCKED",
            "reason": "variance gate failed",
            "diagnosis": diagnosis,
        }

    if baseline_score is None:
        diagnosis["failure_reason"] = "missing_baseline"
        _write_failure_diagnosis(store, diagnosis)
        return {
            "status": "BLOCKED",
            "reason": "missing baseline",
            "diagnosis": diagnosis,
        }

    if direction == "maximize":
        improved = (best_score - baseline_score) > effective_gate_margin
    else:
        improved = (baseline_score - best_score) > effective_gate_margin

    if not improved:
        diagnosis["failure_reason"] = "baseline_gate_failed"
        _write_failure_diagnosis(store, diagnosis)
        return {
            "status": "BLOCKED",
            "reason": "baseline gate failed",
            "diagnosis": diagnosis,
        }

    if not shap_pass:
        diagnosis["failure_reason"] = "shap_gate_failed"
        _write_failure_diagnosis(store, diagnosis)
        return {
            "status": "BLOCKED",
            "reason": "shap gate failed",
            "diagnosis": diagnosis,
        }

    if not human_gate_approved:
        # S6: Surface leakage_mi_advisory at Human Gate 2 if present.
        # These features passed the primary Pearson/NMI blocking check but were
        # flagged by the advisory MI regression subsample. Non-blocking — they do
        # NOT prevent promotion if the operator approves. Shown here so the
        # operator can make an informed decision before setting human_gate_2_*_approved.
        mi_advisory = state.get("leakage_mi_advisory") or []
        if mi_advisory:
            print(
                f"\n  [ADVISORY — Human Gate 2] MI regression check flagged "
                f"{len(mi_advisory)} feature(s) that passed the primary Pearson "
                f"block but showed elevated mutual information with the target:\n"
                f"    {mi_advisory}\n"
                f"  These are NON-BLOCKING. Review before approving: "
                f"are these features genuinely predictive or latent target copies?\n"
                f"  Set '{human_gate_key}' = true to proceed.\n"
            )
        diagnosis["failure_reason"] = "human_gate_missing"
        _write_failure_diagnosis(store, diagnosis)
        return {
            "status": "BLOCKED",
            "reason": "human gate missing",
            "diagnosis": diagnosis,
        }

    round_num = int(state.get("feature_round") or 1)
    new_branch = f"anchor-v{round_num + 1}"

    print(f"\n[OK] GATE PASSED — promoting {branch_name} to {new_branch}")

    try:
        subprocess.run(
            ["git", "checkout", "-b", new_branch], check=True, capture_output=True
        )
        print(f"[OK] Git branch created: {new_branch}")
    except subprocess.CalledProcessError:
        subprocess.run(["git", "checkout", new_branch], check=True, capture_output=True)
        print(f"[OK] Switched to: {new_branch}")

    updates = {
        "anchor_oof_score": best_score,
        f"anchor_oof_{metric_key}": best_score,
        "anchor_git_branch": new_branch,
        "feature_round": round_num + 1,
        "variants_tested": 0,
        "variants_passed": 0,
        "best_variant_this_round": None,
        "best_variant_oof_score": None,
        f"best_variant_oof_{metric_key}": None,
        "dag_phase": "phase_3_anchor_promoted",
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "phase_3_gate_diagnosis": {
            **diagnosis,
            "passed": True,
            "new_branch": new_branch,
        },
    }
    store.update(**updates)
    print(f"[OK] SKILL_STATE.json — new anchor {metric_key.upper()}: {best_score:.5f}")
    print(f"[OK] Feature round advanced to: {round_num + 1}")

    return {
        "status": "PASS",
        "new_branch": new_branch,
        "anchor_metric": best_score,
        "promoted": branch_name,
        "diagnosis": diagnosis,
    }


def _run_multi_target_gate(config, store, state) -> dict:
    """Multi-target gate logic per SoT v2.2.1 A11.

    [v2.7 / H1] Enforces four cumulative conditions before promotion:
      1. variance gate  - composite_fold_score_variance < effective_variance_threshold
      2. baseline gate  - (baseline - avg_score) > effective_gate_margin (composite
                          distance is minimize, so lower is better; H3/D2 resolves
                          the baseline to the _augmented anchor when retraining_required)
      3. multi-target SHAP gate (unchanged)
      4. human gate 2 (unchanged)
    Any failure writes phase_3_gate_diagnosis via _write_failure_diagnosis and
    returns BLOCKED. The anchor file-copy side effects run ONLY on the final
    PASS path - never partially on failure.
    """
    print("\n[TARGET] MULTI-TARGET GATE MODE\n")
    target_config = config.get("target_config", {})
    targets = target_config.get("targets", [])

    multi_metrics = state.get("anchor_multi_target_metrics", {})
    if not multi_metrics:
        return {"status": "BLOCKED", "reason": "no multi-target metrics found"}

    best_variant = state.get("best_variant_this_round") or state.get(
        "best_variant_branch"
    )
    branch_name = str(best_variant or "unknown")
    human_gate_key = f"human_gate_2_{branch_name}_approved"
    human_gate_approved = bool(state.get(human_gate_key, False))

    diagnosis = {
        "branch_name": branch_name,
        "best_variant": best_variant,
        "human_gate_key": human_gate_key,
        "human_gate_approved": human_gate_approved,
        "passed": False,
    }

    # Compute the composite score using config weights.
    weighted_distances = []
    for t in targets:
        target_name = t["name"]
        task_type = t["task_type"]
        weight = _effective_target_weight(config, state, t)

        if task_type == "classification":
            f1 = multi_metrics.get(target_name, {}).get("oof_f1", 0.0)
            distance = 1.0 - f1
        else:
            rmse = multi_metrics.get(target_name, {}).get("oof_rmse", 0.0)
            # Normalize by target std from eda block
            eda_std = float(state.get("eda", {}).get(f"{target_name}_std", 0.0))
            if eda_std <= 0.0:
                eda_std = float(state.get("eda", {}).get("target_std", 1.0))
            distance = rmse / eda_std if eda_std > 0 else rmse
        weighted_distances.append(distance * weight)

    total_weight = sum(_effective_target_weight(config, state, t) for t in targets)
    avg_score = (
        sum(weighted_distances) / total_weight
        if total_weight > 0
        else sum(weighted_distances)
    )
    diagnosis["avg_score"] = avg_score
# -- 1. Variance gate (H1) -----------------------------------------------
    effective_variance_threshold, effective_gate_margin, threshold_warning = (
        _multi_target_effective_thresholds(config, state)
    )
    if threshold_warning is not None:
        existing_warnings = state.get("metadata_warnings") or []
        if not isinstance(existing_warnings, list):
            existing_warnings = []
        store.update(metadata_warnings=existing_warnings + [threshold_warning])

    metric_analysis = state.get("metric_analysis", {}) or {}
    composite_variance = _to_float(
        metric_analysis.get("composite_fold_score_variance")
        if isinstance(metric_analysis, dict)
        else None
    )
    if composite_variance is None:
        composite_variance = 0.0
    if not (composite_variance < effective_variance_threshold):
        diagnosis["failure_reason"] = "variance gate failed"
        diagnosis["composite_fold_score_variance"] = composite_variance
        diagnosis["effective_variance_threshold"] = effective_variance_threshold
        _write_failure_diagnosis(store, diagnosis)
        return {
            "status": "BLOCKED",
            "reason": "variance gate failed",
            "diagnosis": diagnosis,
        }

    # -- 2. Baseline gate (H3/D2) --------------------------------------------
    # composite_direction is fixed "minimize_composite_distance": a lower
    # composite distance is better, so improvement means avg is below baseline
    # by more than the margin. When retraining_required == True the baseline
        # resolves to anchor_oof_score_augmented via _baseline_score.
    baseline_score, baseline_key = _baseline_score(state, "score")
    diagnosis["baseline_key"] = baseline_key
    if baseline_score is not None:
        diagnosis["baseline_score"] = baseline_score
    if baseline_score is None:
        diagnosis["failure_reason"] = "baseline gate failed (no baseline)"
        _write_failure_diagnosis(store, diagnosis)
        return {
            "status": "BLOCKED",
            "reason": "baseline gate failed",
            "diagnosis": diagnosis,
        }

    improved = (baseline_score - avg_score) > effective_gate_margin
    if not improved:
        diagnosis["failure_reason"] = "baseline gate failed"
        diagnosis["baseline_key"] = baseline_key
        diagnosis["baseline_score"] = baseline_score
        diagnosis["effective_gate_margin"] = effective_gate_margin
        _write_failure_diagnosis(store, diagnosis)
        return {
            "status": "BLOCKED",
            "reason": "baseline gate failed",
            "diagnosis": diagnosis,
        }

    # -- 3. Multi-target SHAP gate -------------------------------------------
    shap_results = state.get("shap_multi_target_results")
    if shap_results is None:
        shap_results = {}
    all_pass_all = all(
        shap_results.get(t["name"], {}).get("pruning_pass", False) for t in targets
    )
    if not all_pass_all:
        diagnosis["failure_reason"] = "shap gate failed"
        _write_failure_diagnosis(store, diagnosis)
        return {
            "status": "BLOCKED",
            "reason": "multi-target SHAP gate failed",
            "diagnosis": diagnosis,
        }
# -- 4. Human gate 2 -----------------------------------------------------
    if not human_gate_approved:
        # S6: Surface leakage_mi_advisory at Human Gate 2.
        mi_advisory = state.get("leakage_mi_advisory") or []
        if mi_advisory:
            print(
                f"\n  [ADVISORY \u2014 Human Gate 2] MI regression check flagged "
                f"{len(mi_advisory)} feature(s) across all targets that passed "
                f"the primary Pearson block but showed elevated mutual information:\n"
                f"    Combined: {mi_advisory}"
            )
            mt_shap = state.get("shap_multi_target_results") or {}
            for t_name, t_result in mt_shap.items():
                t_advisory = (t_result or {}).get("mi_advisory_feature_names", [])
                if t_advisory:
                    print(f"    \u2514\u2500 {t_name}: {t_advisory}")
            print(
                f"\n  These are NON-BLOCKING. Review before approving: "
                f"are these features genuinely predictive or latent target copies?\n"
                f"  Set '{human_gate_key}' = true to proceed.\n"
            )
        diagnosis["failure_reason"] = "human_gate_missing"
        _write_failure_diagnosis(store, diagnosis)
        return {
            "status": "BLOCKED",
            "reason": "human gate missing",
            "diagnosis": diagnosis,
        }

    round_num = int(state.get("feature_round") or 1)
    new_branch = f"anchor-multi-v{round_num + 1}"

    # -- File-copy side effects: ONLY after all four gates pass ------------
    try:
        import shutil
        from zindian.paths import resolve_competition_paths

        comp_paths = resolve_competition_paths()
        proc_dir = comp_paths.data_processed_dir

        src_train = proc_dir / f"features_train_{branch_name}.csv"
        dst_train = proc_dir / f"features_train_{new_branch}.csv"
        if src_train.exists():
            shutil.copy2(src_train, dst_train)
            print(f"  [OK] Copied train features to new anchor -> {dst_train}")

        src_test = proc_dir / f"features_test_{branch_name}.csv"
        dst_test = proc_dir / f"features_test_{new_branch}.csv"
        if src_test.exists():
            shutil.copy2(src_test, dst_test)
            print(f"  [OK] Copied test features to new anchor -> {dst_test}")

        for t in targets:
            t_name = t["name"]
            src_oof = proc_dir / f"oof_{branch_name}_{t_name}.csv"
            dst_oof = proc_dir / f"oof_{new_branch}_{t_name}.csv"
            if src_oof.exists():
                shutil.copy2(src_oof, dst_oof)
                print(f"  [OK] Copied OOF probs to new anchor -> {dst_oof}")

            src_tprobs = proc_dir / f"test_probs_{branch_name}_{t_name}.csv"
            dst_tprobs = proc_dir / f"test_probs_{new_branch}_{t_name}.csv"
            if src_tprobs.exists():
                shutil.copy2(src_tprobs, dst_tprobs)
                print(f"  [OK] Copied test probs to new anchor -> {dst_tprobs}")
    except Exception as e:
        print(f"  [WARNING] Failed to copy files for new anchor branch: {e}")

    store.update(
        anchor_oof_score=avg_score,
        anchor_git_branch=new_branch,
        feature_round=round_num + 1,
        dag_phase="phase_3_anchor_promoted",
        last_updated=datetime.now(timezone.utc).isoformat(),
        phase_3_gate_diagnosis={
            **diagnosis,
            "passed": True,
            "new_branch": new_branch,
            "avg_score": avg_score,
        },
    )
    print(f"\n[OK] Multi-target gate PASSED. New branch: {new_branch}")
    return {
        "status": "PASS",
        "new_branch": new_branch,
        "avg_score": avg_score,
        "baseline_key": baseline_key,
        "diagnosis": {**diagnosis, "passed": True, "new_branch": new_branch},
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2))
