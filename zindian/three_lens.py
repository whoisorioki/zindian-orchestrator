"""Three-lens evaluation framework — SoT Section 4 phase gate checks.

Each phase has three lenses (general, specific, generalisation) that
evaluate whether pipeline execution satisfies the Source of Truth gates.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from zindian.state import SkillStateStore


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class LensResult:
    """Result of a single lens evaluation."""

    lens: str  # "general" | "specific" | "generalisation"
    verdict: str  # "PASS" | "FAIL" | "WARN"
    findings: List[str]  # empty on PASS, non-empty on FAIL/WARN

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lens": self.lens,
            "verdict": self.verdict,
            "findings": list(self.findings),
        }


@dataclass
class ThreeLensReport:
    """Complete three-lens evaluation for one phase."""

    phase: str
    general: LensResult
    specific: LensResult
    generalisation: LensResult
    overall: str  # "PASS" iff all three PASS; "FAIL" if any FAIL; "WARN" otherwise
    timestamp: str

    def __post_init__(self):
        self.overall = _derive_overall(self.general, self.specific, self.generalisation)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _derive_overall(general: LensResult, specific: LensResult, generalisation: LensResult) -> str:
    verdicts = (general.verdict, specific.verdict, generalisation.verdict)
    if all(v == "PASS" for v in verdicts):
        return "PASS"
    if any(v == "FAIL" for v in verdicts):
        return "FAIL"
    return "WARN"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Supported phases
# ---------------------------------------------------------------------------

SUPPORTED_PHASES = frozenset([
    "phase_1",
    "phase_2a",
    "phase_2b",
    "phase_3a",
    "phase_3b",
])

# ---------------------------------------------------------------------------
# Phase 1 evaluators
# ---------------------------------------------------------------------------

def _eval_phase1_general(config: dict, state: dict) -> LensResult:
    findings: List[str] = []

    task_type = config.get("task_type")
    valid_task_types = {"classification", "regression", "ranking"}
    if task_type not in valid_task_types:
        findings.append(f"task_type must be one of {valid_task_types}, got '{task_type}'")

    metric = config.get("metric")
    if not metric:
        findings.append("metric is missing or null")

    direction = config.get("metric_direction")
    valid_directions = {"maximize", "minimize"}
    if direction not in valid_directions:
        findings.append(f"metric_direction must be one of {valid_directions}, got '{direction}'")

    # CV strategy consistency with dataset signals
    cv_type = config.get("cv_strategy", {}).get("type", "")
    temporal = state.get("eda", {}).get("temporal_index_confirmed", False)
    group = state.get("eda", {}).get("group_structure_confirmed", False)
    spatial = config.get("spatial_signal", {}).get("present", False)
    minority_ratio = config.get("minority_ratio")

    if temporal and cv_type != "TimeSeriesSplit":
        findings.append(f"temporal signal detected but cv_strategy.type is '{cv_type}', expected 'TimeSeriesSplit'")
    elif (group or spatial) and cv_type != "GroupKFold":
        findings.append(f"group/spatial signal detected but cv_strategy.type is '{cv_type}', expected 'GroupKFold'")
    elif (task_type == "classification" and minority_ratio is not None and minority_ratio < 0.15
          and cv_type not in ("StratifiedKFold", "GroupKFold", "TimeSeriesSplit")):
        findings.append(f"imbalanced classification (minority_ratio={minority_ratio}) but cv_strategy.type is '{cv_type}', expected 'StratifiedKFold'")

    verdict = "PASS" if not findings else "FAIL"
    return LensResult(lens="general", verdict=verdict, findings=findings)


def _eval_phase1_specific(config: dict, state: dict) -> LensResult:
    findings: List[str] = []

    eda = state.get("eda")
    if eda is None:
        findings.append("state['eda'] block is missing")
        return LensResult(lens="specific", verdict="FAIL", findings=findings)

    # Required EDA fields (target_std is regression-only)
    for field in ("mnar_columns", "mcar_columns", "group_structure_confirmed", "temporal_index_confirmed"):
        if field not in eda:
            findings.append(f"state['eda']['{field}'] is missing")

    # target_std > 0 for regression
    task_type = config.get("task_type")
    if task_type == "regression":
        target_std = eda.get("target_std")
        if target_std is None or not isinstance(target_std, (int, float)) or target_std <= 0:
            findings.append(f"target_std must be a float > 0 for regression, got '{target_std}'")

    # CV type matches actual signals
    cv_type = config.get("cv_strategy", {}).get("type", "")
    temporal = eda.get("temporal_index_confirmed", False)
    group = eda.get("group_structure_confirmed", False)
    spatial = config.get("spatial_signal", {}).get("present", False)

    if temporal and cv_type != "TimeSeriesSplit":
        findings.append(f"temporal_index_confirmed=true but cv_strategy.type is '{cv_type}', expected 'TimeSeriesSplit'")
    if (group or spatial) and cv_type != "GroupKFold":
        findings.append(f"group_structure_confirmed=true or spatial_signal.present=true but cv_strategy.type is '{cv_type}', expected 'GroupKFold'")

    # Spatial group_col populated when spatial present and group absent
    if spatial and not config.get("group_signal", {}).get("present", False):
        group_col = config.get("spatial_signal", {}).get("group_col")
        if not group_col:
            findings.append("spatial_signal.present=true and group_signal.present=false but spatial_signal.group_col is not populated")

    verdict = "PASS" if not findings else "FAIL"
    return LensResult(lens="specific", verdict=verdict, findings=findings)


def _eval_phase1_generalisation(config: dict, state: dict) -> LensResult:
    findings: List[str] = []

    # file_hashes
    file_hashes = config.get("file_hashes")
    if not file_hashes or not isinstance(file_hashes, dict) or len(file_hashes) == 0:
        findings.append("file_hashes must be a non-empty dict")

    # policy_filters
    policy_filters = config.get("policy_filters")
    if not isinstance(policy_filters, list):
        findings.append("policy_filters must be a list")

    # reproducibility.seed
    seed = config.get("reproducibility", {}).get("seed")
    if seed is None or not isinstance(seed, int):
        findings.append("reproducibility.seed must be present and an int")

    # cv_strategy block
    cv = config.get("cv_strategy", {})
    if not isinstance(cv, dict):
        findings.append("cv_strategy block must be a dict")
    else:
        required_cv_fields = ["type", "n_splits", "shuffle", "random_state", "group_col", "stratify_col", "selection_reason"]
        for field in required_cv_fields:
            if field not in cv:
                findings.append(f"cv_strategy missing required field '{field}'")

        selection_reason = cv.get("selection_reason")
        if not selection_reason:
            findings.append("cv_strategy.selection_reason must be a non-empty string")

    verdict = "PASS" if not findings else "FAIL"
    return LensResult(lens="generalisation", verdict=verdict, findings=findings)


# ---------------------------------------------------------------------------
# Phase 2A evaluators
# ---------------------------------------------------------------------------

def _eval_phase2a_general(config: dict, state: dict) -> LensResult:
    findings: List[str] = []

    policy_filters = config.get("policy_filters")
    if policy_filters is None or not isinstance(policy_filters, list):
        findings.append("policy_filters must be present and a list (config lock confirmed)")

    direction = config.get("metric_direction")
    if direction not in ("maximize", "minimize"):
        findings.append("metric_direction still present and valid (config read-only check)")

    verdict = "PASS" if not findings else "FAIL"
    return LensResult(lens="general", verdict=verdict, findings=findings)


def _eval_phase2a_specific(config: dict, state: dict) -> LensResult:
    findings: List[str] = []

    eda = state.get("eda", {})
    if "mnar_columns" not in eda:
        findings.append("state['eda']['mnar_columns'] missing — cleaning uninformed by MNAR profile")
    if "mcar_columns" not in eda:
        findings.append("state['eda']['mcar_columns'] missing — cleaning uninformed by MCAR profile")

    cleaning_complete = state.get("cleaning_complete")
    if not cleaning_complete:
        findings.append("state['cleaning_complete'] not set or false — cleaning may not be complete")

    verdict = "PASS" if not findings else "FAIL"
    return LensResult(lens="specific", verdict=verdict, findings=findings)


def _eval_phase2a_generalisation(config: dict, state: dict) -> LensResult:
    findings: List[str] = []

    # MNAR indicator-before-fill order check
    mnar_order = state.get("mnar_indicator_before_fill")
    if mnar_order is False:
        findings.append("MNAR indicators were NOT generated before fills — order violation")

    # Policy gate: blocked columns absent from post-cleaning feature matrix
    # (the actual feature matrix isn't available here; we check state flags)
    policy_gate_passed = state.get("policy_gate_passed")
    if policy_gate_passed is False:
        findings.append("policy gate did not pass — blocked columns still present in feature matrix")

    verdict = "PASS" if not findings else "FAIL"
    return LensResult(lens="generalisation", verdict=verdict, findings=findings)


# ---------------------------------------------------------------------------
# Phase 2B evaluators
# ---------------------------------------------------------------------------

def _eval_phase2b_general(config: dict, state: dict) -> LensResult:
    findings: List[str] = []

    anchor_oof = state.get("anchor_oof_score") or state.get("branch_anchor_oof")
    if anchor_oof is None:
        findings.append("anchor OOF score not present in state")

    direction = config.get("metric_direction")

    # Check cv_strategy_id tag on anchor OOF entry
    anchor_key = next((k for k in state if k.startswith("branch_") and k.endswith("_oof") and "anchor" in k), None)
    if anchor_key:
        oof_entry = state.get(anchor_key, {})
        if not oof_entry.get("cv_strategy_id"):
            findings.append(f"anchor OOF entry '{anchor_key}' is missing cv_strategy_id tag")
    else:
        findings.append("no branch_anchor_oof entry found in state to verify cv_strategy_id tag")

    verdict = "PASS" if not findings else "FAIL"
    return LensResult(lens="general", verdict=verdict, findings=findings)


def _eval_phase2b_specific(config: dict, state: dict) -> LensResult:
    findings: List[str] = []

    # Anchor OOF is a finite float
    anchor_oof = state.get("anchor_oof_score") or state.get("branch_anchor_oof")
    if anchor_oof is not None:
        try:
            val = float(anchor_oof)
            if val != val:  # NaN check
                findings.append("anchor OOF score is NaN")
        except (TypeError, ValueError):
            findings.append(f"anchor OOF score is not a finite float: {anchor_oof}")

    # At least one feature variant OOF score present
    variant_oof_keys = [k for k in state if k.startswith("branch_") and k.endswith("_oof") and "anchor" not in k]
    if not variant_oof_keys:
        findings.append("no feature variant OOF scores present in state")

    # All OOF outputs carry cv_strategy_id matching active strategy
    active_cv_id = _resolve_active_cv_id_for_check(state, config)
    for key in [k for k in state if k.startswith("branch_") and k.endswith("_oof")]:
        entry = state.get(key, {})
        entry_cv = entry.get("cv_strategy_id")
        if not entry_cv:
            findings.append(f"OOF entry '{key}' missing cv_strategy_id tag")
        elif active_cv_id and entry_cv != active_cv_id:
            findings.append(f"OOF entry '{key}' has cv_strategy_id='{entry_cv}', expected '{active_cv_id}'")

    verdict = "PASS" if not findings else "FAIL"
    return LensResult(lens="specific", verdict=verdict, findings=findings)


def _eval_phase2b_generalisation(config: dict, state: dict) -> LensResult:
    findings: List[str] = []

    # Human Gate 1 approved
    if state.get("human_gate_1_approved") is not True:
        findings.append("human_gate_1_approved is not true — anchor not approved")

    # Preflight confirmed
    if state.get("preflight_confirmed") is not True:
        findings.append("preflight_confirmed is not true — static checks may not have passed")

    # CV strategy override safe access
    override_active = state.get("cv_strategy_override", {}).get("active", False)
    if override_active:
        override_strategy = state.get("cv_strategy_override", {}).get("override_strategy")
        if not override_strategy:
            findings.append("cv_strategy_override.active=true but override_strategy not set")

    verdict = "PASS" if not findings else "FAIL"
    return LensResult(lens="generalisation", verdict=verdict, findings=findings)


def _resolve_active_cv_id_for_check(state: dict, config: dict) -> Optional[str]:
    """Resolve the active CV strategy ID for validation, returns None if indeterminate."""
    try:
        override = state.get("cv_strategy_override", {}) or {}
        if override.get("active", False):
            return f"override:{override.get('override_strategy') or 'unknown'}"
        cv = config.get("cv_strategy", {}) if isinstance(config, dict) else {}
        if isinstance(cv, dict) and cv.get("type"):
            return f"config:{cv['type']}"
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Phase 3A evaluators
# ---------------------------------------------------------------------------

def _eval_phase3a_general(config: dict, state: dict) -> LensResult:
    findings: List[str] = []

    task_type = config.get("task_type")
    if task_type == "classification":
        calib = state.get("calibration_complete")
        if not calib:
            findings.append("task_type=classification but calibration_complete is not true in state")

    # Fold score variance written for all candidate branches
    branch_oof_keys = [k for k in state if k.startswith("branch_") and k.endswith("_oof")]
    if not branch_oof_keys:
        findings.append("no branch OOF entries found in state")

    verdict = "PASS" if not findings else "FAIL"
    return LensResult(lens="general", verdict=verdict, findings=findings)


def _eval_phase3a_specific(config: dict, state: dict) -> LensResult:
    findings: List[str] = []

    branch_oof_keys = [k for k in state if k.startswith("branch_") and k.endswith("_oof")]

    for key in branch_oof_keys:
        entry = state.get(key, {})
        scores = entry.get("scores", [])
        if not scores or not isinstance(scores, list):
            findings.append(f"OOF entry '{key}' has no scores list")
            continue

        # Fold count matches n_splits
        n_splits = config.get("cv_strategy", {}).get("n_splits", 5)
        if len(scores) != n_splits:
            findings.append(f"OOF entry '{key}' has {len(scores)} fold scores, expected {n_splits}")

        # SHAP audit: leaked_features key should be written
        branch_name = entry.get("branch_name", key.replace("branch_", "").replace("_oof", ""))
        shap_audit_key = f"shap_leaked_features_{branch_name}"
        if shap_audit_key not in state:
            # Check for leaked_features block by branch scan in state
            leaked = state.get("leaked_features", {})
            if isinstance(leaked, dict) and branch_name not in leaked:
                findings.append(f"SHAP leak audit not found for branch '{branch_name}' (expected key '{shap_audit_key}')")

    verdict = "PASS" if not findings else "FAIL"
    return LensResult(lens="specific", verdict=verdict, findings=findings)


def _eval_phase3a_generalisation(config: dict, state: dict) -> LensResult:
    findings: List[str] = []

    branch_oof_keys = [k for k in state if k.startswith("branch_") and k.endswith("_oof")]

    for key in branch_oof_keys:
        entry = state.get(key, {})
        cv_id = entry.get("cv_strategy_id")
        if not cv_id:
            findings.append(f"OOF entry '{key}' is missing cv_strategy_id tag")

    # Leaked features evaluated — branches with non-empty leaked blocked
    leaked = state.get("leaked_features", {})
    if isinstance(leaked, dict):
        for branch_name, features in leaked.items():
            if features and len(features) > 0:
                findings.append(f"branch '{branch_name}' has leaked features: {features} — should be blocked")

    verdict = "PASS" if not findings else "FAIL"
    return LensResult(lens="generalisation", verdict=verdict, findings=findings)


# ---------------------------------------------------------------------------
# Phase 3B evaluators
# ---------------------------------------------------------------------------

def _eval_phase3b_general(config: dict, state: dict) -> LensResult:
    findings: List[str] = []

    # At least one branch passed skill_11 gate
    promoted = state.get("promoted_branches", [])
    if not promoted or not isinstance(promoted, list) or len(promoted) == 0:
        findings.append("no branches promoted (promoted_branches is empty or absent)")

    # Fusion strategy written
    fusion = state.get("fusion_strategy")
    if not fusion:
        findings.append("fusion_strategy not written to state")

    verdict = "PASS" if not findings else "FAIL"
    return LensResult(lens="general", verdict=verdict, findings=findings)


def _eval_phase3b_specific(config: dict, state: dict) -> LensResult:
    findings: List[str] = []

    gate2 = state.get("human_gate_2_by_branch", {})
    promoted = state.get("promoted_branches", [])

    for branch in promoted:
        branch_key = f"{branch}_approved"
        if branch_key not in gate2:
            findings.append(f"human_gate_2_by_branch missing entry for promoted branch '{branch}'")
        elif gate2[branch_key] is not True:
            findings.append(f"human_gate_2_by_branch.{branch_key} is not true")

    if state.get("human_gate_3_approved") is not True:
        findings.append("human_gate_3_approved is not true — fusion not authorized")

    # Diversity check — no two candidates with correlation > 0.95
    diversity = state.get("diversity_check")
    if diversity is not None:
        if isinstance(diversity, dict) and diversity.get("max_correlation", 0) > 0.95:
            findings.append(f"diversity check max_correlation={diversity['max_correlation']} exceeds 0.95 threshold")

    verdict = "PASS" if not findings else "FAIL"
    return LensResult(lens="specific", verdict=verdict, findings=findings)


def _eval_phase3b_generalisation(config: dict, state: dict) -> LensResult:
    findings: List[str] = []

    # Pseudo-label retraining: augmented OOF namespace check
    pseudo = state.get("pseudo_label_result", {})
    if pseudo.get("retraining_required") is True:
        # Check augmented anchor OOF is present
        anchor_augmented = state.get("branch_anchor_oof_augmented") or state.get("branch_anchor_oof_augmented_score")
        if not any(k for k in state if k.endswith("_augmented")):
            findings.append("pseudo_label retraining required but no augmented OOF entries found in state")

    # Fusion uses most recent OOF arrays only — check timestamp ordering
    fusion = state.get("fusion_strategy", {})
    if isinstance(fusion, dict) and fusion.get("oof_source") == "augmented":
        # If fusion uses augmented, ensure non-augmented originals exist
        branch_oof_keys = [k for k in state if k.startswith("branch_") and k.endswith("_oof") and not k.endswith("_augmented")]
        if not branch_oof_keys:
            findings.append("fusion uses augmented OOF but no original (non-augmented) OOF entries found")

    verdict = "PASS" if not findings else "FAIL"
    return LensResult(lens="generalisation", verdict=verdict, findings=findings)


# ---------------------------------------------------------------------------
# Phase evaluator dispatch
# ---------------------------------------------------------------------------

_PHASE_EVALUATORS = {
    "phase_1": (_eval_phase1_general, _eval_phase1_specific, _eval_phase1_generalisation),
    "phase_2a": (_eval_phase2a_general, _eval_phase2a_specific, _eval_phase2a_generalisation),
    "phase_2b": (_eval_phase2b_general, _eval_phase2b_specific, _eval_phase2b_generalisation),
    "phase_3a": (_eval_phase3a_general, _eval_phase3a_specific, _eval_phase3a_generalisation),
    "phase_3b": (_eval_phase3b_general, _eval_phase3b_specific, _eval_phase3b_generalisation),
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_three_lenses(phase: str, config: dict, state: SkillStateStore) -> ThreeLensReport:
    """Evaluate all three lenses for a given phase.

    Args:
        phase: One of SUPPORTED_PHASES.
        config: Contents of challenge_config.json as a dict.
        state: SkillStateStore instance backed by SKILL_STATE.json.

    Returns:
        A ThreeLensReport with all lens results and derived overall verdict.

    Raises:
        ValueError: If phase is not in SUPPORTED_PHASES.
    """
    phase_norm = phase.lower()

    if phase_norm not in SUPPORTED_PHASES:
        raise ValueError(
            f"Unknown phase '{phase}'. Supported phases: {sorted(SUPPORTED_PHASES)}"
        )

    state_dict = state.read()
    general_fn, specific_fn, generalisation_fn = _PHASE_EVALUATORS[phase_norm]

    general = general_fn(config, state_dict)
    specific = specific_fn(config, state_dict)
    generalisation = generalisation_fn(config, state_dict)

    report = ThreeLensReport(
        phase=phase_norm,
        general=general,
        specific=specific,
        generalisation=generalisation,
        overall="",  # Will be set by __post_init__
        timestamp=_now_iso(),
    )

    return report