"""Tests for the three-lens evaluation framework (zindian/three_lens.py).

Covers all 5 phases plus contract/edge cases — 32 tests total.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from zindian.three_lens import (
    SUPPORTED_PHASES,
    evaluate_three_lenses,
    LensResult,
    ThreeLensReport,
)
from zindian.state import SkillStateStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path: Path) -> SkillStateStore:
    """Return a SkillStateStore backed by a temp file."""
    path = tmp_path / "SKILL_STATE.json"
    store = SkillStateStore(path)
    # Initialize with valid skeleton
    store.write({
        "competition": "test",
        "md5_target_hash": None,
        "anchor_oof_f1": None,
        "anchor_oof_rmse": None,
        "anchor_lb_score": None,
        "submissions_used_today": 0,
        "submissions_used_total": 0,
        "remaining_submissions": None,
        "dag_phase": "phase_0_foundation",
        "human_gate_1_approved": False,
        "human_gate_2_by_branch": {},
        "human_gate_3_approved": False,
        "human_gate_4_approved": False,
        "human_gate_5_selection": [],
        "selected_submissions": [],
        "last_updated": None,
    })
    return store


@pytest.fixture
def minimal_config() -> dict:
    return {
        "task_type": "classification",
        "metric": "f1_score",
        "metric_direction": "maximize",
        "use_probabilities": False,
        "target_col": "target",
        "target_domain_bounds": {"min": 0, "max": 1},
        "drift_threshold": 0.05,
        "reproducibility": {"seed": 42},
        "cv_strategy": {
            "type": "StratifiedKFold",
            "n_splits": 5,
            "shuffle": True,
            "random_state": 42,
            "group_col": None,
            "stratify_col": "target",
            "selection_reason": "imbalanced classification minority_ratio<0.15",
        },
        "shap_leak_threshold": 3.0,
        "variance_gate_threshold": 0.01,
        "gate_margin": 0.001,
        "spatial_signal": {"present": False, "lat_col": None, "lon_col": None, "group_col": None},
        "group_signal": {"present": False, "col": None, "type": None},
        "temporal_signal": {"present": False, "col": None},
        "minority_ratio": 0.08,
        "file_hashes": {"train.csv": "abc123"},
        "policy_filters": [],
        "community_signals": [],
        "submission_budget": {"total": 300, "daily": 10, "used": 0},
    }


@pytest.fixture
def minimal_state(store: SkillStateStore) -> SkillStateStore:
    state = store.read()
    state["eda"] = {
        "mnar_columns": [],
        "mcar_columns": [],
        "target_std": 0.45,
        "group_structure_confirmed": False,
        "temporal_index_confirmed": False,
    }
    store.write(state)
    return store


@pytest.fixture
def regression_config(minimal_config: dict) -> dict:
    cfg = dict(minimal_config)
    cfg["task_type"] = "regression"
    cfg["metric"] = "rmse"
    cfg["metric_direction"] = "minimize"
    cfg["cv_strategy"] = dict(cfg["cv_strategy"])
    cfg["cv_strategy"]["type"] = "KFold"
    cfg["cv_strategy"]["selection_reason"] = "standard regression fallback"
    cfg["minority_ratio"] = None
    return cfg


@pytest.fixture
def classification_config(minimal_config: dict) -> dict:
    cfg = dict(minimal_config)
    cfg["task_type"] = "classification"
    cfg["metric"] = "logloss"
    cfg["metric_direction"] = "minimize"
    cfg["minority_ratio"] = 0.08
    return cfg


@pytest.fixture
def temporal_config(minimal_config: dict) -> dict:
    cfg = dict(minimal_config)
    cfg["cv_strategy"] = dict(cfg["cv_strategy"])
    cfg["cv_strategy"]["type"] = "TimeSeriesSplit"
    cfg["cv_strategy"]["selection_reason"] = "temporal signal detected"
    return cfg


@pytest.fixture
def group_config(minimal_config: dict) -> dict:
    cfg = dict(minimal_config)
    cfg["spatial_signal"] = {"present": True, "lat_col": "lat", "lon_col": "lon", "group_col": "location_id"}
    cfg["group_signal"] = {"present": False, "col": None, "type": None}
    cfg["cv_strategy"] = dict(cfg["cv_strategy"])
    cfg["cv_strategy"]["type"] = "GroupKFold"
    cfg["cv_strategy"]["group_col"] = "location_id"
    cfg["cv_strategy"]["selection_reason"] = "spatial signal detected"
    return cfg


# ===================================================================
# Phase 1 tests
# ===================================================================

def test_phase1_all_pass(minimal_config: dict, minimal_state: SkillStateStore):
    report = evaluate_three_lenses("phase_1", minimal_config, minimal_state)
    assert report.overall == "PASS", f"Expected PASS, got {report.overall}: {report.to_dict()}"
    assert report.general.verdict == "PASS"
    assert report.specific.verdict == "PASS"
    assert report.generalisation.verdict == "PASS"


def test_phase1_general_fail_bad_metric_direction(minimal_config: dict, minimal_state: SkillStateStore):
    cfg = dict(minimal_config)
    cfg["metric_direction"] = "sideways"
    report = evaluate_three_lenses("phase_1", cfg, minimal_state)
    assert report.overall == "FAIL"
    assert report.general.verdict == "FAIL"
    assert "metric_direction" in " ".join(report.general.findings)


def test_phase1_general_fail_missing_task_type(minimal_config: dict, minimal_state: SkillStateStore):
    cfg = dict(minimal_config)
    cfg["task_type"] = None
    report = evaluate_three_lenses("phase_1", cfg, minimal_state)
    assert report.overall == "FAIL"
    assert report.general.verdict == "FAIL"
    assert "task_type" in " ".join(report.general.findings)


def test_phase1_general_fail_cv_mismatch_temporal(temporal_config: dict, minimal_state: SkillStateStore):
    cfg = dict(temporal_config)
    cfg["cv_strategy"] = dict(cfg["cv_strategy"])
    cfg["cv_strategy"]["type"] = "KFold"  # wrong type
    state = minimal_state.read()
    state["eda"]["temporal_index_confirmed"] = True
    minimal_state.write(state)
    report = evaluate_three_lenses("phase_1", cfg, minimal_state)
    assert report.overall == "FAIL"
    assert report.general.verdict == "FAIL"
    assert "TimeSeriesSplit" in " ".join(report.general.findings)


def test_phase1_specific_fail_missing_eda_block(minimal_config: dict, store: SkillStateStore):
    # store has no EDA block
    report = evaluate_three_lenses("phase_1", minimal_config, store)
    assert report.overall == "FAIL"
    assert report.specific.verdict == "FAIL"
    assert "eda" in " ".join(report.specific.findings).lower()


def test_phase1_specific_fail_missing_target_std_regression(regression_config: dict, store: SkillStateStore):
    state = store.read()
    state["eda"] = {
        "mnar_columns": [],
        "mcar_columns": [],
        # target_std deliberately missing
        "group_structure_confirmed": False,
        "temporal_index_confirmed": False,
    }
    store.write(state)
    report = evaluate_three_lenses("phase_1", regression_config, store)
    assert report.overall == "FAIL"
    assert report.specific.verdict == "FAIL"
    assert "target_std" in " ".join(report.specific.findings)


def test_phase1_specific_pass_target_std_not_required_classification(
    classification_config: dict, store: SkillStateStore
):
    state = store.read()
    state["eda"] = {
        "mnar_columns": [],
        "mcar_columns": [],
        # target_std absent — should be OK for classification
        "group_structure_confirmed": False,
        "temporal_index_confirmed": False,
    }
    store.write(state)
    report = evaluate_three_lenses("phase_1", classification_config, store)
    # specific may fail on other fields but not on target_std
    for f in report.specific.findings:
        assert "target_std" not in f, f"target_std should not be required for classification: {f}"


def test_phase1_specific_fail_missing_spatial_group_col(group_config: dict, store: SkillStateStore):
    cfg = dict(group_config)
    cfg["spatial_signal"] = {
        "present": True,
        "lat_col": "lat",
        "lon_col": "lon",
        "group_col": None,  # deliberately missing
    }
    state = store.read()
    state["eda"] = {
        "mnar_columns": [],
        "mcar_columns": [],
        "target_std": 0.5,
        "group_structure_confirmed": False,
        "temporal_index_confirmed": False,
    }
    store.write(state)
    report = evaluate_three_lenses("phase_1", cfg, store)
    assert report.overall == "FAIL"
    assert report.specific.verdict == "FAIL"
    assert "group_col" in " ".join(report.specific.findings)


def test_phase1_generalisation_fail_empty_file_hashes(minimal_config: dict, minimal_state: SkillStateStore):
    cfg = dict(minimal_config)
    cfg["file_hashes"] = {}
    report = evaluate_three_lenses("phase_1", cfg, minimal_state)
    assert report.overall == "FAIL"
    assert report.generalisation.verdict == "FAIL"
    assert "file_hashes" in " ".join(report.generalisation.findings)


def test_phase1_generalisation_fail_missing_seed(minimal_config: dict, minimal_state: SkillStateStore):
    cfg = dict(minimal_config)
    cfg["reproducibility"] = {"seed": None}
    report = evaluate_three_lenses("phase_1", cfg, minimal_state)
    assert report.overall == "FAIL"
    assert report.generalisation.verdict == "FAIL"
    assert "seed" in " ".join(report.generalisation.findings)


def test_phase1_generalisation_fail_empty_selection_reason(minimal_config: dict, minimal_state: SkillStateStore):
    cfg = dict(minimal_config)
    cfg["cv_strategy"] = dict(cfg["cv_strategy"])
    cfg["cv_strategy"]["selection_reason"] = ""
    report = evaluate_three_lenses("phase_1", cfg, minimal_state)
    assert report.overall == "FAIL"
    assert report.generalisation.verdict == "FAIL"


def test_phase1_generalisation_fail_cv_block_missing_subfield(minimal_config: dict, minimal_state: SkillStateStore):
    cfg = dict(minimal_config)
    cfg["cv_strategy"] = dict(cfg["cv_strategy"])
    del cfg["cv_strategy"]["group_col"]
    report = evaluate_three_lenses("phase_1", cfg, minimal_state)
    assert report.overall == "FAIL"
    assert report.generalisation.verdict == "FAIL"
    assert "group_col" in " ".join(report.generalisation.findings)


def test_phase1_report_written_to_state(minimal_config: dict, minimal_state: SkillStateStore):
    report = evaluate_three_lenses("phase_1", minimal_config, minimal_state)
    d = report.to_dict()
    assert d["phase"] == "phase_1"
    assert d["overall"] == "PASS"
    assert "general" in d
    assert "specific" in d
    assert "generalisation" in d
    assert "timestamp" in d
    # Verify JSON-serializable
    json.dumps(d)


def test_phase1_overall_fail_if_one_lens_fails(minimal_config: dict, minimal_state: SkillStateStore):
    cfg = dict(minimal_config)
    cfg["metric_direction"] = "sideways"  # breaks general
    report = evaluate_three_lenses("phase_1", cfg, minimal_state)
    assert report.overall == "FAIL"


def test_phase1_overall_pass_requires_all_three(minimal_config: dict, minimal_state: SkillStateStore):
    report = evaluate_three_lenses("phase_1", minimal_config, minimal_state)
    assert report.overall == "PASS"
    assert report.general.verdict == "PASS"
    assert report.specific.verdict == "PASS"
    assert report.generalisation.verdict == "PASS"


# ===================================================================
# Phase 2A tests
# ===================================================================

def test_phase2a_all_pass(minimal_config: dict, minimal_state: SkillStateStore):
    state = minimal_state.read()
    state["cleaning_complete"] = True
    state["mnar_indicator_before_fill"] = True
    state["policy_gate_passed"] = True
    minimal_state.write(state)
    report = evaluate_three_lenses("phase_2a", minimal_config, minimal_state)
    assert report.overall == "PASS", f"Expected PASS, got {report.overall}: {report.to_dict()}"


def test_phase2a_specific_fail_missing_cleaning_complete(minimal_config: dict, minimal_state: SkillStateStore):
    report = evaluate_three_lenses("phase_2a", minimal_config, minimal_state)
    assert report.overall == "FAIL"
    assert report.specific.verdict == "FAIL"
    assert "cleaning_complete" in " ".join(report.specific.findings)


def test_phase2a_generalisation_fail_mnar_order_violated(minimal_config: dict, minimal_state: SkillStateStore):
    state = minimal_state.read()
    state["cleaning_complete"] = True
    state["mnar_indicator_before_fill"] = False  # violated
    state["policy_gate_passed"] = True
    minimal_state.write(state)
    report = evaluate_three_lenses("phase_2a", minimal_config, minimal_state)
    assert report.overall == "FAIL"
    assert report.generalisation.verdict == "FAIL"
    assert "MNAR" in " ".join(report.generalisation.findings)


# ===================================================================
# Phase 2B tests
# ===================================================================

def test_phase2b_all_pass(minimal_config: dict, minimal_state: SkillStateStore):
    state = minimal_state.read()
    state["branch_anchor_oof"] = {"scores": [0.85, 0.87, 0.86, 0.88, 0.84], "cv_strategy_id": "config:StratifiedKFold"}
    state["branch_variant01_oof"] = {"scores": [0.86, 0.88, 0.87, 0.89, 0.85], "cv_strategy_id": "config:StratifiedKFold"}
    state["anchor_oof_score"] = 0.86
    state["human_gate_1_approved"] = True
    state["preflight_confirmed"] = True
    minimal_state.write(state)
    report = evaluate_three_lenses("phase_2b", minimal_config, minimal_state)
    assert report.overall == "PASS", f"Expected PASS, got {report.overall}: {report.to_dict()}"


def test_phase2b_general_fail_missing_cv_strategy_id_on_anchor(minimal_config: dict, minimal_state: SkillStateStore):
    state = minimal_state.read()
    state["branch_anchor_oof"] = {"scores": [0.85, 0.87, 0.86, 0.88, 0.84]}  # no cv_strategy_id
    state["anchor_oof_score"] = 0.86
    minimal_state.write(state)
    report = evaluate_three_lenses("phase_2b", minimal_config, minimal_state)
    assert report.overall == "FAIL"
    assert report.general.verdict == "FAIL"
    assert "cv_strategy_id" in " ".join(report.general.findings)


def test_phase2b_generalisation_fail_gate1_not_approved(minimal_config: dict, minimal_state: SkillStateStore):
    state = minimal_state.read()
    state["branch_anchor_oof"] = {"scores": [0.85], "cv_strategy_id": "config:StratifiedKFold"}
    state["anchor_oof_score"] = 0.86
    state["human_gate_1_approved"] = False  # not approved
    state["preflight_confirmed"] = True
    minimal_state.write(state)
    report = evaluate_three_lenses("phase_2b", minimal_config, minimal_state)
    assert report.overall == "FAIL"
    assert report.generalisation.verdict == "FAIL"
    assert "human_gate_1_approved" in " ".join(report.generalisation.findings)


def test_phase2b_generalisation_pass_with_cv_override(minimal_config: dict, minimal_state: SkillStateStore):
    state = minimal_state.read()
    state["branch_anchor_oof"] = {"scores": [0.85], "cv_strategy_id": "override:custom_cv"}
    state["anchor_oof_score"] = 0.86
    state["human_gate_1_approved"] = True
    state["preflight_confirmed"] = True
    state["cv_strategy_override"] = {"active": True, "override_strategy": "custom_cv"}
    minimal_state.write(state)
    report = evaluate_three_lenses("phase_2b", minimal_config, minimal_state)
    assert report.overall == "PASS", f"Expected PASS with override, got {report.overall}: {report.to_dict()}"


# ===================================================================
# Phase 3A tests
# ===================================================================

def test_phase3a_all_pass(minimal_config: dict, minimal_state: SkillStateStore):
    state = minimal_state.read()
    state["branch_variant01_oof"] = {
        "scores": [0.85, 0.87, 0.86, 0.88, 0.84],
        "cv_strategy_id": "config:StratifiedKFold",
        "branch_name": "variant01",
    }
    state["calibration_complete"] = True
    state["leaked_features"] = {"variant01": []}
    minimal_state.write(state)
    report = evaluate_three_lenses("phase_3a", minimal_config, minimal_state)
    assert report.overall == "PASS", f"Expected PASS, got {report.overall}: {report.to_dict()}"


def test_phase3a_specific_fail_shap_audit_incomplete(minimal_config: dict, minimal_state: SkillStateStore):
    state = minimal_state.read()
    state["branch_variant01_oof"] = {
        "scores": [0.85, 0.87, 0.86, 0.88, 0.84],
        "cv_strategy_id": "config:StratifiedKFold",
        "branch_name": "variant01",
    }
    # No leaked_features entry
    minimal_state.write(state)
    report = evaluate_three_lenses("phase_3a", minimal_config, minimal_state)
    assert report.overall == "FAIL"
    assert report.specific.verdict == "FAIL"


def test_phase3a_generalisation_fail_oof_missing_cv_strategy_id(minimal_config: dict, minimal_state: SkillStateStore):
    state = minimal_state.read()
    state["branch_variant01_oof"] = {
        "scores": [0.85, 0.87, 0.86, 0.88, 0.84],
        "branch_name": "variant01",
        # no cv_strategy_id
    }
    state["calibration_complete"] = True
    state["leaked_features"] = {"variant01": []}
    minimal_state.write(state)
    report = evaluate_three_lenses("phase_3a", minimal_config, minimal_state)
    assert report.overall == "FAIL"
    assert report.generalisation.verdict == "FAIL"
    assert "cv_strategy_id" in " ".join(report.generalisation.findings)


# ===================================================================
# Phase 3B tests
# ===================================================================

def test_phase3b_all_pass(minimal_config: dict, minimal_state: SkillStateStore):
    state = minimal_state.read()
    state["promoted_branches"] = ["variant01", "variant02"]
    state["fusion_strategy"] = {"method": "average", "oof_source": "original"}
    state["human_gate_2_by_branch"] = {
        "variant01_approved": True,
        "variant02_approved": True,
    }
    state["human_gate_3_approved"] = True
    state["diversity_check"] = {"max_correlation": 0.82}
    minimal_state.write(state)
    report = evaluate_three_lenses("phase_3b", minimal_config, minimal_state)
    assert report.overall == "PASS", f"Expected PASS, got {report.overall}: {report.to_dict()}"


def test_phase3b_specific_fail_gate2_not_approved_for_branch(minimal_config: dict, minimal_state: SkillStateStore):
    state = minimal_state.read()
    state["promoted_branches"] = ["variant01"]
    state["fusion_strategy"] = {"method": "average"}
    state["human_gate_2_by_branch"] = {}  # no approval for variant01
    state["human_gate_3_approved"] = True
    minimal_state.write(state)
    report = evaluate_three_lenses("phase_3b", minimal_config, minimal_state)
    assert report.overall == "FAIL"
    assert report.specific.verdict == "FAIL"
    assert "variant01" in " ".join(report.specific.findings)


def test_phase3b_generalisation_fail_pseudo_label_augmented_missing(
    minimal_config: dict, minimal_state: SkillStateStore
):
    state = minimal_state.read()
    state["promoted_branches"] = ["variant01"]
    state["fusion_strategy"] = {"method": "average", "oof_source": "original"}
    state["human_gate_2_by_branch"] = {"variant01_approved": True}
    state["human_gate_3_approved"] = True
    state["diversity_check"] = {"max_correlation": 0.82}
    state["pseudo_label_result"] = {"retraining_required": True}
    # No augmented OOF entries
    minimal_state.write(state)
    report = evaluate_three_lenses("phase_3b", minimal_config, minimal_state)
    assert report.overall == "FAIL"
    assert report.generalisation.verdict == "FAIL"
    assert "augmented" in " ".join(report.generalisation.findings)


# ===================================================================
# Contract and edge case tests
# ===================================================================

def test_unknown_phase_raises_value_error(minimal_config: dict, minimal_state: SkillStateStore):
    with pytest.raises(ValueError, match="phase_99"):
        evaluate_three_lenses("phase_99", minimal_config, minimal_state)


def test_to_dict_is_json_serializable(minimal_config: dict, minimal_state: SkillStateStore):
    report = evaluate_three_lenses("phase_1", minimal_config, minimal_state)
    d = report.to_dict()
    # Should not raise
    json.dumps(d)


def test_overall_derivation_warn_not_fail(minimal_config: dict, minimal_state: SkillStateStore):
    # Build a scenario where no lens FAILs but not all PASS
    cfg = dict(minimal_config)
    cfg["metric"] = None  # will fail general
    report = evaluate_three_lenses("phase_1", cfg, minimal_state)
    # If general fails, overall is FAIL — so test WARN via a different approach:
    # We need all lenses to be WARN. LensResult only has PASS/FAIL/WARN.
    # Override via direct construction to test the derivation logic.
    general = LensResult(lens="general", verdict="WARN", findings=["minor issue"])
    specific = LensResult(lens="specific", verdict="WARN", findings=["minor issue"])
    generalisation = LensResult(lens="generalisation", verdict="PASS", findings=[])
    r = ThreeLensReport(
        phase="phase_1", general=general, specific=specific,
        generalisation=generalisation, overall="", timestamp="now"
    )
    assert r.overall == "WARN"


def test_findings_empty_on_pass(minimal_config: dict, minimal_state: SkillStateStore):
    report = evaluate_three_lenses("phase_1", minimal_config, minimal_state)
    assert report.general.findings == []
    assert report.specific.findings == []
    assert report.generalisation.findings == []


def test_supported_phases_contains_all_five():
    assert SUPPORTED_PHASES == frozenset({
        "phase_1", "phase_2a", "phase_2b", "phase_3a", "phase_3b",
    })


def test_lens_result_to_dict():
    lr = LensResult(lens="general", verdict="FAIL", findings=["something wrong"])
    d = lr.to_dict()
    assert d["lens"] == "general"
    assert d["verdict"] == "FAIL"
    assert d["findings"] == ["something wrong"]


def test_report_has_phase_and_timestamp(minimal_config: dict, minimal_state: SkillStateStore):
    report = evaluate_three_lenses("phase_1", minimal_config, minimal_state)
    assert report.phase == "phase_1"
    assert report.timestamp is not None and len(report.timestamp) > 0