"""
Tests for Gate 2 Option B (F6 fix).

Verifies that human gate approval satisfies ONLY condition 5.
Conditions 1-4 (leak-free, variance, baseline improvement, SHAP) are always
evaluated independently regardless of human_gate_approved state.

This is the key regression the Option A force-satisfy block was masking.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch, MagicMock


from zindian.schemas import skill_state_skeleton


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store(tmp_path: Path, extra: dict | None = None):
    from zindian.state import SkillStateStore

    state_path = tmp_path / "SKILL_STATE.json"
    state = skill_state_skeleton()
    # Baseline passing values for all 5 conditions
    state.update(
        {
            "best_variant_this_round": "variant_lgb",
            "best_variant_oof_score": 0.90,
            "anchor_oof_score": 0.85,
            "variants_passed": 1,
            "shap_completed_at": "2026-01-01T00:00:00+00:00",
            "pruning_pass": True,
            "leaked_features": [],
            "metric_analysis": {
                "fold_score_variance": 0.001,
                "se_oof": 0.005,
            },
            "human_gate_2_variant_lgb_approved": True,
        }
    )
    if extra:
        state.update(extra)
    state_path.write_text(json.dumps(state), encoding="utf-8")
    return SkillStateStore(state_path)


def _make_config(task_type="classification", direction="maximize", **kw) -> dict:
    cfg = {
        "metric": "auc",
        "task_type": task_type,
        "metric_direction": direction,
        "variance_gate_threshold": 0.01,
        "gate_margin": 0.001,
        "reproducibility": {"seed": 42},
        "cv_strategy": {"type": "stratifiedkfold", "n_splits": 5},
    }
    cfg.update(kw)
    return cfg


def _run_gate(tmp_path, store_extra=None, config_extra=None):
    """Run skill_11.run() with mocked paths and config."""
    import zindian.skills.skill_11_gate as s11
    import zindian.paths as zp
    from zindian.config import ChallengeConfig

    store = _make_store(tmp_path, store_extra)
    cfg_dict = _make_config(**(config_extra or {}))

    class _FakePaths:
        state_path = store.path
        competition_dir = tmp_path

    fake_config = MagicMock(spec=ChallengeConfig)
    fake_config.get = lambda key, default=None: cfg_dict.get(key, default)

    with (
        patch.object(zp, "resolve_competition_paths", return_value=_FakePaths()),
        patch.object(s11, "resolve_competition_paths", return_value=_FakePaths()),
        patch.object(s11, "ChallengeConfig") as mock_cc,
        patch("subprocess.run"),  # prevent actual git calls
    ):
        mock_cc.load.return_value = fake_config
        result = s11.run()

    return result, store.read()


# ---------------------------------------------------------------------------
# Core Option B assertion: human approval alone must NOT bypass conditions 1-4
# ---------------------------------------------------------------------------


class TestOptionBHumanGateAloneDoesNotPromote:

    def test_human_gate_approved_but_variance_fails_is_blocked(self, tmp_path):
        """Condition 2 (variance) must still be enforced when human approved."""
        result, _ = _run_gate(
            tmp_path,
            store_extra={
                "metric_analysis": {
                    "fold_score_variance": 999.0,  # far above threshold
                    "se_oof": 0.0,
                },
                "human_gate_2_variant_lgb_approved": True,
            },
        )
        assert result["status"] == "BLOCKED"
        assert result["diagnosis"]["failure_reason"] == "variance_gate_failed"

    def test_human_gate_approved_but_no_variants_passed_is_blocked(self, tmp_path):
        """Condition 0 (variants_passed > 0) must still block even with approval."""
        result, _ = _run_gate(
            tmp_path,
            store_extra={
                "variants_passed": 0,
                "human_gate_2_variant_lgb_approved": True,
            },
        )
        assert result["status"] == "BLOCKED"
        assert result["diagnosis"]["failure_reason"] == "no_variants_passed"

    def test_human_gate_approved_but_shap_failed_is_blocked(self, tmp_path):
        """Condition 4 (SHAP pass) must still be enforced when human approved."""
        result, _ = _run_gate(
            tmp_path,
            store_extra={
                "pruning_pass": False,
                "shap_audit_skipped_reason": None,
                "human_gate_2_variant_lgb_approved": True,
            },
        )
        assert result["status"] == "BLOCKED"
        assert result["diagnosis"]["failure_reason"] == "shap_gate_failed"

    def test_human_gate_approved_but_baseline_not_improved_is_blocked(self, tmp_path):
        """Condition 3 (improvement margin) must still fire when human approved."""
        result, _ = _run_gate(
            tmp_path,
            store_extra={
                "best_variant_oof_score": 0.85,  # same as anchor — no improvement
                "anchor_oof_score": 0.85,
                "human_gate_2_variant_lgb_approved": True,
            },
            config_extra={"gate_margin": 0.01},
        )
        assert result["status"] == "BLOCKED"
        assert result["diagnosis"]["failure_reason"] == "baseline_gate_failed"

    def test_human_gate_approved_but_branch_leaked_is_blocked(self, tmp_path):
        """Condition 1 (not in leaked_features) must still block even with approval."""
        result, _ = _run_gate(
            tmp_path,
            store_extra={
                "leaked_features": ["variant_lgb"],
                "human_gate_2_variant_lgb_approved": True,
            },
        )
        assert result["status"] == "BLOCKED"
        assert result["diagnosis"]["failure_reason"] == "branch_leaked"


class TestOptionBAllFiveConditionsNeeded:

    def test_all_five_passing_promotes(self, tmp_path):
        """All 5 conditions passing with human approval → PASS."""
        result, _ = _run_gate(tmp_path)
        assert result["status"] == "PASS"

    def test_missing_human_approval_alone_blocks(self, tmp_path):
        """All 4 automated conditions passing but no human approval → BLOCKED."""
        result, _ = _run_gate(
            tmp_path,
            store_extra={
                "human_gate_2_variant_lgb_approved": False,
            },
        )
        assert result["status"] == "BLOCKED"
        assert result["diagnosis"]["failure_reason"] == "human_gate_missing"

    def test_each_condition_independently_blocks(self, tmp_path):
        """Each of the 5 conditions, when failing in isolation, must block."""
        failing_cases: list[tuple[str, dict[str, Any], dict[str, Any], str]] = [
            # (description, store_extra, config_extra, expected_failure_reason)
            (
                "no_variants_passed",
                {"variants_passed": 0},
                {},
                "no_variants_passed",
            ),
            (
                "variance_gate_failed",
                {"metric_analysis": {"fold_score_variance": 999.0, "se_oof": 0.0}},
                {},
                "variance_gate_failed",
            ),
            (
                "shap_gate_failed",
                {"pruning_pass": False, "shap_audit_skipped_reason": None},
                {},
                "shap_gate_failed",
            ),
            (
                "human_gate_missing",
                {"human_gate_2_variant_lgb_approved": False},
                {},
                "human_gate_missing",
            ),
        ]
        for desc, store_extra, config_extra, expected_reason in failing_cases:
            result, _ = _run_gate(
                tmp_path, store_extra=store_extra, config_extra=config_extra
            )
            assert (
                result["status"] == "BLOCKED"
            ), f"Expected BLOCKED for '{desc}' but got {result['status']}"
            assert result["diagnosis"]["failure_reason"] == expected_reason, (
                f"For '{desc}': expected failure_reason='{expected_reason}', "
                f"got '{result['diagnosis'].get('failure_reason')}'"
            )


class TestShapSkipReasonEscapeHatch:

    def test_single_feature_skip_still_promotes(self, tmp_path):
        """shap_audit_skipped_reason='single_feature' satisfies condition 4."""
        result, _ = _run_gate(
            tmp_path,
            store_extra={
                "pruning_pass": False,
                "shap_audit_skipped_reason": "single_feature",
                "human_gate_2_variant_lgb_approved": True,
            },
        )
        assert (
            result["status"] == "PASS"
        ), f"single_feature skip should satisfy SHAP condition. Got: {result}"

    def test_pca_excluded_skip_promotes_after_f1(self, tmp_path):
        """shap_audit_skipped_reason='pca_columns_excluded' satisfies condition 4."""
        result, _ = _run_gate(
            tmp_path,
            store_extra={
                "pruning_pass": False,
                "shap_audit_skipped_reason": "pca_columns_excluded",
                "human_gate_2_variant_lgb_approved": True,
            },
        )
        assert (
            result["status"] == "PASS"
        ), f"pca_columns_excluded skip should satisfy SHAP condition. Got: {result}"

    def test_unknown_skip_reason_does_not_promote(self, tmp_path):
        """An unrecognised shap_audit_skipped_reason must NOT satisfy condition 4."""
        result, _ = _run_gate(
            tmp_path,
            store_extra={
                "pruning_pass": False,
                "shap_audit_skipped_reason": "my_custom_override",
                "human_gate_2_variant_lgb_approved": True,
            },
        )
        assert result["status"] == "BLOCKED"
        assert result["diagnosis"]["failure_reason"] == "shap_gate_failed"
