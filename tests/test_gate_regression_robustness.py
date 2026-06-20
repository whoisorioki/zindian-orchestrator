"""
Tests for regression gate threshold correctness.
Covers RMSE scaling, RMSLE raw passthrough, degenerate target_std guard,
and classification passthrough.

SoT §4 / §8 references:
  Gate condition 2 (variance_threshold):
    regression (RMSE): config["variance_gate_threshold"] * (target_std ** 2)
    regression (RMSLE): config["variance_gate_threshold"] raw (no scaling)
    classification: config["variance_gate_threshold"] raw

  Gate condition 3 (gate_margin):
    regression (RMSE): config["gate_margin"] * target_std
    regression (RMSLE): config["gate_margin"] raw (no scaling)
    classification: config["gate_margin"] raw

  Degenerate target_std (== 0.0) on non-RMSLE regression:
    Falls back to raw thresholds; returns non-None warning string.
    Pipeline does not halt — warning is advisory only.
"""

import json
import tempfile
from pathlib import Path

from zindian.config import ChallengeConfig
from zindian.skills.skill_11_gate import _effective_thresholds

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(data: dict) -> ChallengeConfig:
    """Build a minimal ChallengeConfig from a plain dict for testing."""
    full = {
        "metric": data.get("metric", "f1"),
        "metric_direction": data.get("metric_direction", "maximize"),
        "use_probabilities": False,
        "automl_permitted": False,
        "data_modality": "tabular",
        **data,
    }
    tmp = Path(tempfile.mktemp(suffix=".json"))
    tmp.write_text(json.dumps(full))
    return ChallengeConfig(path=tmp, _data=full)


def _reg_config(
    metric: str, variance: float = 0.01, margin: float = 0.001
) -> ChallengeConfig:
    return _make_config(
        {
            "task_type": "regression",
            "metric": metric,
            "metric_direction": "minimize",
            "variance_gate_threshold": variance,
            "gate_margin": margin,
        }
    )


def _clf_config(
    metric: str = "auc", variance: float = 0.01, margin: float = 0.001
) -> ChallengeConfig:
    return _make_config(
        {
            "task_type": "classification",
            "metric": metric,
            "metric_direction": "maximize",
            "variance_gate_threshold": variance,
            "gate_margin": margin,
        }
    )


def _state(target_std: float) -> dict:
    return {"eda": {"target_std": target_std}}


# ---------------------------------------------------------------------------
# RMSE — thresholds must scale by target_std
# ---------------------------------------------------------------------------


class TestRMSEScaling:
    """RMSE: effective thresholds must scale by target_std."""

    def test_variance_threshold_scaled_by_std_squared(self):
        eff_var, eff_margin, warning = _effective_thresholds(
            _reg_config("rmse"), _state(2.5)
        )
        assert (
            abs(eff_var - 0.01 * (2.5**2)) < 1e-9
        ), f"Expected {0.01 * 6.25}, got {eff_var}"
        assert abs(eff_margin - 0.001 * 2.5) < 1e-9
        assert warning is None

    def test_large_target_std_scales_correctly(self):
        """σ_y=500 regression: gate_margin must become 0.5, not 0.001."""
        eff_var, eff_margin, warning = _effective_thresholds(
            _reg_config("rmse"), _state(500.0)
        )
        assert abs(eff_var - 0.01 * (500.0**2)) < 1e-4
        assert abs(eff_margin - 0.001 * 500.0) < 1e-9
        assert warning is None

    def test_rmse_treats_mae_identically(self):
        """MAE is also an original-scale metric and must scale like RMSE."""
        eff_var, eff_margin, warning = _effective_thresholds(
            _reg_config("mae"), _state(3.0)
        )
        assert abs(eff_var - 0.01 * (3.0**2)) < 1e-9
        assert abs(eff_margin - 0.001 * 3.0) < 1e-9
        assert warning is None


# ---------------------------------------------------------------------------
# RMSLE — raw thresholds must be returned, no scaling
# ---------------------------------------------------------------------------


class TestRMSLERawPassthrough:
    """RMSLE: raw thresholds must be returned without any scaling."""

    def test_raw_variance_threshold_returned(self):
        eff_var, eff_margin, warning = _effective_thresholds(
            _reg_config("rmsle"), _state(1.0)
        )
        assert eff_var == 0.01
        assert eff_margin == 0.001
        assert warning is None

    def test_large_std_does_not_scale_rmsle_margin(self):
        """
        Core regression test: RMSLE with target_std=500 must NOT produce
        gate_margin=0.5 (0.001 * 500). It must remain 0.001.
        Scaling a log-ratio metric by σ_y applies wrong units.
        """
        eff_var, eff_margin, warning = _effective_thresholds(
            _reg_config("rmsle"), _state(500.0)
        )
        assert eff_margin == 0.001, (
            f"RMSLE gate_margin must be raw 0.001; "
            f"got {eff_margin} — target_std scaling was incorrectly applied."
        )
        assert (
            eff_var == 0.01
        ), f"RMSLE variance_threshold must be raw 0.01; got {eff_var}."
        assert warning is None

    def test_rmsle_zero_std_still_returns_raw(self):
        """
        RMSLE branch exits before the zero-std guard — zero target_std must
        not trigger the fallback warning for RMSLE.
        """
        eff_var, eff_margin, warning = _effective_thresholds(
            _reg_config("rmsle"), _state(0.0)
        )
        assert eff_var == 0.01
        assert eff_margin == 0.001
        assert warning is None

    def test_rmsle_no_eda_key_still_raw(self):
        """State with no eda key at all: RMSLE still returns raw."""
        eff_var, eff_margin, warning = _effective_thresholds(_reg_config("rmsle"), {})
        assert eff_var == 0.01
        assert eff_margin == 0.001
        assert warning is None


# ---------------------------------------------------------------------------
# Degenerate target_std — zero std on non-RMSLE regression
# ---------------------------------------------------------------------------


class TestDegenerateTargetStd:
    """Zero target_std on non-RMSLE regression: fall back to raw, return warning."""

    def test_zero_std_returns_raw_thresholds(self):
        eff_var, eff_margin, _ = _effective_thresholds(_reg_config("rmse"), _state(0.0))
        assert eff_var == 0.01
        assert eff_margin == 0.001

    def test_zero_std_returns_warning_string(self):
        _, _, warning = _effective_thresholds(_reg_config("rmse"), _state(0.0))
        assert warning is not None
        assert isinstance(warning, str)
        assert len(warning) > 0
        assert "target_std" in warning.lower()

    def test_missing_eda_key_treated_as_zero(self):
        """State with no eda key at all behaves as target_std=0."""
        eff_var, eff_margin, warning = _effective_thresholds(_reg_config("rmse"), {})
        assert eff_var == 0.01
        assert eff_margin == 0.001
        assert warning is not None

    def test_none_target_std_treated_as_zero(self):
        """Explicit None in target_std falls through to zero path."""
        _, _, warning = _effective_thresholds(
            _reg_config("rmse"), {"eda": {"target_std": None}}
        )
        assert warning is not None

    def test_warning_does_not_halt_pipeline(self):
        """Function must complete normally — not raise — on zero target_std."""
        eff_var, eff_margin, warning = _effective_thresholds(
            _reg_config("rmse"), _state(0.0)
        )
        # Verify all three elements are returned (no exception raised)
        assert isinstance(eff_var, float)
        assert isinstance(eff_margin, float)
        assert warning is not None


# ---------------------------------------------------------------------------
# Classification — raw thresholds regardless of state
# ---------------------------------------------------------------------------


class TestClassificationPassthrough:
    """Classification: raw thresholds regardless of metric or target_std."""

    def test_classification_returns_raw_thresholds(self):
        eff_var, eff_margin, warning = _effective_thresholds(
            _clf_config("auc"), _state(500.0)
        )
        assert eff_var == 0.01
        assert eff_margin == 0.001
        assert warning is None

    def test_classification_no_warning_returned(self):
        """Even with zero target_std, classification must not trigger the
        degenerate-target warning — the zero-std branch is regression-only."""
        _, _, warning = _effective_thresholds(_clf_config("f1"), _state(0.0))
        assert warning is None

    def test_f1_classification_raw(self):
        eff_var, eff_margin, warning = _effective_thresholds(
            _clf_config("f1_score"), _state(1.0)
        )
        assert eff_var == 0.01
        assert eff_margin == 0.001
        assert warning is None
