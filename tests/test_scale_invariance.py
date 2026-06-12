import numpy as np
from zindian.skills.skill_11_gate import _effective_thresholds, _fold_score_variance
from zindian.skills.skill_12_metric import run as run_skill_12


class DummyConfig:
    def __init__(self, data):
        self.data = data

    def get(self, key, default=None):
        return self.data.get(key, default)


def test_effective_thresholds_regression():
    config = DummyConfig(
        {
            "task_type": "regression",
            "variance_gate_threshold": 0.01,
            "gate_margin": 0.005,
        }
    )
    state = {"eda": {"target_std": 2.5}}

    # Equation:
    # effective_variance_threshold = variance_gate_threshold * (target_std ** 2)
    # effective_gate_margin = gate_margin * target_std
    eff_var, eff_margin = _effective_thresholds(config, state)

    assert np.isclose(eff_var, 0.01 * (2.5**2))
    assert np.isclose(eff_margin, 0.005 * 2.5)


def test_effective_thresholds_classification():
    config = DummyConfig(
        {
            "task_type": "classification",
            "variance_gate_threshold": 0.01,
            "gate_margin": 0.005,
        }
    )
    state = {"eda": {"target_std": 2.5}}  # Should be ignored for classification

    eff_var, eff_margin = _effective_thresholds(config, state)

    assert np.isclose(eff_var, 0.01)
    assert np.isclose(eff_margin, 0.005)


def test_fold_score_variance_unbiased_sample():
    # ddof=1 is used for fold score variance calculations
    scores = [0.82, 0.84, 0.81, 0.85, 0.83]
    state = {"eda": {"fold_scores": scores}}

    # Calculate expected unbiased sample variance manually
    expected_variance = np.var(scores, ddof=1)

    actual_variance = _fold_score_variance(state)
    assert np.isclose(actual_variance, expected_variance)


def test_skill_12_variance_ddof():
    # Verify skill_12 metric calculation uses ddof=1
    config = {
        "slug": "ey-frogs",
        "task_type": "regression",
        "metric": "rmse",
        "metric_direction": "minimize",
    }
    state = {"eda": {"fold_scores": [0.82, 0.84, 0.81, 0.85, 0.83], "target_std": 2.5}}
    updated_state = run_skill_12(config, state)

    scores = state["eda"]["fold_scores"]
    expected_variance = np.var(scores, ddof=1)

    actual_variance = updated_state["metric_analysis"]["fold_score_variance"]
    assert np.isclose(actual_variance, expected_variance)
