import numpy as np
from zindian.config import ChallengeConfig
from zindian.skills.skill_11_gate import _effective_thresholds, _fold_score_variance
from zindian.skills.skill_12_metric import run as run_skill_12


def _make_config(data: dict) -> ChallengeConfig:
    """Build a minimal ChallengeConfig from a plain dict for testing."""
    from pathlib import Path
    import json
    import tempfile

    tmp = Path(tempfile.mktemp(suffix=".json"))
    # Ensure required fields are present so ChallengeConfig.load won't raise
    full = {
        "metric": data.get("metric", "f1"),
        "metric_direction": data.get("metric_direction", "maximize"),
        "use_probabilities": False,
        "automl_permitted": False,
        "data_modality": "tabular",
        **data,
    }
    tmp.write_text(json.dumps(full))
    return ChallengeConfig(path=tmp, _data=full)


def test_effective_thresholds_regression():
    config = _make_config(
        {
            "task_type": "regression",
            "variance_gate_threshold": 0.01,
            "gate_margin": 0.005,
        }
    )
    state = {"eda": {"target_std": 2.5}}

    eff_var, eff_margin, warning = _effective_thresholds(config, state)

    assert np.isclose(eff_var, 0.01 * (2.5**2))
    assert np.isclose(eff_margin, 0.005 * 2.5)
    assert warning is None


def test_effective_thresholds_classification():
    config = _make_config(
        {
            "task_type": "classification",
            "variance_gate_threshold": 0.01,
            "gate_margin": 0.005,
        }
    )
    state = {"eda": {"target_std": 2.5}}

    eff_var, eff_margin, warning = _effective_thresholds(config, state)

    assert np.isclose(eff_var, 0.01)
    assert np.isclose(eff_margin, 0.005)
    assert warning is None


def test_fold_score_variance_unbiased_sample():
    scores = [0.82, 0.84, 0.81, 0.85, 0.83]
    state = {"eda": {"fold_scores": scores}}

    expected_variance = np.var(np.array(scores, dtype=np.float64), ddof=1)

    actual_variance = _fold_score_variance(state)
    assert actual_variance is not None
    assert np.isclose(actual_variance, expected_variance)


def test_skill_12_variance_ddof():
    config = {
        "slug": "ey-frogs",
        "task_type": "regression",
        "metric": "rmse",
        "metric_direction": "minimize",
    }
    state = {"eda": {"fold_scores": [0.82, 0.84, 0.81, 0.85, 0.83], "target_std": 2.5}}
    updated_state = run_skill_12(config, state)

    scores = np.array(state["eda"]["fold_scores"], dtype=np.float64)
    expected_variance = np.var(scores, ddof=1)

    actual_variance = float(updated_state["metric_analysis"]["fold_score_variance"])
    assert np.isclose(actual_variance, expected_variance)
