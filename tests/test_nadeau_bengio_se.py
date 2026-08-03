import pytest
import numpy as np
from zindian.skills import skill_12_metric


def test_nadeau_bengio_se_exact_value():
    # S1 - implemented 2026-08-03
    state = {
        "best_variant_this_round": "test-branch",
        "branch_test-branch_oof": {
            "model_config": {"fold_scores": [0.8, 0.82, 0.79, 0.81, 0.83]}
        },
    }
    config = {"metric": "f1", "target_config": {"targets": []}}

    # Run metric analysis in-memory
    result = skill_12_metric.run(config=config, state=state)
    analysis = result["metric_analysis"]

    # Extract calculated metrics
    fold_scores = analysis["fold_scores"]
    fold_score_variance_sample = analysis["fold_score_variance_sample"]
    fold_score_variance_nb = analysis["fold_score_variance_nb"]
    se_oof = analysis["se_oof"]

    # Hand-computed values for [0.8, 0.82, 0.79, 0.81, 0.83]
    # Mean = 0.81
    # Sample Variance (ddof=1) = 0.00025
    # nb_factor = 1/5 + 1/4 = 0.45
    # nb_variance = 0.00025 * 0.45 = 0.0001125
    # se_oof = sqrt(nb_variance) = 0.010606601717798213

    assert fold_scores == [0.8, 0.82, 0.79, 0.81, 0.83]
    assert pytest.approx(fold_score_variance_sample) == 0.00025
    assert pytest.approx(fold_score_variance_nb) == 0.0001125
    assert pytest.approx(se_oof) == np.sqrt(0.0001125)
