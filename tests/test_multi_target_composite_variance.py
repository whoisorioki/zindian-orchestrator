"""Test composite fold variance for multi-target competitions."""

import numpy as np
from zindian.skills.skill_12_metric import run


def test_composite_fold_variance_multi_target():
    """Compute weighted composite variance for multi-target."""
    config = {
        "target_config": {
            "targets": [
                {
                    "name": "goals",
                    "task_type": "regression",
                    "metric": "rmse",
                    "weight": 0.6,
                },
                {
                    "name": "label",
                    "task_type": "classification",
                    "metric": "f1",
                    "weight": 0.4,
                },
            ]
        },
        "metric": "composite",
    }

    state = {
        "best_variant_this_round": "test",
        "branch_test_goals_oof": {
            "model_config": {"fold_scores": [0.5, 0.52, 0.48, 0.51, 0.49]}
        },
        "branch_test_label_oof": {
            "model_config": {"fold_scores": [0.8, 0.82, 0.79, 0.81, 0.80]}
        },
        "eda": {"goals_std": 2.5},
    }

    result = run(config, state)

    # Verify composite variance computed
    assert "metric_analysis" in result
    metric_analysis = result["metric_analysis"]
    assert "composite_fold_score_variance" in metric_analysis
    assert metric_analysis["composite_fold_score_variance"] > 0

    # Verify variance uses ddof=1 (unbiased)
    composite_scores = []
    for i in range(5):
        # Regression: normalize by std, convert to 0-1 scale
        reg_score = state["branch_test_goals_oof"]["model_config"]["fold_scores"][i]
        normalized = reg_score / 2.5
        reg_val = max(0.0, 1.0 - normalized)

        # Classification: use raw score
        cls_score = state["branch_test_label_oof"]["model_config"]["fold_scores"][i]

        # Weighted composite
        composite = 0.6 * reg_val + 0.4 * cls_score
        composite_scores.append(composite)

    expected_variance = float(np.var(composite_scores, ddof=1))
    assert (
        abs(metric_analysis["composite_fold_score_variance"] - expected_variance) < 1e-6
    )


def test_single_target_backward_compatibility():
    """Single-target competitions should work as before."""
    config = {"metric": "f1"}

    state = {
        "best_variant_this_round": "anchor",
        "branch_anchor_oof": {
            "model_config": {
                "fold_scores": [0.8, 0.82, 0.79, 0.81, 0.80],
                "threshold": 0.5,
            }
        },
    }

    result = run(config, state)

    assert "metric_analysis" in result
    metric_analysis = result["metric_analysis"]
    assert "fold_score_variance" in metric_analysis
    assert metric_analysis["fold_score_variance"] > 0
    # Should NOT have composite variance for single target
    assert (
        "composite_fold_score_variance" not in metric_analysis
        or metric_analysis.get("composite_fold_score_variance") is None
    )
