"""Unit tests for the [v2.7] composite `se_oof` in skill_12.

Covers the two hard rules locked in the SoT:
  - Fix-1: composite_se_oof = sqrt( sum(w_eff * per-target NB variance) ) over
    regression targets, with NO /sqrt(K) — the A.1 double-scaling error.
  - Fix-3: the composite sums regression targets ONLY; classification-target
    variance must not shift it.
Also asserts per-target `se_oof` is emitted for every target (L2).
"""

import numpy as np
import pytest

from zindian.skills.skill_12_metric import _get_nb_factor, run


REG_FOLDS = [0.5, 0.52, 0.48, 0.51, 0.49]
CLS_FOLDS = [0.8, 0.82, 0.79, 0.81, 0.80]


def _nb_variance(fold_scores, K):
    sample_var = float(np.var(fold_scores, ddof=1))
    return float(sample_var * _get_nb_factor(K, None))


def _make_config():
    return {
        "target_config": {
            "targets": [
                {"name": "goals", "task_type": "regression",
                 "metric": "rmse", "weight": 0.6},
                {"name": "label", "task_type": "classification",
                 "metric": "f1", "weight": 0.4},
            ]
        },
        "metric": "composite",
    }


def _make_state(reg_folds, cls_folds):
    return {
        "best_variant_this_round": "test",
        "branch_test_goals_oof": {"model_config": {"fold_scores": reg_folds}},
        "branch_test_label_oof": {"model_config": {"fold_scores": cls_folds}},
        "eda": {"goals_std": 2.5},
    }


def test_composite_se_oof_matches_hand_computed():
    ma = run(_make_config(), _make_state(REG_FOLDS, CLS_FOLDS))["metric_analysis"]
    expected = float(np.sqrt(0.6 * _nb_variance(REG_FOLDS, len(REG_FOLDS))))
    assert "composite_se_oof" in ma
    assert ma["composite_se_oof"] == pytest.approx(expected, rel=1e-9)


def test_composite_se_oof_has_no_extra_sqrt_K():
    ma = run(_make_config(), _make_state(REG_FOLDS, CLS_FOLDS))["metric_analysis"]
    buggy = float(
        np.sqrt(0.6 * _nb_variance(REG_FOLDS, len(REG_FOLDS)))
        / np.sqrt(len(REG_FOLDS))
    )
    assert ma["composite_se_oof"] != pytest.approx(buggy, rel=1e-9)


def test_composite_se_oof_insensitive_to_classification_variance():
    base = run(_make_config(), _make_state(REG_FOLDS, CLS_FOLDS))[
        "metric_analysis"
    ]["composite_se_oof"]
    shifted = run(
        _make_config(), _make_state(REG_FOLDS, [0.5, 0.9, 0.1, 0.99, 0.7])
    )["metric_analysis"]["composite_se_oof"]
    assert shifted == pytest.approx(base, rel=1e-9)


def test_per_target_se_oof_emitted_for_all_targets():
    per_target = run(_make_config(), _make_state(REG_FOLDS, CLS_FOLDS))[
        "metric_analysis"
    ]["per_target"]
    for name in ("goals", "label"):
        assert "se_oof" in per_target[name]
        expected = float(np.sqrt(per_target[name]["fold_score_variance"]))
        assert per_target[name]["se_oof"] == pytest.approx(expected, rel=1e-9)


def test_no_composite_se_oof_without_regression_target():
    config = {
        "target_config": {
            "targets": [
                {"name": "label_a", "task_type": "classification",
                 "metric": "f1", "weight": 0.5},
                {"name": "label_b", "task_type": "classification",
                 "metric": "f1", "weight": 0.5},
            ]
        },
        "metric": "composite",
    }
    state = {
        "best_variant_this_round": "test",
        "branch_test_label_a_oof": {"model_config": {"fold_scores": CLS_FOLDS}},
        "branch_test_label_b_oof": {
            "model_config": {"fold_scores": [0.7, 0.72, 0.71, 0.69, 0.70]}
        },
        "eda": {},
    }
    ma = run(config, state)["metric_analysis"]
    assert "composite_se_oof" not in ma