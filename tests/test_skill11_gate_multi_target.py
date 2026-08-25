"""[v2.7 / H1] multi-target gate enforcement tests. Direct _run_multi_target_gate
calls with a MagicMock store. Confirms states that previously passed the bypass
logic are now blocked by the variance gate alone and the baseline gate alone,
and that retraining consumes anchor_oof_score_augmented."""

from unittest.mock import MagicMock

from zindian.skills.skill_11_gate import _run_multi_target_gate


def _make_config():
    return {
        "task_type": "regression",
        "metric": "composite",
        "variance_gate_threshold": 0.01,
        "gate_margin": 0.001,
        "use_inverse_variance_weighting": False,
        "target_config": {
            "targets": [
                {"name": "goals", "task_type": "regression",
                 "metric": "rmse", "weight": 0.6},
                {"name": "label", "task_type": "classification",
                 "metric": "f1", "weight": 0.4},
            ]
        },
    }


def _make_state(**overrides):
    # reg distance = 0.5/2.5 = 0.2 ; cls distance = 1-0.8 = 0.2
    # avg_score = (0.2*0.6 + 0.2*0.4)/1.0 = 0.2
    state = {
        "best_variant_this_round": "variant-a",
        "feature_round": 1,
        "anchor_multi_target_metrics": {
            "goals": {"oof_rmse": 0.5},
            "label": {"oof_f1": 0.8},
        },
        "shap_multi_target_results": {
            "goals": {"pruning_pass": True},
            "label": {"pruning_pass": True},
        },
        "human_gate_2_variant-a_approved": True,
        "anchor_oof_score": 0.25,
        "anchor_oof_score_augmented": None,
        "anchor_oof_score_challenged": None,
        "eda": {"goals_std": 2.5},
        "metric_analysis": {
            "composite_fold_score_variance": 0.001,
            "composite_se_oof": None,
        },
    }
    state.update(overrides)
    return state


def _run(config, state):
    store = MagicMock()
    return _run_multi_target_gate(config, store, state), store


def test_variance_gate_blocks_state_that_would_formerly_pass():
    # composite variance 0.1 >= effective_threshold 0.0625 -> BLOCK
    config = _make_config()
    state = _make_state(
        metric_analysis={"composite_fold_score_variance": 0.10,
                         "composite_se_oof": None},
    )
    result, store = _run(config, state)
    assert result["status"] == "BLOCKED"
    assert result["reason"] == "variance gate failed"
    calls = store.update.call_args_list
    assert calls, "expected a store.update call"
    last_kw = calls[-1].kwargs
    assert last_kw["dag_phase"] == "phase_3_gate_blocked"
    assert last_kw["phase_3_gate_diagnosis"]["failure_reason"] == "variance gate failed"


def test_baseline_gate_blocks_with_variance_held_open():
    config = _make_config()
    state = _make_state(
        anchor_oof_score=0.20,  # == avg -> no improvement
        metric_analysis={"composite_fold_score_variance": 0.001,
                         "composite_se_oof": 0.0},
    )
    result, _ = _run(config, state)
    assert result["status"] == "BLOCKED"
    assert result["reason"] == "baseline gate failed"


def test_augmented_baseline_is_consumed_when_retraining():
    config = _make_config()
    state = _make_state(
        pseudo_label_result={"retraining_required": True},
        anchor_oof_score=0.25,            # favourable: 0.25-0.2 = 0.05
        anchor_oof_score_augmented=0.20,  # == avg -> baseline gate fails
        metric_analysis={"composite_fold_score_variance": 0.001,
                         "composite_se_oof": None},
    )
    result, _ = _run(config, state)
    assert result["status"] == "BLOCKED"
    assert result["reason"] == "baseline gate failed"
    assert result["diagnosis"]["baseline_key"] == "anchor_oof_score_augmented"


def test_gate_passes_when_all_conditions_met():
    config = _make_config()
    state = _make_state(
        anchor_oof_score=0.25,  # baseline - avg = 0.05 > margin
        metric_analysis={"composite_fold_score_variance": 0.001,
                         "composite_se_oof": 0.0},
    )
    result, store = _run(config, state)
    assert result["status"] == "PASS"
    assert result["new_branch"] == "anchor-multi-v2"
    final_kwargs = store.update.call_args_list[-1].kwargs
    assert final_kwargs["anchor_git_branch"] == "anchor-multi-v2"
    assert final_kwargs["phase_3_gate_diagnosis"]["passed"] is True


def test_composite_direction_is_minimize_lower_is_better():
    config = _make_config()
    fail_state = _make_state(
        anchor_oof_score=0.20,  # == avg, equal -> fail
        metric_analysis={"composite_fold_score_variance": 0.001,
                         "composite_se_oof": 0.0},
    )
    assert _run(config, fail_state)[0]["status"] == "BLOCKED"
    pass_state = _make_state(
        anchor_oof_score=0.30,  # 0.1 improvement -> pass
        metric_analysis={"composite_fold_score_variance": 0.001,
                         "composite_se_oof": 0.0},
    )
    assert _run(config, pass_state)[0]["status"] == "PASS"