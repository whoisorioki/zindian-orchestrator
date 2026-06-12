from zindian.skills.skill_11_gate import _baseline_score


def test_baseline_score_resolution_retraining_active():
    """
    Injects a state state-tree where anchor_challenge.active == True and
    pseudo_label_result.retraining_required == True, and asserts that
    the module checks for anchor_oof_{metric}_augmented as the calculation
    baseline instead of anchor_oof_{metric}_challenged or the default anchor_oof_{metric}.
    """
    state = {
        "pseudo_label_result": {"retraining_required": True},
        "anchor_challenge": {"active": True},
        "anchor_oof_rmse_augmented": 0.75,
        "anchor_oof_rmse_challenged": 0.80,
        "anchor_oof_rmse": 0.85,
    }

    score, key = _baseline_score(state, "rmse")
    assert key == "anchor_oof_rmse_augmented"
    assert score == 0.75
