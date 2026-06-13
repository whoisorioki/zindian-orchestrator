from zindian.skills.skill_22_reproducibility_audit import _audit_oof_strategy_tags


def test_audit_oof_strategy_tags_prefix_normalization():
    state = {
        "branch_main_oof": {"cv_strategy_id": "stratifiedkfold", "scores": [1.0, 2.0]},
        "branch_dev_oof": {
            "cv_strategy_id": "config:stratifiedkfold",
            "scores": [2.0, 3.0],
        },
        "branch_other_oof": {
            "cv_strategy_id": "override:stratifiedkfold",
            "scores": [3.0, 4.0],
        },
    }

    # All of the above should match "config:stratifiedkfold" under normalization
    ok, issues = _audit_oof_strategy_tags(state, "config:stratifiedkfold")
    assert ok
    assert not issues

    # All of the above should match "override:stratifiedkfold" under normalization
    ok, issues = _audit_oof_strategy_tags(state, "override:stratifiedkfold")
    assert ok
    assert not issues

    # All of the above should match "stratifiedkfold" under normalization
    ok, issues = _audit_oof_strategy_tags(state, "stratifiedkfold")
    assert ok
    assert not issues

    # Mismatch case
    ok, issues = _audit_oof_strategy_tags(state, "config:kfold")
    assert not ok
    assert len(issues) == 3
