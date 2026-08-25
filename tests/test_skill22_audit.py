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


def test_check_telemetry_aggregate():
    from zindian.skills.skill_22_reproducibility_audit import _check_telemetry_aggregate

    # 1. Successful case with non-null carbon
    state_ok = {
        "telemetry.aggregate": {
            "phase": "1",
            "total_duration_sec": 4.5,
            "total_carbon_kg_estimate": 0.00012,
            "skill_count": 2,
            "written_at": "2026-08-24T12:00:00Z"
        }
    }
    cfg_ok = {
        "infrastructure": {
            "tdp_watts": 15.0,
            "pue": 1.0,
            "carbon_intensity_gco2_per_kwh": 494.0
        }
    }
    ok, issues = _check_telemetry_aggregate(state_ok, cfg_ok)
    assert ok
    assert not issues

    # 2. Case where total_carbon_kg_estimate is None (null) and not instrumented is properly documented -> passes
    state_not_instrumented = {
        "telemetry.aggregate": {
            "phase": "1",
            "total_duration_sec": 4.5,
            "total_carbon_kg_estimate": None,
            "skill_count": 2,
            "written_at": "2026-08-24T12:00:00Z"
        },
        "telemetry.skill_01": {
            "tracker_method": "not_instrumented",
            "reason": "CodeCarbon and ML formulas failed"
        }
    }
    ok, issues = _check_telemetry_aggregate(state_not_instrumented, cfg_ok)
    assert ok
    assert not issues

    # 3. Mismatch/Fail: total_carbon_kg_estimate is None, but no skill is 'not_instrumented' with reason
    state_fail_no_reason = {
        "telemetry.aggregate": {
            "phase": "1",
            "total_duration_sec": 4.5,
            "total_carbon_kg_estimate": None,
            "skill_count": 2,
            "written_at": "2026-08-24T12:00:00Z"
        }
    }
    ok, issues = _check_telemetry_aggregate(state_fail_no_reason, cfg_ok)
    assert not ok
    assert any("valid 'reason'" in iss for iss in issues)

    # 4. Fail: missing infrastructure block
    ok, issues = _check_telemetry_aggregate(state_ok, {})
    assert not ok
    assert any("infrastructure" in iss for iss in issues)


def test_telemetry_aggregate_written(tmp_path, monkeypatch):
    import json
    from zindian.orchestrator import run_phase
    from zindian.state import SkillStateStore

    comp_dir = tmp_path / "comp"
    comp_dir.mkdir()
    state_path = comp_dir / "SKILL_STATE.json"

    # Write skeleton state
    from zindian.state import skill_state_skeleton
    state_data = skill_state_skeleton()
    state_path.write_text(json.dumps(state_data), encoding="utf-8")

    class MockPaths:
        competition_dir = comp_dir
        state_path = comp_dir / "SKILL_STATE.json"
        root = tmp_path
        reports_dir = comp_dir / "reports"

    MockPaths.reports_dir.mkdir()

    # Mock resolve_competition_paths
    monkeypatch.setattr(
        "zindian.paths.resolve_competition_paths",
        lambda *args, **kwargs: MockPaths
    )

    # Mock ChallengeConfig
    class MockConfig:
        _data = {
            "slug": "test-comp",
            "phase_skill_map": {"1": []},
            "infrastructure": {}
        }
        @classmethod
        def load(cls):
            return cls()
        def get(self, key, default=None):
            return self._data.get(key, default)
    monkeypatch.setattr("zindian.config.ChallengeConfig", MockConfig)

    # Ensure deduplication / summaries do not crash due to missing _current_run_dir
    monkeypatch.setattr("zindian.orchestrator._current_run_dir", None)

    # Run the pipeline phase 1
    run_phase("1")

    # Read state and check telemetry.aggregate
    store = SkillStateStore(state_path)
    state = store.read()
    assert "telemetry.aggregate" in state
    agg = state["telemetry.aggregate"]
    assert agg["phase"] == "1"
    assert agg["skill_count"] == 0
    assert "written_at" in agg
