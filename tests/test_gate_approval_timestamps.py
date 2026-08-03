"""
Unit tests for F7 (Gate approval timestamps).

Verifies that prompt_human_gate writes ISO 8601 timestamps when approving gates.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from datetime import datetime

from zindian import orchestrator
from zindian.state import SkillStateStore


def test_gate_approval_timestamps(tmp_path: Path, monkeypatch):
    """Verifies that human_gate_X_approved_at timestamps are written upon gate approval."""
    state_file = tmp_path / "SKILL_STATE.json"
    initial_state = {
        "competition": "test-slug",
        "md5_target_hash": "dummy",
        "anchor_oof_score": 0.85,
        "anchor_lb_score": None,
        "submissions_used_today": 0,
        "submissions_used_total": 0,
        "remaining_submissions": 10,
        "dag_phase": "phase_2_complete",
        "selected_submissions": [],
        "last_updated": "2026-08-03T00:00:00Z",
    }
    state_file.write_text(json.dumps(initial_state))
    store = SkillStateStore(state_file)
    state = store.read()

    config_mock = type("Config", (), {"_data": {"cv_strategy": {"type": "KFold"}}})()

    # Simulate interactive prompt choice "A" for Gate 1
    monkeypatch.setattr("builtins.input", lambda prompt="": "A")
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    result = orchestrator.prompt_human_gate(
        gate_name="Gate 1",
        store=store,
        state=state,
        config=config_mock,
        non_interactive=False,
    )
    assert result is True

    updated_state = store.read()
    assert updated_state.get("human_gate_1_approved") is True
    ts = updated_state.get("human_gate_1_approved_at")
    assert ts is not None
    # Validate ISO 8601 parsing
    parsed_dt = datetime.fromisoformat(ts)
    assert parsed_dt is not None
