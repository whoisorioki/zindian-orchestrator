"""Test skill_00_zindi_monitor writes only to SKILL_STATE.json, not challenge_config.json."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from zindian.zindi_monitor_core import update_state
from zindian.schemas import skill_state_skeleton


def test_monitor_writes_community_signals_to_state_only():
    """Test monitor writes community_signals to SKILL_STATE.json, not config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "SKILL_STATE.json"
        config_path = Path(tmpdir) / "challenge_config.json"

        # Create valid initial state
        initial_state = skill_state_skeleton()
        initial_state.update(
            {
                "competition": "test-competition",
                "dag_phase": "phase_1",
                "anchor_oof_score": 0.5545,
                "drift_threshold": 0.05,
            }
        )
        state_path.write_text(json.dumps(initial_state))

        # Create config
        config = {
            "metric": "rmsle",
            "metric_direction": "minimize",
            "drift_threshold": 0.05,
        }
        config_path.write_text(json.dumps(config))

        # Mock paths
        mock_paths = MagicMock()
        mock_paths.state_path = state_path
        mock_paths.competition_dir = Path(tmpdir)

        # Mock data
        comp_intel = {"metric": "rmsle", "external_banned": True}
        lb_intel = {"my_rank": 25, "remaining": 3}
        sub_intel = {"best_score": 0.552}
        flagged: list[dict] = [
            {
                "title": "External data clarification",
                "published": "2024-01-15",
                "url": "https://zindi.africa/discussions/123",
                "classification": "clarify",
                "external_sources": ["worldclim"],
                "resolved_by_organizer": True,
            }
        ]
        all_discussions: list[dict] = [{"title": "Test", "published_at": "2024-01-15"}]

        # Call update_state
        with patch("zindian.zindi_monitor_core.ChallengeConfig") as mock_config:
            mock_config.load.return_value = config
            update_state(
                comp_intel, lb_intel, sub_intel, flagged, all_discussions, mock_paths
            )

        # Verify state was updated with community_signals
        updated_state = json.loads(state_path.read_text())
        assert "community_signals" in updated_state
        assert len(updated_state["community_signals"]) == 1
        assert (
            updated_state["community_signals"][0]["title"]
            == "External data clarification"
        )
        assert updated_state["community_signals"][0]["classification"] == "clarify"

        # Verify config was NOT modified
        final_config = json.loads(config_path.read_text())
        assert final_config == config
        assert "community_signals" not in final_config


def test_monitor_does_not_write_compliance_to_state():
    """Test monitor no longer writes compliance dict to state (replaced by community_signals)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "SKILL_STATE.json"
        config_path = Path(tmpdir) / "challenge_config.json"

        initial_state = skill_state_skeleton()
        initial_state.update(
            {
                "competition": "test-competition",
                "dag_phase": "phase_1",
                "anchor_oof_score": 0.5545,
            }
        )
        state_path.write_text(json.dumps(initial_state))

        config = {
            "metric": "rmsle",
            "metric_direction": "minimize",
            "drift_threshold": 0.05,
        }
        config_path.write_text(json.dumps(config))

        mock_paths = MagicMock()
        mock_paths.state_path = state_path
        mock_paths.competition_dir = Path(tmpdir)

        comp_intel = {"metric": "rmsle"}
        lb_intel = {"my_rank": 25, "remaining": 3}
        sub_intel = {"best_score": 0.552}
        flagged: list[dict] = []
        all_discussions: list[dict] = []

        with patch("zindian.zindi_monitor_core.ChallengeConfig") as mock_config:
            mock_config.load.return_value = config
            update_state(
                comp_intel, lb_intel, sub_intel, flagged, all_discussions, mock_paths
            )

        updated_state = json.loads(state_path.read_text())
        # Old compliance dict should not exist
        assert "compliance" not in updated_state
        # New community_signals should exist (empty list)
        assert "community_signals" in updated_state
        assert updated_state["community_signals"] == []


def test_monitor_preserves_existing_state_fields():
    """Test monitor preserves existing state fields when updating."""
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "SKILL_STATE.json"
        config_path = Path(tmpdir) / "challenge_config.json"

        initial_state = skill_state_skeleton()
        initial_state.update(
            {
                "competition": "test-competition",
                "dag_phase": "phase_3_anchor_promoted",
                "anchor_oof_score": 0.5545,
                "submissions_used_today": 10,
                "selected_submissions": [1, 2],
            }
        )
        state_path.write_text(json.dumps(initial_state))

        config = {
            "metric": "rmsle",
            "metric_direction": "minimize",
            "drift_threshold": 0.05,
        }
        config_path.write_text(json.dumps(config))

        mock_paths = MagicMock()
        mock_paths.state_path = state_path
        mock_paths.competition_dir = Path(tmpdir)

        comp_intel = {"metric": "rmsle"}
        lb_intel = {"my_rank": 30, "remaining": 2}
        sub_intel = {"best_score": 0.560}
        flagged: list[dict] = []
        all_discussions: list[dict] = []

        with patch("zindian.zindi_monitor_core.ChallengeConfig") as mock_config:
            mock_config.load.return_value = config
            update_state(
                comp_intel, lb_intel, sub_intel, flagged, all_discussions, mock_paths
            )

        updated_state = json.loads(state_path.read_text())
        # Existing fields preserved
        assert updated_state["dag_phase"] == "phase_3_anchor_promoted"
        assert updated_state["anchor_oof_score"] == 0.5545
        assert updated_state["submissions_used_today"] == 10
        assert updated_state["selected_submissions"] == [1, 2]
        # New fields added
        assert updated_state["anchor_rank"] == 30
        assert updated_state["remaining_submissions"] == 2
        assert "community_signals" in updated_state
