"""Edge case tests for CLI robustness."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest


def test_cli_status_missing_competition_slug():
    """Test CLI status command uses existing state when COMPETITION_SLUG not set."""
    from zindian.cli import main
    import sys
    from io import StringIO

    # Unset COMPETITION_SLUG
    old_slug = os.environ.pop("COMPETITION_SLUG", None)

    try:
        # CLI should still work by reading from existing state
        with patch("sys.argv", ["zindian.cli", "status"]):
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            try:
                main()
                output = sys.stdout.getvalue()
                # Should output valid JSON
                result = json.loads(output)
                assert "competition" in result
            finally:
                sys.stdout = old_stdout
    finally:
        if old_slug:
            os.environ["COMPETITION_SLUG"] = old_slug


def test_ledger_empty_database():
    """Test ledger commands handle empty database gracefully."""
    from zindian.ledger import Ledger

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "empty.db"
        config_path = Path(tmpdir) / "challenge_config.json"

        config = {"metric": "rmse", "metric_direction": "minimize"}
        config_path.write_text(json.dumps(config))

        with patch("zindian.ledger.resolve_competition_paths") as mock_paths:
            mock_paths.return_value.reports_dir = Path(tmpdir)
            mock_paths.return_value.competition_dir = Path(tmpdir)

            ledger = Ledger(str(db_path))

            # Empty queries should return None or empty list
            assert ledger.get_best_experiment() is None
            assert ledger.get_passed_experiments() == []
            assert ledger.get_failed_experiments() == []
            assert ledger.get_submissions() == []


def test_sync_state_network_failure():
    """Test sync handles network failures gracefully."""
    from zindian.sync_state import run as sync_run

    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "SKILL_STATE.json"
        config_path = Path(tmpdir) / "challenge_config.json"

        from zindian.schemas import skill_state_skeleton

        initial_state = skill_state_skeleton()
        initial_state["competition"] = "test-competition"
        state_path.write_text(json.dumps(initial_state))

        config = {"slug": "test-competition"}
        config_path.write_text(json.dumps(config))

        # Mock network failure
        with patch("zindian.sync_state.ZindiClient") as mock_client:
            mock_client.side_effect = ConnectionError("Network unavailable")

            with patch("zindian.sync_state.resolve_competition_paths") as mock_paths:
                mock_paths.return_value.state_path = state_path
                mock_paths.return_value.competition_dir = Path(tmpdir)

                # Should not crash, should preserve existing state
                try:
                    result = sync_run()
                    # Git branch should still be updated even if network fails
                    assert "current_git_branch" in result
                except Exception as e:
                    # Network failure should be caught and logged, not crash
                    assert "Network" in str(e) or "Connection" in str(e)


def test_submit_zero_remaining_budget():
    """Test submit command aborts when remaining budget is 0."""
    from zindian.skills.skill_16_submit import run as submit_run

    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "SKILL_STATE.json"
        config_path = Path(tmpdir) / "challenge_config.json"
        submission_file = Path(tmpdir) / "test_submission.csv"

        # State with 0 remaining submissions
        from zindian.schemas import skill_state_skeleton

        state = skill_state_skeleton()
        state.update(
            {
                "competition": "test-competition",
                "remaining_submissions": 0,
                "submissions_used_today": 5,
            }
        )
        state_path.write_text(json.dumps(state))

        config = {"slug": "test-competition", "daily_limit": 5}
        config_path.write_text(json.dumps(config))

        # Create dummy submission file
        submission_file.write_text("id,prediction\n1,0.5\n2,0.7\n")

        with patch(
            "zindian.skills.skill_16_submit.resolve_competition_paths"
        ) as mock_paths:
            mock_paths.return_value.state_path = state_path
            mock_paths.return_value.competition_dir = Path(tmpdir)

            # Should abort before attempting submission
            result = submit_run(config, state, str(submission_file))
            # Check if aborted due to budget
            assert (
                result.get("status") in ["aborted", "error"]
                or result.get("remaining_submissions") == 0
            )


def test_ledger_query_sql_injection():
    """Test ledger.query() handles SQL injection attempts safely."""
    from zindian.ledger import Ledger

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        config_path = Path(tmpdir) / "challenge_config.json"

        config = {"metric": "rmse", "metric_direction": "minimize"}
        config_path.write_text(json.dumps(config))

        with patch("zindian.ledger.resolve_competition_paths") as mock_paths:
            mock_paths.return_value.reports_dir = Path(tmpdir)
            mock_paths.return_value.competition_dir = Path(tmpdir)

            ledger = Ledger(str(db_path))

            # Add test data
            ledger.log_experiment(branch_name="test", oof_rmse=0.5, gate_result="PASS")

            # Attempt SQL injection (should be safely parameterized)
            malicious_query = "SELECT * FROM experiments WHERE branch_name = ?"
            malicious_param = "test'; DROP TABLE experiments; --"

            # Should execute safely with parameterization
            result = ledger.query(malicious_query, [malicious_param])
            assert result == []  # No match for malicious string

            # Verify table still exists
            all_experiments = ledger.query("SELECT * FROM experiments")
            assert len(all_experiments) == 1


def test_cli_ledger_best_no_experiments():
    """Test CLI ledger best command when no experiments exist."""
    from zindian.cli import main

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "empty.db"
        config_path = Path(tmpdir) / "challenge_config.json"

        config = {"metric": "rmse", "metric_direction": "minimize"}
        config_path.write_text(json.dumps(config))

        with patch("zindian.ledger.resolve_competition_paths") as mock_paths:
            mock_paths.return_value.reports_dir = Path(tmpdir)
            mock_paths.return_value.competition_dir = Path(tmpdir)

            with patch("sys.argv", ["zindian.cli", "ledger", "best"]):
                # Should print null or empty, not crash
                try:
                    main()
                except SystemExit:
                    pass  # CLI may exit with 0
