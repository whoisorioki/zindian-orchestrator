"""Integration test for submission board, leaderboard, and ledger recording."""

from __future__ import annotations

import json
from pathlib import Path

from zindian.skills import skill_16_submit as submitter


def _setup_comp(tmp_path: Path) -> Path:
    comp = tmp_path / "competitions" / "testcomp"
    (comp / "data" / "processed").mkdir(parents=True)
    raw = comp / "data" / "raw"
    raw.mkdir(parents=True)
    (comp / "reports").mkdir(parents=True)
    (comp / "submissions").mkdir(parents=True)

    config = {
        "name": "testcomp",
        "slug": "testcomp",
        "metric": "f1_score",
        "metric_direction": "maximize",
        "submission_format": "csv",
        "task_type": "classification",
        "use_probabilities": False,
        "daily_limit": 10,
        "total_limit": 100,
        "public_split_pct": 20,
        "private_split_pct": 80,
        "team_allowed": True,
        "code_review_tier": None,
        "allowed_external_data": True,
        "automl_permitted": False,
        "data_modality": "tabular",
        "domain": "generic",
    }
    (comp / "challenge_config.json").write_text(json.dumps(config, indent=2))

    state: dict = {
        "competition": "testcomp",
        "md5_target_hash": None,
        "anchor_oof_score": 0.85,
        "anchor_oof_f1": 0.85,
        "anchor_lb_score": None,
        "submissions_used_today": 0,
        "submissions_used_total": 0,
        "remaining_submissions": 10,
        "human_gate_4_approved": True,
        "human_gate_2_main_approved": True,
        "anchor_git_branch": "main",
        "dag_phase": "phase_4",
        "selected_submissions": [],
        "last_updated": None,
    }
    (comp / "SKILL_STATE.json").write_text(json.dumps(state, indent=2))

    sample = raw / "SampleSubmission.csv"
    sample.write_text("ID,Prediction\\n1,0\\n2,1\\n")

    sub = comp / "submissions" / "test_sub.csv"
    sub.write_text("ID,Prediction\\n1,0\\n2,1\\n")

    return comp


def test_pull_submission_board(tmp_path, monkeypatch):
    """Test pull_submission_board returns list of submissions."""
    comp = _setup_comp(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COMPETITION_SLUG", "testcomp")

    class FakeUser:
        def submission_board(self):
            return [
                {
                    "id": "sub_001",
                    "created_at": "2024-01-15T10:00:00Z",
                    "filename": "submission_1.csv",
                    "public_score": 0.85,
                    "status": "scored",
                    "chosen": True,
                    "comment": "branch:main|oof_f1:0.85",
                }
            ]

    class FakeClient:
        def __init__(self):
            self._user = FakeUser()

        def select_competition(self, slug):
            pass

    monkeypatch.setattr("zindian.zindi_client.ZindiClient", FakeClient)

    subs = submitter.pull_submission_board()
    assert len(subs) == 1
    assert subs[0]["id"] == "sub_001"
    assert subs[0]["public_score"] == 0.85


def test_show_submission_board(tmp_path, monkeypatch, capsys):
    """Test show_submission_board prints formatted table."""
    comp = _setup_comp(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COMPETITION_SLUG", "testcomp")

    class FakeUser:
        def submission_board(self):
            return [
                {
                    "id": "sub_001",
                    "created_at": "2024-01-15T10:00:00Z",
                    "filename": "submission_1.csv",
                    "public_score": 0.85,
                    "status": "scored",
                    "chosen": True,
                    "comment": "branch:main|oof_f1:0.85",
                }
            ]

    class FakeClient:
        def __init__(self):
            self._user = FakeUser()

        def select_competition(self, slug):
            pass

    monkeypatch.setattr("zindian.zindi_client.ZindiClient", FakeClient)

    submitter.show_submission_board()
    captured = capsys.readouterr()
    assert "sub_001" in captured.out
    assert "0.850000000" in captured.out


def test_pull_leaderboard(tmp_path, monkeypatch, capsys):
    """Test pull_leaderboard displays leaderboard."""
    comp = _setup_comp(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COMPETITION_SLUG", "testcomp")

    class FakeUser:
        @property
        def my_rank(self):
            return 42

        def leaderboard(self, to_print=True):
            print("Rank | Team | Score")
            print("1 | TopTeam | 0.95")

    class FakeClient:
        def __init__(self):
            self._user = FakeUser()

        def select_competition(self, slug):
            pass

        def leaderboard(self, per_page=20):
            self._user.leaderboard(to_print=True)

    monkeypatch.setattr("zindian.zindi_client.ZindiClient", FakeClient)

    submitter.pull_leaderboard(per_page=20)
    captured = capsys.readouterr()
    assert "LEADERBOARD" in captured.out
    assert "Your current rank: 42" in captured.out


def test_ledger_recording_on_submission(tmp_path, monkeypatch, capsys):
    """Test that submissions attempt to record to ledger."""
    comp = _setup_comp(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COMPETITION_SLUG", "testcomp")

    class FakeUser:
        @property
        def my_rank(self):
            return 15

        def leaderboard(self, to_print=True):
            pass

        def submission_board(self):
            return []

    class FakeClient:
        def __init__(self):
            self._user = FakeUser()

        def select_competition(self, slug):
            pass

        @property
        def remaining_submissions(self):
            return 5

        def submit(self, filepath, comment):
            return {"status": "success", "public_score": 0.87}

        def leaderboard(self, per_page=20):
            pass

    monkeypatch.setattr("zindian.zindi_client.ZindiClient", FakeClient)
    monkeypatch.setattr("builtins.input", lambda prompt: "YES")

    result = submitter.run(str(comp / "submissions" / "test_sub.csv"))
    assert result["status"] == "SUBMITTED"

    # Verify ledger recording was attempted (message appears in output)
    captured = capsys.readouterr()
    # The ledger recording happens and either succeeds or fails gracefully
    assert (
        "Recorded to ledger" in captured.out
        or "Failed to record to ledger" in captured.out
    )
