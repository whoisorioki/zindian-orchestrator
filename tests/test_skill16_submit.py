from __future__ import annotations

import json
import pytest
from pathlib import Path
import pandas as pd

from zindian.skills import skill_16_submit as submitter
from zindian.skills.skill_16_submit import HardAbortException

def _setup_comp(tmp_path: Path) -> Path:
    comp = tmp_path / "competitions" / "subcomp"
    (comp / "data" / "processed").mkdir(parents=True)
    raw = comp / "data" / "raw"
    raw.mkdir(parents=True)
    (comp / "reports").mkdir(parents=True)
    (comp / "submissions").mkdir(parents=True)

    config = {
        "name": "subcomp",
        "slug": "subcomp",
        "metric": "f1_score",
        "metric_direction": "maximize",
        "submission_format": "csv",
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
        "task_type": "classification",
        "reproducibility": {"seed": 42},
        "gate_margin": 0.05,
        "variance_gate_threshold": 0.5,
    }
    (comp / "challenge_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    state = {
        "competition": "subcomp",
        "md5_target_hash": None,
        "anchor_oof_score": 0.80,
        "anchor_oof_f1": 0.80,
        "anchor_oof_rmse": None,
        "anchor_lb_score": None,
        "submissions_used_today": 0,
        "submissions_used_total": 0,
        "remaining_submissions": 10,
        "dag_phase": "phase_3_features",
        "selected_submissions": [],
        "last_updated": None,
        "human_gate_4_approved": True,
        "human_gate_2_main_approved": True,
        "anchor_git_branch": "main",
    }
    (comp / "SKILL_STATE.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    # Create dummy sample and submission files
    sample = raw / "SampleSubmission.csv"
    sample.write_text("ID,Prediction\n1,0\n2,1\n", encoding="utf-8")

    sub = comp / "submissions" / "sub.csv"
    sub.write_text("ID,Prediction\n1,0\n2,1\n", encoding="utf-8")

    return comp


def test_budget_remaining_zero_raises_hard_abort(tmp_path, monkeypatch):
    comp = _setup_comp(tmp_path)
    state_path = comp / "SKILL_STATE.json"
    
    # 1. State-side budget is 0
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["remaining_submissions"] = 0
    state_path.write_text(json.dumps(state), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COMPETITION_SLUG", "subcomp")

    # Mock ZindiClient to select competition and report remaining
    class FakeZindiClient:
        def select_competition(self, slug):
            pass
        @property
        def remaining_submissions(self):
            return 10  # platform says 10 but state says 0

    monkeypatch.setattr("zindian.zindi_client.ZindiClient", FakeZindiClient)

    with pytest.raises(HardAbortException, match="zero submissions remaining"):
        submitter.run(str(comp / "submissions" / "sub.csv"))

    # 2. Platform budget is 0
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["remaining_submissions"] = 10
    state_path.write_text(json.dumps(state), encoding="utf-8")

    class FakeZindiClientExhausted:
        def select_competition(self, slug):
            pass
        @property
        def remaining_submissions(self):
            return 0  # platform says 0

    monkeypatch.setattr("zindian.zindi_client.ZindiClient", FakeZindiClientExhausted)

    with pytest.raises(HardAbortException, match="zero remaining submissions today"):
        submitter.run(str(comp / "submissions" / "sub.csv"))


def test_budget_remaining_one_warns_and_prompts(tmp_path, monkeypatch, capsys):
    comp = _setup_comp(tmp_path)
    state_path = comp / "SKILL_STATE.json"
    
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["remaining_submissions"] = 10
    state_path.write_text(json.dumps(state), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COMPETITION_SLUG", "subcomp")

    class FakeZindiClientOne:
        def select_competition(self, slug):
            pass
        @property
        def remaining_submissions(self):
            return 1  # only 1 remaining
        def submit(self, filepath, comment):
            return {"status": "success"}

    monkeypatch.setattr("zindian.zindi_client.ZindiClient", FakeZindiClientOne)

    # Mock input to return YES so we proceed
    monkeypatch.setattr("builtins.input", lambda prompt: "YES")

    res = submitter.run(str(comp / "submissions" / "sub.csv"))
    assert res["status"] == "success" or "success" in str(res)

    captured = capsys.readouterr()
    assert "Only 1 live submission remaining today" in captured.out



def test_budget_remaining_two_proceeds_normally(tmp_path, monkeypatch):
    comp = _setup_comp(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COMPETITION_SLUG", "subcomp")

    class FakeZindiClientTwo:
        def select_competition(self, slug):
            pass
        @property
        def remaining_submissions(self):
            return 2  # exactly 2 remaining
        def submit(self, filepath, comment):
            return {"status": "success"}

    monkeypatch.setattr("zindian.zindi_client.ZindiClient", FakeZindiClientTwo)
    monkeypatch.setattr("builtins.input", lambda prompt: "YES")

    res = submitter.run(str(comp / "submissions" / "sub.csv"))
    assert res["status"] == "success" or "success" in str(res)
