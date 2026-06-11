import json
from pathlib import Path

import pytest

from zindian.skills import skill_17_governance as gov


def make_comp(tmp_path, slug="cmp-gov"):
    comp = tmp_path / "competitions" / slug
    comp.mkdir(parents=True)
    # minimal challenge_config
    cfg = {
        "slug": slug,
        "metric": "f1",
    }
    (comp / "challenge_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    state = {"current_git_branch": "main", "selected_submissions": []}
    (comp / "SKILL_STATE.json").write_text(json.dumps(state), encoding="utf-8")
    (comp / "reports").mkdir()
    return comp


def test_governance_runs_and_records_selection(tmp_path, monkeypatch):
    comp = make_comp(tmp_path, "cmp-gov-ok")
    monkeypatch.chdir(tmp_path)

    # fake scored submissions
    fake_scored = [
        {"id": "s1", "score": 0.9, "filename": "sub1.csv", "date": "2026-05-01"},
        {"id": "s2", "score": 0.8, "filename": "sub2.csv", "date": "2026-05-02"},
    ]

    monkeypatch.setattr(gov, "fetch_scored_submissions", lambda slug: fake_scored)
    # bypass interactive selection with deterministic choices
    monkeypatch.setattr(gov, "human_selection_gate", lambda scored, state: [fake_scored[0], fake_scored[1]])
    # force audit to pass
    from zindian.skills import skill_22_reproducibility_audit as audit
    monkeypatch.setattr(audit, "audit_pipeline", lambda slug: True)

    res = gov.run("cmp-gov-ok")
    assert res["status"] == "OK"

    state = json.loads((Path("competitions") / "cmp-gov-ok" / "SKILL_STATE.json").read_text(encoding="utf-8"))
    assert state.get("selected_submissions") == ["s1", "s2"]
    assert state.get("dag_phase") == "phase_5_governance_complete"


def test_governance_aborts_on_audit_failure(tmp_path, monkeypatch):
    comp = make_comp(tmp_path, "cmp-gov-fail")
    monkeypatch.chdir(tmp_path)

    fake_scored = [
        {"id": "s1", "score": 0.9, "filename": "sub1.csv", "date": "2026-05-01"},
        {"id": "s2", "score": 0.8, "filename": "sub2.csv", "date": "2026-05-02"},
    ]

    monkeypatch.setattr(gov, "fetch_scored_submissions", lambda slug: fake_scored)
    monkeypatch.setattr(gov, "human_selection_gate", lambda scored, state: [fake_scored[0], fake_scored[1]])
    from zindian.skills import skill_22_reproducibility_audit as audit
    monkeypatch.setattr(audit, "audit_pipeline", lambda slug: False)

    res = gov.run("cmp-gov-fail")
    assert res["status"] == "AUDIT_FAILED"

    state = json.loads((Path("competitions") / "cmp-gov-fail" / "SKILL_STATE.json").read_text(encoding="utf-8"))
    assert state.get("governance_audit_failed") is True
