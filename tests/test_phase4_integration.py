import json
import pytest

from zindian.skills import skill_17_governance as gov


def make_comp(tmp_path, slug="cmp-gov"):
    comp = tmp_path / "competitions" / slug
    comp.mkdir(parents=True)
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

    config = {"slug": "cmp-gov-ok"}
    state = {
        "human_gate_1_approved": "2026-06-01T00:00:00",
        "human_gate_2_approved": "2026-06-01T00:00:00",
        "human_gate_3_approved": "2026-06-01T00:00:00",
        "human_gate_4_approved": "2026-06-01T00:00:00",
        "scored_submissions": [
            {"filename": "sub1.csv", "score": 0.9, "date": "2026-05-01"},
            {"filename": "sub2.csv", "score": 0.8, "date": "2026-05-02"},
        ],
    }

    # bypass interactive selection with deterministic choices
    monkeypatch.setattr(
        gov, "_human_selection_prompt", lambda scored, current: [scored[0], scored[1]]
    )

    res_state = gov.run(config, state)
    assert res_state.get("selected_submissions_final") is True
    assert len(res_state.get("selected_submissions") or []) == 2
    assert res_state["selected_submissions"][0]["filename"] == "sub1.csv"
    assert res_state["selected_submissions"][1]["filename"] == "sub2.csv"

    # Verify report is written
    report_file = comp / "reports" / "final_selections.json"
    assert report_file.exists()
    report_data = json.loads(report_file.read_text(encoding="utf-8"))
    assert report_data["slug"] == "cmp-gov-ok"
    assert len(report_data["selections"]) == 2


def test_governance_raises_on_missing_gates(tmp_path, monkeypatch):
    make_comp(tmp_path, "cmp-gov-missing")
    monkeypatch.chdir(tmp_path)

    config = {"slug": "cmp-gov-missing"}
    state = {
        "human_gate_1_approved": "2026-06-01T00:00:00",
        "human_gate_2_approved": "2026-06-01T00:00:00",
        # human_gate_3_approved is missing!
        "human_gate_4_approved": "2026-06-01T00:00:00",
    }

    with pytest.raises(RuntimeError, match="Missing prerequisite human gate approvals"):
        gov.run(config, state)
