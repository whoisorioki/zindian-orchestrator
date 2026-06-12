from __future__ import annotations

import json
from pathlib import Path


def _make_phase3_comp(tmp_path: Path) -> Path:
    comp = tmp_path / "competitions" / "phase3-cmp"
    (comp / "data" / "processed").mkdir(parents=True)
    (comp / "data" / "raw").mkdir(parents=True)
    (comp / "reports").mkdir(parents=True)
    (comp / "submissions").mkdir(parents=True)

    config = {
        "name": "phase3-test",
        "slug": "phase3-cmp",
        "metric": "f1_score",
        "metric_direction": "maximize",
        "submission_format": None,
        "use_probabilities": False,
        "daily_limit": None,
        "total_limit": None,
        "public_split_pct": None,
        "private_split_pct": None,
        "team_allowed": None,
        "code_review_tier": None,
        "allowed_external_data": True,
        "automl_permitted": False,
        "data_modality": "tabular",
        "domain": None,
        "task_type": "classification",
        "variance_gate_threshold": 0.01,
        "gate_margin": 0.001,
        "reproducibility": {"seed": 42},
    }
    (comp / "challenge_config.json").write_text(json.dumps(config), encoding="utf-8")
    state = {
        "competition": "phase3-cmp",
        "md5_target_hash": None,
        "anchor_oof_f1": 0.80,
        "anchor_oof_rmse": None,
        "anchor_lb_score": None,
        "submissions_used_today": 0,
        "submissions_used_total": 0,
        "remaining_submissions": 3,
        "dag_phase": "phase_3_features",
        "selected_submissions": [],
        "last_updated": None,
        "best_variant_this_round": "variant-a",
        "best_variant_oof_f1": 0.812,
        "variants_passed": 1,
        "metric_analysis": {"fold_score_variance": 0.001},
        "leaked_features": [],
        "shap_completed_at": "2026-05-25T00:00:00+00:00",
        "pruning_pass": True,
        "human_gate_2_variant-a_approved": True,
    }
    (comp / "SKILL_STATE.json").write_text(json.dumps(state), encoding="utf-8")
    return comp


def test_skill11_gate_promotes_passing_branch(tmp_path, monkeypatch):
    comp = _make_phase3_comp(tmp_path)
    monkeypatch.chdir(tmp_path)

    import zindian.skills.skill_11_gate as gate

    class DummyCompletedProcess:
        returncode = 0
        stdout = b""
        stderr = b""

    def fake_run(*args, **kwargs):
        return DummyCompletedProcess()

    monkeypatch.setattr(gate.subprocess, "run", fake_run)

    result = gate.run()

    assert result["status"] == "PASS"
    state = json.loads((comp / "SKILL_STATE.json").read_text(encoding="utf-8"))
    assert state["anchor_git_branch"] == "anchor-v2"
    assert state["phase_3_gate_diagnosis"]["passed"] is True


def test_skill11_gate_blocks_without_human_approval(tmp_path, monkeypatch):
    comp = _make_phase3_comp(tmp_path)
    state_path = comp / "SKILL_STATE.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.pop("human_gate_2_variant-a_approved")
    state_path.write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    import zindian.skills.skill_11_gate as gate

    result = gate.run()

    assert result["status"] == "BLOCKED"
    assert result["reason"] == "human gate missing"
    updated = json.loads(state_path.read_text(encoding="utf-8"))
    assert updated["dag_phase"] == "phase_3_gate_blocked"
    assert updated["phase_3_gate_diagnosis"]["failure_reason"] == "human_gate_missing"
