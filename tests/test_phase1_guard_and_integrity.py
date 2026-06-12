import json


from zindian.skills import skill_02_intake as intake
from zindian.skills import skill_01_integrity as integrity
from zindian.paths import resolve_competition_paths
from zindian.schemas import skill_state_skeleton


def make_competition_dir(tmp_path, slug: str, state_phase: str | None):
    comp = tmp_path / "competitions" / slug
    comp.mkdir(parents=True)
    state_path = comp / "SKILL_STATE.json"
    state = skill_state_skeleton()
    state["dag_phase"] = state_phase
    state["last_updated"] = "2026-01-01T00:00:00Z"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    return comp


def test_intake_skips_write_when_phase_prohibits(tmp_path, monkeypatch):
    slug = "cmp-skip"
    make_competition_dir(tmp_path, slug, "phase_2_legality_checked")

    # Monkeypatch cwd to tmp_path so resolve_competition_paths finds our comp
    monkeypatch.chdir(tmp_path)

    # Replace network call with minimal payload
    def fake_fetch(slug_arg, headers):
        return {"name": "X", "metric": "auc", "sections": []}

    monkeypatch.setattr(intake, "fetch_competition", fake_fetch)

    # Run intake; should skip writing challenge_config.json due to phase
    intake.run(slug, headers={}, dry_run=False, merge=False)
    cfg_path = resolve_competition_paths(slug=slug).config_path
    assert not cfg_path.exists()


def test_intake_merge_preserves_existing_nonnull(tmp_path, monkeypatch):
    slug = "cmp-merge"
    comp = make_competition_dir(tmp_path, slug, None)
    # create existing challenge_config with a non-null 'name'
    cfg_path = comp / "challenge_config.json"
    existing = {"name": "Existing", "metric": None}
    cfg_path.write_text(json.dumps(existing), encoding="utf-8")

    monkeypatch.chdir(tmp_path)

    def fake_fetch(slug_arg, headers):
        return {"name": "NewName", "metric": "auc", "sections": []}

    monkeypatch.setattr(intake, "fetch_competition", fake_fetch)

    intake.run(slug, headers={}, dry_run=False, merge=True)
    final = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert final["name"] == "Existing"
    assert final["metric"] == "auc"


def test_integrity_update_writes_state(tmp_path):
    comp = tmp_path / "competitions" / "cmp-int"
    comp.mkdir(parents=True)
    state_path = comp / "SKILL_STATE.json"
    from zindian.schemas import skill_state_skeleton

    state_path.write_text(json.dumps(skill_state_skeleton()), encoding="utf-8")

    sample_integrity = {
        "md5_target_hash": "aaa",
        "md5_train_file": "bbb",
        "md5_test_file": "ccc",
        "md5_sample_sub_file": "ddd",
    }

    integrity.update_skill_state(sample_integrity, state_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["md5_target_hash"] == "aaa"
    assert state.get("dag_phase") == "phase_1_complete"
