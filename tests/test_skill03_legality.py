import json
from pathlib import Path

from zindian.schemas import skill_state_skeleton
from zindian.skills.skill_03_legality import (
    check_planned_features,
    run,
    synthesise_feature_policy,
)


class FakeConfig:
    def __init__(self, data: dict, slug: str = "test-slug"):
        self._data = data
        self.slug = slug


def _write_state(
    comp_dir: Path, *, phase: str, anchor_features=None, planned_features=None
) -> None:
    state = skill_state_skeleton()
    state["dag_phase"] = phase
    state["last_updated"] = "2026-01-01T00:00:00Z"
    if anchor_features is not None:
        state["anchor_features"] = anchor_features
    if planned_features is not None:
        state["planned_features"] = planned_features
    (comp_dir / "SKILL_STATE.json").write_text(json.dumps(state), encoding="utf-8")


def test_check_planned_features_requires_exact_ban_match():
    policy = {
        "allowed_data_sources": ["competition_provided_only"],
        "banned_transformations": ["dem"],
        "external_data_permitted": True,
        "coordinate_features_permitted": True,
    }

    checks = check_planned_features(
        policy,
        [
            {
                "name": "ok_feature",
                "transforms": ["demographic_encoding"],
                "uses_lat_lon": False,
            }
        ],
    )

    assert checks[0]["status"] == "PASS"
    assert checks[0]["blocks"] is False


def test_synthesise_feature_policy_collects_nested_monitor_bans():
    monitor_data = {
        "competition_intel": {
            "banned_features": ["from_monitor"],
            "allowed_data_sources": ["competition_provided_only"],
        }
    }
    config = {
        "banned_features": ["from_config"],
        "use_probabilities": True,
        "metric": "auc",
    }

    policy = synthesise_feature_policy(monitor_data, config, ["flagged title"])

    assert policy["banned_transformations"] == ["from_monitor", "from_config"]
    assert policy["source_flags"] == ["flagged title"]


def test_run_normalizes_anchor_features_and_advances_from_phase_one_integrity(
    tmp_path, monkeypatch
):
    slug = "cmp-legality"
    comp = tmp_path / "competitions" / slug
    (comp / "reports").mkdir(parents=True)
    _write_state(
        comp,
        phase="phase_1_integrity",
        anchor_features=[
            {"name": "demographic_encoding", "transforms": [], "uses_lat_lon": False},
            ["region_bucket", {"feature": "age_band", "transforms": []}],
        ],
    )
    monitor = {
        "competition_intel": {
            "allowed_data_sources": ["competition_provided_only"],
            "banned_features": [],
            "external_banned": True,
            "automl_banned": False,
            "use_probabilities": True,
            "metric": "auc",
        },
        "compliance": {"flagged_titles": []},
    }
    (comp / "reports" / "zindi_monitor.json").write_text(
        json.dumps(monitor), encoding="utf-8"
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "zindian.skills.skill_03_legality.ChallengeConfig.load",
        lambda: FakeConfig(
            {"use_probabilities": True, "metric": "auc", "banned_features": []},
            slug=slug,
        ),
    )

    out = run(slug, planned_features=None)

    assert out["status"] == "GO"
    state = json.loads((comp / "SKILL_STATE.json").read_text(encoding="utf-8"))
    assert state["dag_phase"] == "phase_2_legality_checked"
    assert state["legality_status"] == "GO"
    report = (comp / "reports" / "legality_report.md").read_text(encoding="utf-8")
    assert "region_bucket" in report
    assert "age_band" in report
