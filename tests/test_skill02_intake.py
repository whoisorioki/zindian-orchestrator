import json
from pathlib import Path

import pytest

from zindian.config import ConfigNotPopulated
from zindian.skills.skill_02_intake import extract_config, run


def test_extract_config_monitor_fallback_preserves_missing_external_flag(
    monkeypatch, tmp_path
):
    data: dict = {
        "name": "Test Comp",
        "metric": None,
        "sections": [],
    }

    monitor = {
        "competition_intel": {
            "metric": "auc",
            "use_probabilities": True,
        }
    }
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True)
    (reports_dir / "zindi_monitor.json").write_text(
        json.dumps(monitor), encoding="utf-8"
    )

    class SimplePaths:
        def __init__(self, root: Path):
            self.reports_dir = root / "reports"

    monkeypatch.setattr(
        "zindian.skills.skill_02_intake.resolve_competition_paths",
        lambda slug=None: SimplePaths(tmp_path),
    )

    cfg = extract_config(data, "test-slug")
    assert cfg["metric"] == "auc"
    assert cfg["use_probabilities"] is True
    assert cfg["allowed_external_data"] is None


def test_extract_config_minimal_auc():
    data: dict = {
        "name": "Test Comp",
        "metric": "auc",
        "metric_direction": None,
        "sections": [],
    }
    cfg = extract_config(data, "test-slug")
    assert cfg["slug"] == "test-slug"
    assert cfg["name"] == "Test Comp"
    assert cfg["metric"] == "auc"
    # AUC should imply use_probabilities True per intake heuristics
    assert cfg["use_probabilities"] is True


def test_run_raises_before_write_when_metric_missing(tmp_path, monkeypatch):
    slug = "cmp-missing-metric"
    comp = tmp_path / "competitions" / slug
    (comp / "data" / "raw").mkdir(parents=True)
    (comp / "data" / "processed").mkdir(parents=True)
    state_path = comp / "SKILL_STATE.json"
    state_path.write_text(
        json.dumps({"dag_phase": "phase_1", "last_updated": "2026-01-01T00:00:00Z"}),
        encoding="utf-8",
    )

    def fake_fetch(slug_arg, headers):
        return {
            "name": "Missing Metric",
            "sections": [],
            "metric": None,
            "metric_direction": None,
            "submission_format": {},
            "use_probabilities": None,
        }

    class SimplePaths:
        def __init__(self, root: Path):
            self.competition_dir = root / "competitions" / slug
            self.config_path = self.competition_dir / "challenge_config.json"
            self.state_path = state_path
            self.reports_dir = self.competition_dir / "reports"
            self.data_raw_dir = self.competition_dir / "data" / "raw"

    monkeypatch.setattr(
        "zindian.skills.skill_02_intake.resolve_competition_paths",
        lambda slug=None: SimplePaths(tmp_path),
    )
    monkeypatch.setattr("zindian.skills.skill_02_intake.fetch_competition", fake_fetch)

    called = {}

    def fake_write_config(*args, **kwargs):
        called["write_config"] = True

    monkeypatch.setattr(
        "zindian.skills.skill_02_intake.write_config", fake_write_config
    )

    with pytest.raises(ConfigNotPopulated):
        run(slug, headers={}, dry_run=False, merge=False)
    assert called.get("write_config") is None
