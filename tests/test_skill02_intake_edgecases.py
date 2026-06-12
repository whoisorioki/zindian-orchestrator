import importlib

skill_02_intake = importlib.import_module("zindian.skills.skill_02_intake")

extract_config = skill_02_intake.extract_config


def test_extract_config_missing_fields_preserves_none(monkeypatch, tmp_path):
    data = {
        "name": "example",
        # metric intentionally omitted
    }

    # Monkeypatch resolve_competition_paths to a tmp reports dir to avoid repo monitor fallback
    class DummyPaths:
        def __init__(self, reports_dir):
            self.reports_dir = reports_dir

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    monkeypatch.setattr(
        skill_02_intake,
        "resolve_competition_paths",
        lambda slug=None: DummyPaths(reports_dir),
    )

    cfg = extract_config(data, slug="example")
    # When metric not present in API and no monitor fallback, use_probabilities should remain None
    assert cfg.get("metric") is None
    assert cfg.get("use_probabilities") is None
    # allowed_external_data should be absent or None, not flipped to boolean
    assert "allowed_external_data" in cfg
    assert cfg.get("allowed_external_data") is None
