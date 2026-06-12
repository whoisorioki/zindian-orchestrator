import pytest
from unittest.mock import MagicMock
from zindian.skills.skill_21_pseudo_label import run as run_skill_21


class DummyConfig:
    def __init__(self, data):
        self.data = data

    def get(self, key, default=None):
        return self.data.get(key, default)


def test_pseudo_label_blocks_regression(monkeypatch):
    # Configure mock challenge config to return regression
    mock_cfg = DummyConfig({"task_type": "regression"})

    # Mock ChallengeConfig.load to return our dummy config
    monkeypatch.setattr(
        "zindian.skills.skill_21_pseudo_label.ChallengeConfig.load", lambda: mock_cfg
    )

    # Mock other requirements of run() like paths and store so it doesn't fail on them
    mock_paths = MagicMock()
    mock_paths.state_path = "dummy_state_path.json"
    monkeypatch.setattr(
        "zindian.skills.skill_21_pseudo_label.resolve_competition_paths",
        lambda: mock_paths,
    )

    mock_store = MagicMock()
    mock_store.read.return_value = {}
    monkeypatch.setattr(
        "zindian.skills.skill_21_pseudo_label.SkillStateStore", lambda path: mock_store
    )

    # Assert that ValueError is raised preserving classification-only guard
    with pytest.raises(ValueError, match="strictly prohibited for task_type"):
        run_skill_21()
