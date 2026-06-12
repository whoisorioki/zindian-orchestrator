import json
from pathlib import Path

import pandas as pd

from zindian.schemas import skill_state_skeleton
from zindian.skills import skill_01_integrity as integrity


class SimplePaths:
    def __init__(self, root: Path):
        self.data_raw_dir = root / "data" / "raw"
        self.state_path = root / "SKILL_STATE.json"


def test_skill01_uses_configured_target_column(tmp_path, monkeypatch):
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)

    train = pd.DataFrame(
        {
            "ID": [1, 2],
            "custom_target": [0, 1],
            "Latitude": [10.0, 20.0],
            "Longitude": [30.0, 40.0],
        }
    )
    test = pd.DataFrame({"ID": [3], "Latitude": [50.0], "Longitude": [60.0]})
    sample = pd.DataFrame({"ID": [3], "Prediction": [0]})

    train.to_csv(raw / "Training_Data.csv", index=False)
    test.to_csv(raw / "Test.csv", index=False)
    sample.to_csv(raw / "SampleSubmission.csv", index=False)

    state_path = tmp_path / "SKILL_STATE.json"
    state_path.write_text(json.dumps(skill_state_skeleton()), encoding="utf-8")

    monkeypatch.setattr(
        integrity, "resolve_competition_paths", lambda: SimplePaths(tmp_path)
    )

    class FakeCfg:
        def __init__(self):
            self._data = {
                "target_column": "custom_target",
                "submission_target_column": "Prediction",
            }

        def get(self, key, default=None):
            return self._data.get(key, default)

    monkeypatch.setattr(
        integrity,
        "ChallengeConfig",
        type("C", (), {"load": staticmethod(lambda: FakeCfg())}),
    )

    result = integrity.run(re_verify=False)
    assert result["target_col"] == "custom_target"
    assert result["submission_target_col"] == "Prediction"
    assert result["class_distribution"] == {0: 1, 1: 1}
