import json
from pathlib import Path

import pandas as pd

from zindian.schemas import skill_state_skeleton
from zindian.skills import skill_05_cv as cv_architect


class SimplePaths:
    def __init__(self, root: Path):
        self.competition_dir = root
        self.data_processed_dir = root / "data" / "processed"
        self.data_raw_dir = root / "data" / "raw"
        self.reports_dir = root / "reports"
        self.config_path = root / "challenge_config.json"
        self.state_path = root / "SKILL_STATE.json"


def _write_state(root: Path, state: dict) -> None:
    root.joinpath("SKILL_STATE.json").write_text(json.dumps(state), encoding="utf-8")


def _fake_cfg(payload: dict):
    class FakeCfg:
        def __init__(self) -> None:
            self._data = payload

        def get(self, key, default=None):
            return self._data.get(key, default)

        @property
        def slug(self):
            return self._data.get("slug", "cmp")

    return FakeCfg()


def test_skill05_deterministic_temporal_choice(tmp_path, monkeypatch):
    paths = SimplePaths(tmp_path)
    paths.data_processed_dir.mkdir(parents=True)
    paths.data_raw_dir.mkdir(parents=True)
    paths.reports_dir.mkdir(parents=True)

    frame = pd.DataFrame(
        {
            "ID": [1, 2, 3, 4],
            "target": [0, 1, 0, 1],
            "Latitude": [0.1, 0.2, 0.3, 0.4],
            "Longitude": [1.1, 1.2, 1.3, 1.4],
            "blocked_feature": [10, 11, 12, 13],
            "keep_feature": [100, 101, 102, 103],
        }
    )
    frame.to_csv(paths.data_processed_dir / "features_train.csv", index=False)

    state = skill_state_skeleton()
    state.update(
        {
            "competition": "cmp",
            "dag_phase": "phase_1",
            "eda": {"temporal_index_confirmed": True},
        }
    )
    _write_state(tmp_path, state)

    cfg_payload = {
        "slug": "cmp",
        "task_type": "classification",
        "target_column": "target",
        "policy_filters": ["blocked_feature"],
        "cv_strategy": {"n_splits": 4},
        "reproducibility": {"seed": 7},
        "temporal_signal": {"present": True},
        "spatial_signal": {"present": False},
        "group_signal": {"present": False},
    }
    monkeypatch.setattr(
        cv_architect,
        "resolve_competition_paths",
        lambda require_competition=True: paths,
    )
    monkeypatch.setattr(
        cv_architect,
        "ChallengeConfig",
        type("C", (), {"load": staticmethod(lambda: _fake_cfg(cfg_payload))}),
    )

    result = cv_architect.run(strategy="compare")
    assert result["status"] == "OK"
    assert result["strategy_chosen"] == "TimeSeriesSplit"
    assert result["selection_reason"] == "temporal_index_confirmed"

    written = json.loads(paths.config_path.read_text(encoding="utf-8"))
    assert written["cv_strategy"]["type"] == "TimeSeriesSplit"
    assert written["cv_strategy"]["selection_reason"] == "temporal_index_confirmed"


def test_skill05_spatial_choice_uses_group_col(tmp_path, monkeypatch):
    paths = SimplePaths(tmp_path)
    paths.data_processed_dir.mkdir(parents=True)
    paths.data_raw_dir.mkdir(parents=True)
    paths.reports_dir.mkdir(parents=True)

    frame = pd.DataFrame(
        {
            "ID": [1, 2, 3, 4, 5, 6],
            "target": [0, 1, 0, 1, 0, 1],
            "Latitude": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "Longitude": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
            "site_id": [10, 10, 11, 11, 12, 12],
            "blocked_feature": [20, 21, 22, 23, 24, 25],
            "keep_feature": [200, 201, 202, 203, 204, 205],
        }
    )
    frame.to_csv(paths.data_processed_dir / "features_train.csv", index=False)

    state = skill_state_skeleton()
    state.update(
        {
            "competition": "cmp",
            "dag_phase": "phase_1",
            "eda": {"group_structure_confirmed": True},
        }
    )
    _write_state(tmp_path, state)

    cfg_payload = {
        "slug": "cmp",
        "task_type": "classification",
        "target_column": "target",
        "policy_filters": ["blocked_feature"],
        "cv_strategy": {"n_splits": 3},
        "reproducibility": {"seed": 7},
        "spatial_signal": {"present": True, "group_col": "site_id"},
        "group_signal": {"present": False},
    }
    monkeypatch.setattr(
        cv_architect,
        "resolve_competition_paths",
        lambda require_competition=True: paths,
    )
    monkeypatch.setattr(
        cv_architect,
        "ChallengeConfig",
        type("C", (), {"load": staticmethod(lambda: _fake_cfg(cfg_payload))}),
    )

    result = cv_architect.run(strategy="compare")
    assert result["status"] == "OK"
    assert result["strategy_chosen"] == "GroupKFold"
    assert result["selection_reason"] == "group_structure_confirmed"

    written = json.loads(paths.config_path.read_text(encoding="utf-8"))
    assert written["cv_strategy"]["type"] == "GroupKFold"
    assert written["cv_strategy"]["group_col"] == "site_id"
    assert written["cv_strategy"]["selection_reason"] == "group_structure_confirmed"


def test_skill05_group_structure_confirmed_without_config_signal_does_not_select_group_cv(
    tmp_path, monkeypatch
):
    """When eda has group_structure_confirmed=True (e.g. from low-cardinality land_cover_class),
    but challenge_config has NO group_signal or spatial_signal, skill_05_cv should NOT select
    GroupKFold with an unresolvable group column, but fall back to standard CV (KFold/StratifiedKFold).
    """
    paths = SimplePaths(tmp_path)
    paths.data_processed_dir.mkdir(parents=True)
    paths.data_raw_dir.mkdir(parents=True)
    paths.reports_dir.mkdir(parents=True)

    frame = pd.DataFrame(
        {
            "ID": [1, 2, 3, 4, 5, 6],
            "target": [0, 1, 0, 1, 0, 1],
            "land_cover_class": ["forest", "forest", "urban", "urban", "crop", "crop"],
            "keep_feature": [200, 201, 202, 203, 204, 205],
        }
    )
    frame.to_csv(paths.data_processed_dir / "features_train.csv", index=False)

    state = skill_state_skeleton()
    state.update(
        {
            "competition": "cmp",
            "dag_phase": "phase_1",
            "eda": {"group_structure_confirmed": True},
        }
    )
    _write_state(tmp_path, state)

    cfg_payload = {
        "slug": "cmp",
        "task_type": "classification",
        "target_column": "target",
        "cv_strategy": {"n_splits": 3},
        "reproducibility": {"seed": 7},
        "spatial_signal": {"present": False},
        "group_signal": {"present": False},
    }
    monkeypatch.setattr(
        cv_architect,
        "resolve_competition_paths",
        lambda require_competition=True: paths,
    )
    monkeypatch.setattr(
        cv_architect,
        "ChallengeConfig",
        type("C", (), {"load": staticmethod(lambda: _fake_cfg(cfg_payload))}),
    )

    result = cv_architect.run(strategy="compare")
    assert result["status"] == "OK"
    assert result["strategy_chosen"] != "GroupKFold"
    assert result["strategy_chosen"] in ("KFold", "StratifiedKFold")

    written = json.loads(paths.config_path.read_text(encoding="utf-8"))
    assert written["cv_strategy"]["type"] != "GroupKFold"


def test_skill04_and_skill05_end_to_end_group_structure_gating(tmp_path, monkeypatch):
    """End-to-end integration test:
    1. Runs skill_04_eda against a 500-row dataset containing a land_cover_class feature
       with 5 unique categories (ratio = 5/500 = 1% < 5%).
    2. Confirms group_structure_confirmed=True is written to SKILL_STATE.json.
    3. Runs skill_05_cv against the resulting state (with no group_signal/spatial_signal set)
       and verifies it does NOT select GroupKFold.
    """
    from zindian.skills.skill_04_eda import run as eda_run
    import numpy as np

    paths = SimplePaths(tmp_path)
    paths.data_processed_dir.mkdir(parents=True)
    paths.data_raw_dir.mkdir(parents=True)
    paths.reports_dir.mkdir(parents=True)

    n = 500
    categories = ["forest", "urban", "cropland", "water", "wetland"]
    df = pd.DataFrame(
        {
            "ID": range(1, n + 1),
            "target": [0, 1] * (n // 2),
            "land_cover_class": [categories[i % 5] for i in range(n)],
            "numeric_feat": np.random.default_rng(42).standard_normal(n),
        }
    )
    df.to_csv(paths.data_raw_dir / "Train.csv", index=False)
    df.drop(columns=["target"]).to_csv(paths.data_raw_dir / "Test.csv", index=False)
    df.to_csv(paths.data_processed_dir / "features_train.csv", index=False)

    state = skill_state_skeleton()
    state.update({"competition": "cmp-e2e", "dag_phase": "phase_1"})
    _write_state(tmp_path, state)

    cfg_payload = {
        "slug": "cmp-e2e",
        "task_type": "classification",
        "target_column": "target",
        "target_config": {"targets": [{"name": "target", "type": "classification"}]},
        "cv_strategy": {"n_splits": 5},
        "reproducibility": {"seed": 42},
        "spatial_signal": {"present": False},
        "group_signal": {"present": False},
    }
    paths.config_path.write_text(json.dumps(cfg_payload), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "zindian.skills.skill_04_eda.resolve_competition_paths",
        lambda require_competition=True: paths,
    )
    monkeypatch.setattr(
        cv_architect,
        "resolve_competition_paths",
        lambda require_competition=True: paths,
    )
    monkeypatch.setattr(
        cv_architect,
        "ChallengeConfig",
        type("C", (), {"load": staticmethod(lambda: _fake_cfg(cfg_payload))}),
    )

    # 1. Run Skill 04 EDA
    eda_run()

    # 2. Confirm group_structure_confirmed=True in state
    written_state = json.loads(paths.state_path.read_text(encoding="utf-8"))
    assert written_state.get("eda", {}).get("group_structure_confirmed") is True

    # 3. Run Skill 05 CV Architect
    result = cv_architect.run(strategy="compare")
    assert result["status"] == "OK"
    assert result["strategy_chosen"] != "GroupKFold"
    assert result["strategy_chosen"] in ("KFold", "StratifiedKFold")
