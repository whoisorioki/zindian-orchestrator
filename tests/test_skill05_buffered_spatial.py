"""Tests for skill_05 BufferedSpatialCV routing (spatial routing patch checklist).

Covers the Validation & Testing checklist from the Skill05 Spatial Routing Patch:
1. Config with BufferedSpatialCV in challenge_config.json -> routes to build_spatial_splits()
2. Config with dataset_config.spatial_coordinates -> _resolve_decision() returns BufferedSpatialCV
3. CLI --strategy=buffered_spatial -> forces spatial CV
4. Climate-risk-style config (lat/lon declared in spatial_signal) -> selects BufferedSpatialCV
5. Regression: non-spatial competitions still select KFold/StratifiedKFold
6. cv_split_source written correctly for audit trail
"""

from __future__ import annotations

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


def _setup(tmp_path, payload: dict, n_rows: int = 30) -> SimplePaths:
    """Create a competition dir with a spatial-capable features frame + config.

    The feature frame carries Latitude/Longitude columns so both the
    config bypass path and the data-driven _resolve_decision path can
    detect spatial coordinates.
    """
    paths = SimplePaths(tmp_path)
    paths.data_processed_dir.mkdir(parents=True)
    paths.data_raw_dir.mkdir(parents=True)
    paths.reports_dir.mkdir(parents=True)

    df = pd.DataFrame(
        {
            "ID": list(range(n_rows)),
            "target": [i % 2 for i in range(n_rows)],
            "Latitude": [0.1 + (i * 0.05) for i in range(n_rows)],
            "Longitude": [30.0 + (i * 0.05) for i in range(n_rows)],
            "feature_a": [float(i) for i in range(n_rows)],
        }
    )
    df.to_csv(paths.data_processed_dir / "features_train.csv", index=False)

    state = skill_state_skeleton()
    state.update(
        {
            "competition": "cmp",
            "dag_phase": "phase_1",
            "eda": {},
        }
    )
    _write_state(tmp_path, state)

    paths.config_path.write_text(json.dumps(payload), encoding="utf-8")
    return paths


def _patch_runtime(monkeypatch, paths, cfg_payload):
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


# -- Checklist item 1: config-declared BufferedSpatialCV routes to build_spatial_splits --


def test_config_buffered_spatial_cv_routes_to_spatial_splits(tmp_path, monkeypatch):
    cfg_payload = {
        "slug": "cmp",
        "task_type": "classification",
        "target_column": "target",
        "cv_strategy": {
            "type": "BufferedSpatialCV",
            "n_splits": 5,
            "lat_col": "Latitude",
            "lon_col": "Longitude",
            "spatial_buffer_km": 25.0,
            "selection_reason": "configured_in_challenge_config",
        },
        "reproducibility": {"seed": 7},
    }
    paths = _setup(tmp_path, cfg_payload)
    _patch_runtime(monkeypatch, paths, cfg_payload)

    result = cv_architect.run(strategy="compare")
    assert result["status"] == "OK"
    assert result["strategy_chosen"] == "BufferedSpatialCV"
    # spatial buffer applied — not the non-buffered GroupKFold fallback message
    assert "spatial_buffer_applied" in result["selection_reason"]

    written = json.loads(paths.config_path.read_text(encoding="utf-8"))
    assert written["cv_strategy"]["type"] == "BufferedSpatialCV"
    assert written["cv_strategy"]["lat_col"] == "Latitude"
    assert written["cv_strategy"]["lon_col"] == "Longitude"
    assert written["cv_strategy"]["spatial_buffer_km"] == 25.0

    # Audit trail: spatial splits materialized + source recorded
    # (SkillStateStore externalizes the split list to scores/cv_split_indices.json
    # and leaves a pointer dict with the split count.)
    state = json.loads(paths.state_path.read_text(encoding="utf-8"))
    assert state["cv_strategy"]["type"] == "BufferedSpatialCV"
    assert state["cv_split_source"] == "skill_05_spatial"
    assert state["cv_split_indices"]["count"] == 5
    splits_file = paths.competition_dir / "scores" / "cv_split_indices.json"
    materialized = json.loads(splits_file.read_text(encoding="utf-8"))
    assert len(materialized) == 5
    assert len(state.get("cv_split_groups", [])) == 30
    assert state["cv_split_groups"][0] in (0, 1, 2, 3, 4, 5, 6, 7)


# -- Checklist item 2: dataset_config.spatial_coordinates -> _resolve_decision returns BufferedSpatialCV --


def test_dataset_config_spatial_coordinates_detected(tmp_path, monkeypatch):
    cfg_payload = {
        "slug": "cmp",
        "task_type": "classification",
        "target_column": "target",
        "cv_strategy": {"n_splits": 5},  # no type -> data-driven decision path
        "dataset_config": {
            "spatial_coordinates": {"lat_col": "Latitude", "lon_col": "Longitude"}
        },
        "reproducibility": {"seed": 7},
    }
    paths = _setup(tmp_path, cfg_payload)

    ft = pd.read_csv(paths.data_processed_dir / "features_train.csv")
    decision = cv_architect._resolve_decision(_fake_cfg(cfg_payload), {}, ft)
    assert decision["type"] == "BufferedSpatialCV"
    assert decision["selection_reason"] == "spatial_coordinates_detected"
    assert decision["lat_col"] == "Latitude"
    assert decision["lon_col"] == "Longitude"
# -- Checklist item 3: CLI --strategy=buffered_spatial forces spatial CV --


def test_cli_buffered_spatial_forces_spatial_cv(tmp_path, monkeypatch):
    # Config declares a plain non-spatial strategy; the CLI flag must override it.
    cfg_payload = {
        "slug": "cmp",
        "task_type": "classification",
        "target_column": "target",
        "cv_strategy": {
            "type": "StratifiedKFold",
            "n_splits": 5,
            "selection_reason": "configured_in_challenge_config",
        },
        "spatial_signal": {
            "present": True,
            "lat_col": "Latitude",
            "lon_col": "Longitude",
            "spatial_buffer_km": 10.0,
        },
        "reproducibility": {"seed": 7},
    }
    paths = _setup(tmp_path, cfg_payload)
    _patch_runtime(monkeypatch, paths, cfg_payload)

    # CLI equivalent: pass --strategy=buffered_spatial
    result = cv_architect.run(strategy="buffered_spatial")
    assert result["status"] == "OK"
    assert result["strategy_chosen"] == "BufferedSpatialCV"
    assert "spatial_buffer_applied" in result["selection_reason"]

    state = json.loads(paths.state_path.read_text(encoding="utf-8"))
    assert state["cv_split_source"] == "skill_05_spatial"


def test_cli_buffered_spatial_forced_no_config_coords_uses_spatial_signal(
    tmp_path, monkeypatch
):
    # Decision resolved via config bypass (StratifiedKFold) has no lat/lon;
    # the forced path must fall back to spatial_signal / legacy columns.
    cfg_payload = {
        "slug": "cmp",
        "task_type": "classification",
        "target_column": "target",
        "cv_strategy": {
            "type": "StratifiedKFold",
            "n_splits": 5,
            "selection_reason": "configured_in_challenge_config",
        },
        "latitude_column": "Latitude",
        "longitude_column": "Longitude",
        "spatial_signal": {"present": True, "spatial_buffer_km": 5.0},
        "reproducibility": {"seed": 7},
    }
    paths = _setup(tmp_path, cfg_payload)
    _patch_runtime(monkeypatch, paths, cfg_payload)

    result = cv_architect.run(strategy="buffered_spatial")
    assert result["status"] == "OK"
    assert result["strategy_chosen"] == "BufferedSpatialCV"
    assert "spatial_buffer_applied" in result["selection_reason"]


# -- Checklist item 4: climate-risk-style config (spatial_signal lat/lon) selects BufferedSpatialCV --


def test_spatial_signal_lat_lon_selects_buffered_spatial(tmp_path, monkeypatch):
    cfg_payload = {
        "slug": "cmp",
        "task_type": "classification",
        "target_column": "target",
        "cv_strategy": {"n_splits": 5},
        "spatial_signal": {
            "present": True,
            "lat_col": "Latitude",
            "lon_col": "Longitude",
            "spatial_buffer_km": 15.0,
        },
        "reproducibility": {"seed": 7},
    }
    paths = _setup(tmp_path, cfg_payload)
    _patch_runtime(monkeypatch, paths, cfg_payload)

    result = cv_architect.run(strategy="compare")
    assert result["status"] == "OK"
    assert result["strategy_chosen"] == "BufferedSpatialCV"
    assert "spatial_buffer_applied" in result["selection_reason"]

    state = json.loads(paths.state_path.read_text(encoding="utf-8"))
    assert state["cv_split_source"] == "skill_05_spatial"


# -- Checklist item 5: non-spatial competitions still select KFold/StratifiedKFold --


def test_non_spatial_still_selects_plain_kfold(tmp_path, monkeypatch):
    cfg_payload = {
        "slug": "cmp",
        "task_type": "classification",
        "target_column": "target",
        "cv_strategy": {"n_splits": 5},
        "reproducibility": {"seed": 7},
    }
    paths = _setup(tmp_path, cfg_payload)
    _patch_runtime(monkeypatch, paths, cfg_payload)

    result = cv_architect.run(strategy="compare")
    assert result["status"] == "OK"
    assert result["strategy_chosen"] in ("KFold", "StratifiedKFold")

    written = json.loads(paths.config_path.read_text(encoding="utf-8"))
    assert written["cv_strategy"]["type"] in ("KFold", "StratifiedKFold")


def test_non_spatial_minority_imbalance_selects_stratified(tmp_path, monkeypatch):
    cfg_payload = {
        "slug": "cmp",
        "task_type": "classification",
        "target_column": "target",
        "minority_ratio": 0.1,
        "cv_strategy": {"n_splits": 5},
        "reproducibility": {"seed": 7},
    }
    paths = _setup(tmp_path, cfg_payload)
    _patch_runtime(monkeypatch, paths, cfg_payload)

    result = cv_architect.run(strategy="compare")
    assert result["status"] == "OK"
    assert result["strategy_chosen"] == "StratifiedKFold"

    state = json.loads(paths.state_path.read_text(encoding="utf-8"))
    assert state["cv_split_source"] == "skill_05_configured"