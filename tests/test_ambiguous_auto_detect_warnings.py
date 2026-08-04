"""
Unit tests for F4 (Ambiguous Auto-Detect Warnings).

Verifies that Skill 04 EDA emits a metadata_warning when feature column naming
contains ambiguous candidate band patterns (e.g., both prefix_month and month_prefix),
and verifies that detected_bands is correctly persisted into SKILL_STATE["eda"].
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from zindian.skills import skill_04_eda
from zindian.state import SkillStateStore


def test_ambiguous_band_detection_warning(tmp_path: Path, monkeypatch):
    """
    When feature columns contain mixed format candidate band names (e.g. VH_01 and 01_VV),
    EDA should write a descriptive warning to SKILL_STATE["metadata_warnings"].
    """
    comp_dir = tmp_path / "competitions" / "test-slug"
    comp_dir.mkdir(parents=True)
    raw_dir = comp_dir / "data" / "raw"
    raw_dir.mkdir(parents=True)

    config_data = {
        "slug": "test-slug",
        "target_col": "target",
        "input_files": {"train": "Train.csv"},
    }
    initial_state: dict[str, object] = {
        "competition": "test-slug",
        "md5_target_hash": "dummy",
        "anchor_oof_score": None,
        "anchor_lb_score": None,
        "submissions_used_today": 0,
        "submissions_used_total": 0,
        "remaining_submissions": 10,
        "dag_phase": "phase_0_foundation",
        "selected_submissions": [],
        "last_updated": "2026-08-03T00:00:00Z",
    }
    (comp_dir / "challenge_config.json").write_text(json.dumps(config_data))
    (comp_dir / "SKILL_STATE.json").write_text(json.dumps(initial_state))

    # Create dummy DataFrame with mixed band formats
    df = pd.DataFrame(
        {
            "ID": [1, 2, 3, 4, 5],
            "target": [0, 1, 0, 1, 0],
            "VH_01": [1.0, 2.0, 3.0, 4.0, 5.0],
            "VH_02": [1.1, 2.1, 3.1, 4.1, 5.1],
            "01_VV": [0.5, 0.6, 0.7, 0.8, 0.9],
        }
    )
    df.to_csv(raw_dir / "Train.csv", index=False)

    monkeypatch.setenv("ZINDIAN_COMPETITION_SLUG", "test-slug")
    monkeypatch.setattr(
        "zindian.paths.resolve_competition_paths",
        lambda require_competition=True: type(
            "Paths",
            (),
            {
                "competition_dir": comp_dir,
                "config_path": comp_dir / "challenge_config.json",
                "state_path": comp_dir / "SKILL_STATE.json",
                "data_raw_dir": raw_dir,
                "reports_dir": comp_dir / "reports",
            },
        )(),
    )

    skill_04_eda.run()

    store = SkillStateStore(comp_dir / "SKILL_STATE.json")
    state = store.read()

    # Verify detected_bands in eda
    assert "detected_bands" in state.get("eda", {})
    assert "VH" in state["eda"]["detected_bands"]

    # Verify metadata_warnings
    warnings = state.get("metadata_warnings", [])
    assert any(
        "Ambiguous band detection" in w for w in warnings
    ), f"Expected ambiguous band warning in metadata_warnings, got: {warnings}"
