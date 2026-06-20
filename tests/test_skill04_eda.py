import json

import pandas as pd
import pytest

from zindian.schemas import skill_state_skeleton
from zindian.state import SkillStateStore
from zindian.skills.skill_04_eda import (
    _build_categorical_columns,
    _high_correlation_pairs,
    _outlier_summary,
    run,
)


def test_high_correlation_pairs_use_named_labels():
    frame = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4, 5],
            "feature_b": [2, 4, 6, 8, 10],
            "feature_c": [5, 4, 3, 2, 1],
        }
    )

    corr = frame.corr().abs()
    pairs = _high_correlation_pairs(corr, thresh=0.95)

    assert ("feature_a", "feature_b", pytest.approx(1.0)) in pairs
    assert any(left == "feature_a" and right == "feature_c" for left, right, _ in pairs)


def test_outlier_summary_prefers_robust_branch_for_skewed_data():
    series = pd.Series([1] * 35 + [1000] * 6)

    summary = _outlier_summary(series, total_rows=len(series))

    assert summary["flag"] is True
    assert summary["method"] in {"mad", "quantile_fence", "median_deviation"}
    assert summary["skewness"] >= 0.0


def test_categorical_columns_follow_config_rules_without_cardinality_ceiling():
    frame = pd.DataFrame(
        {
            "encoded_int": list(range(21)),
            "object_col": ["x"] * 21,
            "numeric_col": list(range(21)),
        }
    )
    rules = {"encoded_int": "ordinal"}

    categorical = _build_categorical_columns(
        frame, ["encoded_int", "object_col", "numeric_col"], rules
    )

    assert {item["name"] for item in categorical} == {"encoded_int", "object_col"}
    assert (
        next(item for item in categorical if item["name"] == "encoded_int")["encoding"]
        == "ordinal"
    )
    assert (
        next(item for item in categorical if item["name"] == "object_col")["encoding"]
        == "one-hot or ordinal"
    )


def test_run_raises_before_guessing_target(tmp_path, monkeypatch):
    slug = "cmp-eda"
    competition_dir = tmp_path / "competitions" / slug
    (competition_dir / "data" / "raw").mkdir(parents=True)
    (competition_dir / "data" / "processed").mkdir(parents=True)
    (competition_dir / "reports").mkdir(parents=True)

    frame = pd.DataFrame(
        {
            "feature_one": [0, 1, 0, 1],
            "feature_two": [1, 0, 1, 0],
            "candidate_label": [0, 1, 0, 1],
        }
    )
    frame.to_csv(competition_dir / "data" / "raw" / "Training_Data.csv", index=False)

    state_path = competition_dir / "SKILL_STATE.json"
    state = skill_state_skeleton()
    state["dag_phase"] = "phase_1_complete"
    state["last_updated"] = "2026-01-01T00:00:00Z"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="Unable to resolve target column"):
        run()

    assert not (competition_dir / "reports" / "eda_report.json").exists()


def test_run_surfaces_state_write_failures(tmp_path, monkeypatch, capsys):
    slug = "cmp-eda-state"
    competition_dir = tmp_path / "competitions" / slug
    (competition_dir / "data" / "raw").mkdir(parents=True)
    (competition_dir / "data" / "processed").mkdir(parents=True)
    (competition_dir / "reports").mkdir(parents=True)

    frame = pd.DataFrame(
        {
            "target": [0, 1, 0, 1],
            "feature_one": [1, 2, 3, 4],
            "feature_two": [4, 3, 2, 1],
        }
    )
    frame.to_csv(competition_dir / "data" / "raw" / "Training_Data.csv", index=False)

    (competition_dir / "challenge_config.json").write_text(
        json.dumps({"target": "target"}),
        encoding="utf-8",
    )
    state_path = competition_dir / "SKILL_STATE.json"
    state = skill_state_skeleton()
    state["dag_phase"] = "phase_1_complete"
    state["last_updated"] = "2026-01-01T00:00:00Z"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    def failing_update(self, **patch):
        raise RuntimeError("state write failed")

    monkeypatch.setattr(SkillStateStore, "update", failing_update)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RuntimeError, match="state write failed"):
        run()

    captured = capsys.readouterr()
    assert "ERROR: failed to update SKILL_STATE.json after EDA" in captured.out
