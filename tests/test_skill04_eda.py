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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HEAVY_KEYS = {
    "band_summary_stats",
    "seasonal_amplitude",
    "temporal_trends",
    "target_correlation_per_feature",
    "class_separability_index",
}


def _make_eda_competition(tmp_path, slug, df, target_col="label"):
    """Create a minimal competition dir and return the state path."""
    competition_dir = tmp_path / "competitions" / slug
    (competition_dir / "data" / "raw").mkdir(parents=True)
    (competition_dir / "data" / "processed").mkdir(parents=True)
    (competition_dir / "reports").mkdir(parents=True)

    df.to_csv(competition_dir / "data" / "raw" / "Training_Data.csv", index=False)

    (competition_dir / "challenge_config.json").write_text(
        json.dumps({"target": target_col, "reproducibility": {"seed": 42}}),
        encoding="utf-8",
    )
    state_path = competition_dir / "SKILL_STATE.json"
    state = skill_state_skeleton()
    state["dag_phase"] = "phase_1_eda"
    state["last_updated"] = "2026-01-01T00:00:00Z"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    return state_path


# ---------------------------------------------------------------------------
# State-vs-reports boundary tests (core of Part 0 / Part 1 requirement)
# ---------------------------------------------------------------------------


def test_heavy_diagnostics_in_report_not_in_state(tmp_path, monkeypatch):
    """The 5 heavy band/correlation dicts MUST appear in eda_report.json
    and MUST NOT appear in SKILL_STATE.json['eda'].

    This test enforces the A6-B state-vs-reports design boundary.
    """
    import numpy as np

    rng = np.random.default_rng(1)
    n = 60
    df = pd.DataFrame(
        {
            "label": rng.integers(0, 2, n),
            "feat_a": rng.standard_normal(n),
            "feat_b": rng.standard_normal(n),
        }
    )

    slug = "cmp-boundary"
    state_path = _make_eda_competition(tmp_path, slug, df)
    monkeypatch.chdir(tmp_path)
    run()

    competition_dir = tmp_path / "competitions" / slug

    # The 5 heavy keys must be in eda_report.json
    report = json.loads((competition_dir / "reports" / "eda_report.json").read_text())
    for key in _HEAVY_KEYS:
        assert (
            key in report
        ), f"Expected heavy key '{key}' in eda_report.json but it was absent"

    # The 5 heavy keys must NOT be in SKILL_STATE.json['eda']
    written_state = json.loads(state_path.read_text(encoding="utf-8"))
    eda_state = written_state.get("eda", {})
    for key in _HEAVY_KEYS:
        assert key not in eda_state, (
            f"Heavy key '{key}' found in SKILL_STATE.json['eda'] — it should be "
            f"in reports/eda_report.json only (A6-B violation)"
        )


def test_lean_fields_present_in_state_with_correct_types(tmp_path, monkeypatch):
    """All SoT-specified lean eda fields must be in SKILL_STATE.json['eda']
    with the correct types after a successful run."""
    import numpy as np

    rng = np.random.default_rng(17)
    n = 60
    df = pd.DataFrame(
        {
            "label": rng.integers(0, 2, n),
            "feat_x": rng.standard_normal(n),
            "feat_y": rng.standard_normal(n),
        }
    )

    slug = "cmp-lean-fields"
    state_path = _make_eda_competition(tmp_path, slug, df)
    monkeypatch.chdir(tmp_path)
    run()

    written = json.loads(state_path.read_text(encoding="utf-8"))
    eda = written.get("eda", {})

    lean_fields = [
        "mnar_columns",
        "mcar_columns",
        "outlier_columns",
        "target_skew",
        "temporal_index_confirmed",
        "group_structure_confirmed",
    ]
    for field in lean_fields:
        assert field in eda, f"Missing lean eda field: '{field}'"

    assert isinstance(eda["temporal_index_confirmed"], bool)
    assert isinstance(eda["group_structure_confirmed"], bool)
    assert isinstance(eda["target_skew"], float)
    assert isinstance(eda["mnar_columns"], list)
    assert isinstance(eda["mcar_columns"], list)
    assert isinstance(eda["outlier_columns"], list)


# ---------------------------------------------------------------------------
# temporal_index_confirmed detection tests
# ---------------------------------------------------------------------------


def test_temporal_index_confirmed_via_band_mm_pattern(tmp_path, monkeypatch):
    """A dataset with BAND_MM column pattern (e.g. sig_01..sig_12) should produce
    temporal_index_confirmed=True via the seasonal_amplitude / temporal_trends path.
    The sig_NN detection goes via BAND_MM → non-empty seasonal_amplitude → True."""
    n = 60
    data = {"label": [0, 1] * (n // 2)}
    for month in range(1, 13):
        data[f"sig_{month:02d}"] = list(range(n))
    df = pd.DataFrame(data)

    slug = "cmp-temporal-band"
    state_path = _make_eda_competition(tmp_path, slug, df)
    monkeypatch.chdir(tmp_path)
    run()

    written = json.loads(state_path.read_text(encoding="utf-8"))
    eda = written.get("eda", {})
    assert (
        eda.get("temporal_index_confirmed") is True
    ), "Expected temporal_index_confirmed=True for dataset with BAND_MM columns"


def test_temporal_index_confirmed_via_datetime_column(tmp_path, monkeypatch):
    """A dataset with a datetime-dtype column should produce
    temporal_index_confirmed=True via the is_datetime64_any_dtype path."""
    import numpy as np

    n = 60
    df = pd.DataFrame(
        {
            "label": [0, 1] * (n // 2),
            "numeric_feat": np.random.default_rng(42).integers(0, 100, n),
            # An actual datetime64 column — detected by dtype, no string parsing needed
            "event_date": pd.date_range("2023-01-01", periods=n, freq="D"),
        }
    )

    slug = "cmp-temporal-datetime"
    state_path = _make_eda_competition(tmp_path, slug, df)
    monkeypatch.chdir(tmp_path)
    run()

    written = json.loads(state_path.read_text(encoding="utf-8"))
    eda = written.get("eda", {})
    assert (
        eda.get("temporal_index_confirmed") is True
    ), "Expected temporal_index_confirmed=True for dataset with datetime64 column"


def test_temporal_index_confirmed_false_for_plain_numeric_dataset(
    tmp_path, monkeypatch
):
    """A plain numeric dataset with no temporal signals should produce
    temporal_index_confirmed=False."""
    import numpy as np

    rng = np.random.default_rng(7)
    n = 80
    df = pd.DataFrame(
        {
            "label": rng.integers(0, 2, n),
            "feat_a": rng.standard_normal(n),
            "feat_b": rng.standard_normal(n),
        }
    )

    slug = "cmp-no-temporal"
    state_path = _make_eda_competition(tmp_path, slug, df)
    monkeypatch.chdir(tmp_path)
    run()

    written = json.loads(state_path.read_text(encoding="utf-8"))
    eda = written.get("eda", {})
    assert (
        eda.get("temporal_index_confirmed") is False
    ), "Expected temporal_index_confirmed=False for plain numeric dataset"


# ---------------------------------------------------------------------------
# group_structure_confirmed detection tests
# ---------------------------------------------------------------------------


def test_group_structure_confirmed_via_low_cardinality_column(tmp_path, monkeypatch):
    """A dataset with a low-cardinality repeating column should produce
    group_structure_confirmed=True.
    (5 group labels across 200 rows → cardinality ratio = 2.5% < 5% threshold)
    """
    import numpy as np

    rng = np.random.default_rng(13)
    n = 200
    df = pd.DataFrame(
        {
            "label": rng.integers(0, 2, n),
            "numeric_feat": rng.standard_normal(n),
            # 5 distinct groups — cardinality ratio = 0.025 < 0.05 threshold
            "region": rng.integers(1, 6, n),
        }
    )

    slug = "cmp-group-structure"
    state_path = _make_eda_competition(tmp_path, slug, df)
    monkeypatch.chdir(tmp_path)
    run()

    written = json.loads(state_path.read_text(encoding="utf-8"))
    eda = written.get("eda", {})
    assert (
        eda.get("group_structure_confirmed") is True
    ), "Expected group_structure_confirmed=True for dataset with 5 groups over 200 rows"


def test_group_structure_confirmed_false_for_high_cardinality_dataset(
    tmp_path, monkeypatch
):
    """A dataset where all feature columns are either unique-per-row or constant
    should produce group_structure_confirmed=False."""
    import numpy as np

    rng = np.random.default_rng(99)
    n = 100
    df = pd.DataFrame(
        {
            "label": rng.integers(0, 2, n),
            # All distinct values — unique per row, skipped as ID-like
            "unique_id_col": list(range(n)),
            # Continuous numeric — nunique/n likely > 0.05
            "continuous_feat": rng.standard_normal(n),
        }
    )

    slug = "cmp-no-group"
    state_path = _make_eda_competition(tmp_path, slug, df)
    monkeypatch.chdir(tmp_path)
    run()

    written = json.loads(state_path.read_text(encoding="utf-8"))
    eda = written.get("eda", {})
    assert (
        eda.get("group_structure_confirmed") is False
    ), "Expected group_structure_confirmed=False for high-cardinality dataset"


def test_date_like_string_id_column_not_temporal_index(tmp_path, monkeypatch):
    """A date-like string ID column (e.g. sample_id with '20200115_000' style values)
    should NOT trigger temporal_index_confirmed=True because it is an ID column."""
    import numpy as np

    n = 60
    df = pd.DataFrame(
        {
            "label": [0, 1] * (n // 2),
            "numeric_feat": np.random.default_rng(42).integers(0, 100, n),
            "sample_id": [f"{2020 + i % 5}0{1 + i % 9}15_{i:03d}" for i in range(n)],
        }
    )

    slug = "cmp-date-like-id"
    state_path = _make_eda_competition(tmp_path, slug, df)
    monkeypatch.chdir(tmp_path)
    run()

    written = json.loads(state_path.read_text(encoding="utf-8"))
    eda = written.get("eda", {})
    assert (
        eda.get("temporal_index_confirmed") is False
    ), "Date-like string ID column (sample_id) should NOT trigger temporal_index_confirmed"


def test_low_cardinality_categorical_feature_reports_group_structure_confirmed(
    tmp_path, monkeypatch
):
    """A low-cardinality categorical feature (e.g. land_cover_class with 4-5 repeating values)
    should set group_structure_confirmed=True in eda state as an informational report flag.
    """
    import numpy as np

    n = 100
    categories = ["forest", "urban", "cropland", "water"]
    df = pd.DataFrame(
        {
            "label": [0, 1] * (n // 2),
            "numeric_feat": np.random.default_rng(7).standard_normal(n),
            "land_cover_class": [categories[i % 4] for i in range(n)],
        }
    )

    slug = "cmp-land-cover"
    state_path = _make_eda_competition(tmp_path, slug, df)
    monkeypatch.chdir(tmp_path)
    run()

    written = json.loads(state_path.read_text(encoding="utf-8"))
    eda = written.get("eda", {})
    assert (
        eda.get("group_structure_confirmed") is True
    ), "Low-cardinality categorical feature (land_cover_class) should set group_structure_confirmed=True"


def test_group_aware_mae_naive_baseline(tmp_path, monkeypatch):
    """Verify that when a group column is configured, consecutive differences
    are calculated within each group separately, avoiding cross-group boundary jumps."""

    # Create two groups:
    # Group A: target: 10, 11, 12. Diffs: [1, 1]
    # Group B: target: 100, 101, 102. Diffs: [1, 1]
    # If flat: diffs contain |100 - 12| = 88. Flat mean diffs would be (1+1+88+1+1)/5 = 18.4
    # If group-aware: diffs are [1, 1] for group A and [1, 1] for group B. Mean diffs: 1.0
    df = pd.DataFrame(
        {
            "label": [10, 11, 12, 100, 101, 102],
            "group_id": ["A", "A", "A", "B", "B", "B"],
            "event_date": pd.date_range("2023-01-01", periods=6, freq="D"),
        }
    )

    slug = "cmp-group-mae"
    competition_dir = tmp_path / "competitions" / slug
    (competition_dir / "data" / "raw").mkdir(parents=True)
    (competition_dir / "data" / "processed").mkdir(parents=True)
    (competition_dir / "reports").mkdir(parents=True)

    df.to_csv(competition_dir / "data" / "raw" / "Training_Data.csv", index=False)

    # Configure group_signal with group_id col
    config_data = {
        "target": "label",
        "task_type": "regression",
        "reproducibility": {"seed": 42},
        "group_signal": {"present": True, "col": "group_id"},
    }
    (competition_dir / "challenge_config.json").write_text(
        json.dumps(config_data),
        encoding="utf-8",
    )
    state_path = competition_dir / "SKILL_STATE.json"
    state = skill_state_skeleton()
    state["dag_phase"] = "phase_1_eda"
    state["last_updated"] = "2026-01-01T00:00:00Z"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    run()

    written = json.loads(state_path.read_text(encoding="utf-8"))
    eda = written.get("eda", {})

    # Assert that mae_naive is indeed 1.0 (group-aware) rather than 18.4 (flat)
    assert pytest.approx(eda.get("MAE_naive_baseline")) == 1.0
