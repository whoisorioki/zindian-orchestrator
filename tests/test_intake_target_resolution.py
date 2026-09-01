"""Tests for skill_02 target-col resolution from the data dictionary (RFC fix A)."""

import tempfile
from pathlib import Path

from zindian.skills.skill_02_intake import _resolve_actual_target_col


def test_resolves_target_from_dictionary_when_candidate_is_submission_column():
    """The SampleSubmission column (e.g. TargetF1) must not be used as training
    target when the data dictionary names the real target."""
    with tempfile.TemporaryDirectory() as tmp:
        dict_path = Path(tmp) / "data_dictionary.csv"
        dict_path.write_text(
            "column_name,description\n"
            "is_climate_sensitive,Binary target indicating whether the death is climate-sensitive\n"
            "age,Age of the individual at time of death\n"
        )
        train_cols = ["ID", "age", "is_climate_sensitive"]
        resolved = _resolve_actual_target_col("TargetF1", train_cols, dict_path)
        assert resolved == "is_climate_sensitive"


def test_keeps_target_when_candidate_is_train_column():
    """A candidate that already exists in the training columns is kept as-is."""
    with tempfile.TemporaryDirectory() as tmp:
        dict_path = Path(tmp) / "data_dictionary.csv"
        dict_path.write_text("column_name,description\n")
        resolved = _resolve_actual_target_col(
            "is_climate_sensitive", ["ID", "is_climate_sensitive"], dict_path
        )
        assert resolved == "is_climate_sensitive"


def test_keeps_candidate_when_dictionary_missing():
    """Without a data dictionary, the candidate falls back unchanged."""
    with tempfile.TemporaryDirectory() as tmp:
        resolved = _resolve_actual_target_col(
            "TargetF1", ["ID"], Path(tmp) / "missing.csv"
        )
        assert resolved == "TargetF1"