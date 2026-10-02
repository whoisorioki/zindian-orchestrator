"""Regression tests for order-safe as-of joins.

The defect these guard against is silent: labels detach from their features
while row counts, column counts and file-level ID checks all still pass.

The trigger requires the left frame to be *not already* in date order, so
that `sort_values` produces a non-identity permutation of the left index,
and `merge_asof` then returns an index that no longer corresponds to the
left rows. A positional `.loc[RangeIndex]` restore silently misaligns the
tie neighbourhoods. That exact pattern scrambled 387 of 3146 rows in a
real competition run.
"""

from __future__ import annotations

import pandas as pd
import pytest

from zindian.joins import sorted_asof_join


# Deliberately NOT in ascending date order, and containing a date tie.
# This is what makes the old positional-restore pattern fail.
LEFT = pd.DataFrame(
    {
        "ID": ["r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7"],
        "when": pd.to_datetime(
            [
                "2020-01-05",
                "2020-01-01",
                "2020-01-09",
                "2020-01-05",  # tie with r0
                "2020-01-03",
                "2020-01-11",
                "2020-01-07",
                "2020-01-13",
            ]
        ),
        "lat": [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
        "lon": [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
        "label": [50, 10, 90, 40, 30, 110, 70, 130],
    }
)

RIGHT = pd.DataFrame(
    {
        "when": pd.to_datetime(
            ["2020-01-01", "2020-01-04", "2020-01-06", "2020-01-10"]
        ),
        "lat": [1.0, 1.0, 2.0, 2.0],
        "lon": [1.0, 1.0, 2.0, 2.0],
        "ext": [10.0, 40.0, 60.0, 100.0],
    }
)

# id -> most recent reference row at or before that left row's date.
# group (1,1) has refs 01-01(10.0) and 01-04(40.0);
# group (2,2) has refs 01-06(60.0) and 01-10(100.0).
EXPECTED_EXT = {
    "r0": 40.0,    # 01-05, group 1
    "r1": 10.0,    # 01-01, group 1
    "r2": 40.0,    # 01-09, group 1 (01-10 ref belongs to group 2)
    "r3": 40.0,    # 01-05, group 1 -- ties with r0
    "r4": 10.0,    # 01-03, group 1
    "r5": 100.0,   # 01-11, group 2
    "r6": 60.0,    # 01-07, group 2
    "r7": 100.0,   # 01-13, group 2
}


def _old_buggy_pattern(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the historical plugin implementation exactly."""
    ls = left.sort_values(by="when").copy()
    rs = right.sort_values(by="when").copy()
    merged = pd.merge_asof(
        ls, rs, on="when", by=["lat", "lon"], direction="backward"
    )
    return merged.loc[left.index]
def test_old_positional_restore_pattern_scrambles_rows():
    """Guard the fixture itself.

    If this ever stops failing, the fixture has stopped being adversarial and
    the other tests would be passing vacuously.
    """
    scrambled = _old_buggy_pattern(LEFT, RIGHT)
    mismatched = int((scrambled["ID"].values != LEFT["ID"].values).sum())
    assert mismatched > 0, (
        "fixture no longer reproduces the historical defect; the remaining "
        "tests would be vacuous"
    )


def test_join_preserves_row_order_and_identifiers():
    out = sorted_asof_join(
        LEFT, RIGHT, id_col="ID", date_col="when", by_cols=["lat", "lon"]
    )
    assert len(out) == len(LEFT)
    assert list(out["ID"]) == list(LEFT["ID"])
    assert list(out.index) == list(range(len(LEFT)))


def test_join_attaches_correct_values_per_identifier():
    out = sorted_asof_join(
        LEFT, RIGHT, id_col="ID", date_col="when", by_cols=["lat", "lon"]
    )
    got = dict(zip(out["ID"], out["ext"]))
    assert got == EXPECTED_EXT


def test_join_does_not_detach_labels_from_rows():
    """The property that actually matters downstream.

    Every non-key left column must travel with its own identifier.
    """
    out = sorted_asof_join(
        LEFT, RIGHT, id_col="ID", date_col="when", by_cols=["lat", "lon"]
    )
    assert dict(zip(out["ID"], out["label"])) == dict(zip(LEFT["ID"], LEFT["label"]))
    assert dict(zip(out["ID"], out["lat"])) == dict(zip(LEFT["ID"], LEFT["lat"]))


def test_join_is_independent_of_left_input_order():
    """Shuffling the input rows must not change any row's result.

    This is the property the old positional restore could never provide.
    """
    shuffled = LEFT.sample(frac=1.0, random_state=7).reset_index(drop=True)
    a = sorted_asof_join(
        LEFT, RIGHT, id_col="ID", date_col="when", by_cols=["lat", "lon"]
    )
    b = sorted_asof_join(
        shuffled, RIGHT, id_col="ID", date_col="when", by_cols=["lat", "lon"]
    )
    by_id_a = dict(zip(a["ID"], a["ext"]))
    by_id_b = dict(zip(b["ID"], b["ext"]))
    assert by_id_a == by_id_b == EXPECTED_EXT
    # order must track the input order given
    assert list(b["ID"]) == list(shuffled["ID"])


def test_join_accepts_string_dates():
    left = LEFT.copy()
    right = RIGHT.copy()
    left["when"] = left["when"].dt.strftime("%Y-%m-%d")
    right["when"] = right["when"].dt.strftime("%Y-%m-%d")
    out = sorted_asof_join(
        left, right, id_col="ID", date_col="when", by_cols=["lat", "lon"]
    )
    assert list(out["ID"]) == list(left["ID"])
    assert dict(zip(out["ID"], out["ext"])) == EXPECTED_EXT


def test_duplicate_left_ids_raise():
    dup = pd.concat([LEFT, LEFT.head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="must be unique"):
        sorted_asof_join(
            dup, RIGHT, id_col="ID", date_col="when", by_cols=["lat", "lon"]
        )


def test_missing_id_column_raises():
    with pytest.raises(KeyError, match="no identifier column"):
        sorted_asof_join(
            LEFT.drop(columns=["ID"]),
            RIGHT,
            id_col="ID",
            date_col="when",
            by_cols=["lat", "lon"],
        )


def test_missing_group_column_raises():
    with pytest.raises(KeyError, match="group column"):
        sorted_asof_join(
            LEFT, RIGHT, id_col="ID", date_col="when", by_cols=["nope"]
        )


def test_unmatched_rows_raise_by_default_and_pass_when_opted_in():
    """A left row earlier than any reference row has no backward match."""
    left = pd.DataFrame(
        {
            "ID": ["a", "b"],
            "when": pd.to_datetime(["2019-01-01", "2020-01-05"]),
            "lat": [1.0, 1.0],
            "lon": [1.0, 1.0],
        }
    )
    with pytest.raises(ValueError, match="no backward match"):
        sorted_asof_join(
            left, RIGHT, id_col="ID", date_col="when", by_cols=["lat", "lon"]
        )

    out = sorted_asof_join(
        left,
        RIGHT,
        id_col="ID",
        date_col="when",
        by_cols=["lat", "lon"],
        allow_unmatched=True,
    )
    assert len(out) == len(left)
    assert list(out["ID"]) == ["a", "b"]
    assert pd.isna(out.loc[0, "ext"])


def test_invalid_direction_raises():
    with pytest.raises(ValueError, match="direction must be"):
        sorted_asof_join(
            LEFT,
            RIGHT,
            id_col="ID",
            date_col="when",
            by_cols=["lat", "lon"],
            direction="sideways",
        )


def test_join_without_group_columns_works():
    left = LEFT.drop(columns=["lat", "lon"]).copy()
    right = RIGHT.drop(columns=["lat", "lon"]).copy()
    out = sorted_asof_join(left, right, id_col="ID", date_col="when")
    assert list(out["ID"]) == list(left["ID"])
    assert len(out) == len(left)
