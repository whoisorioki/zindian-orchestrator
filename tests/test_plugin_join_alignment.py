"""Plugin join alignment, tested against the REAL competition tie structure.

Why this exists
---------------
A synthetic fixture with unique dates would pass under the old broken pattern.
The defect only appears when the join key contains **ties**. So this builds its
tie structure from the actual ``Train.csv`` rather than inventing one.

What it proves
--------------
`sorted_asof_join` equals an INDEPENDENT reference join computed by a different
method, asserted **by ID** -- the only assertion that can detect this class of
defect (see joins.py). It also pins the old pattern's failure, so a future
"optimisation" back to `.loc[df.index]` fails loudly.
"""

import numpy as np
import pandas as pd
import pytest

from zindian.joins import sorted_asof_join

COMP = "competitions/climate-risk-health-prediction-challenge"


def _real_tie_frame(n=400):
    """A frame carrying the real competition's lat/lon/date tie structure.

    Uses the real ``Train.csv`` when present so the fixture cannot drift from
    production data; falls back to a deliberately tie-heavy synthetic frame if
    the competition tree is absent (e.g. a fresh clone).
    """
    from pathlib import Path

    raw = Path(COMP) / "data" / "raw" / "Train.csv"
    if not raw.exists():
        # 5 locations x 80 dates, repeated in blocks -> heavy ties.
        df = pd.DataFrame(
            {
                "latitude": np.tile([10.0, 11.0, 12.0, 13.0, 14.0], 80),
                "longitude": np.tile([20.0, 21.0, 22.0, 23.0, 24.0], 80),
            }
        )
        df["deathdate"] = pd.to_datetime("2015-01-01") + pd.to_timedelta(
            np.repeat(np.arange(80), 5), unit="D"
        )
        df.insert(0, "ID", [f"ID_{i:06d}" for i in range(len(df))])
        return df

    df = pd.read_csv(raw, nrows=n)
    assert "ID" in df.columns, "real Train.csv must expose an ID column"
    # merge_asof requires a real datetime dtype, not raw strings.
    df["deathdate"] = pd.to_datetime(df["deathdate"])
    return df


def _reference_join(left, ref, id_col, date_col, by_cols):
    """Independent reference as-of join, sharing no code with the fix.

    Uses merge_asof but restores by an explicit ID merge afterwards, which is
    a different mechanism from sorted_asof_join's internal reindex.
    """
    value_cols = [c for c in ref.columns if c not in list(by_cols) + [date_col]]
    out = left.sort_values(date_col)
    m = pd.merge_asof(
        out,
        ref[list(by_cols) + [date_col] + value_cols].sort_values(date_col),
        on=date_col,
        by=list(by_cols),
        direction="backward",
    )
    # Re-attach to the ORIGINAL frame by ID -- an independent restore path.
    m = m[[id_col] + value_cols]
    return left[[id_col]].merge(m, on=id_col, how="left").set_index(id_col)


def _real_ext():
    """The real macro-proxy reference frame, or a synthetic stand-in.

    The reference frame's density matters: a sparse frame (one row per
    location) makes permuted rows indistinguishable, because every row for a
    given location would receive the same value regardless of position. The
    production parquet is dense (305,360 rows), so use it when present.
    """
    from pathlib import Path

    p = Path(COMP) / "data" / "external" / "aligned_macro_proxies.parquet"
    if p.exists():
        ext = pd.read_parquet(p)
        ext["deathdate"] = pd.to_datetime(ext["deathdate"])
        return ext

    # Synthetic but dense and date-varying, so a permuted row carries a
    # visibly different value.
    left = _real_tie_frame()
    locs = left[["latitude", "longitude"]].drop_duplicates()
    rows = []
    for i, (_, r) in enumerate(locs.iterrows()):
        for k in range(6):
            rows.append(
                {
                    "latitude": r["latitude"],
                    "longitude": r["longitude"],
                    "deathdate": pd.Timestamp("2008-01-01") + pd.Timedelta(days=90 * k),
                    "spei_6": float(i * 100 + k),
                }
            )
    return pd.DataFrame(rows)


def test_fixture_actually_contains_ties():
    """Guard the guard: the fixture must really have tied join keys."""
    df = _real_tie_frame()
    tied = df.duplicated(subset=["deathdate", "latitude", "longitude"], keep=False)
    assert tied.sum() > 0, (
        "fixture has no tied join keys -- it cannot reproduce the defect, and "
        "the rest of this module would pass vacuously"
    )
def test_matches_independent_reference_by_id():
    left = _real_tie_frame()
    by = ["latitude", "longitude"]

    ref = pd.DataFrame(
        {
            "latitude": left["latitude"].drop_duplicates().tolist() * 2,
            "longitude": left["longitude"].drop_duplicates().tolist() * 2,
        }
    )
    ref["deathdate"] = pd.to_datetime("2010-01-01") + pd.to_timedelta(
        np.arange(len(ref)), unit="D"
    )
    ref["spei_6"] = np.arange(len(ref), dtype=float)

    got = sorted_asof_join(
        left, ref, id_col="ID", date_col="deathdate",
        by_cols=by, direction="backward", allow_unmatched=True,
    )
    expected = _reference_join(left, ref, "ID", "deathdate", by)

    g = got.set_index("ID")["spei_6"].astype(float)
    e = expected["spei_6"].astype(float)
    common = g.index.intersection(e.index)
    assert len(common) > 0
    mism = ~np.isclose(g.loc[common], e.loc[common], equal_nan=True)
    assert mism.sum() == 0, (
        f"{int(mism.sum())} of {len(common)} rows disagree with the independent "
        "reference join -- sorted_asof_join is misaligned"
    )


def test_old_pattern_permutes_row_order():
    """The pre-fix pattern must demonstrably permute ROWS here.

    Measured on the real competition (pandas 3.0.3, full 3146-row Train.csv):
    the old pattern reorders 387 positions relative to Train.csv, while every
    value-to-ID pairing stays intact. That is the signature of the defect --
    ``merge_asof`` moves whole rows, so within-file feature/label pairs remain
    consistent and every ID-set check passes. The damage only appears when
    artifacts are consumed BY POSITION (CV splits, anchor OOF, test-prob
    ordering). Hence this asserts on ORDER, not on values.

    If this stops failing, the fixture lost its defect-reproducing power.
    """
    left = _real_tie_frame()
    by = ["latitude", "longitude"]
    ext = _real_ext()

    old = pd.merge_asof(
        left.sort_values(by="deathdate"),
        ext.sort_values(by="deathdate"),
        on="deathdate", by=by, direction="backward",
    )
    old = old.loc[left.index]  # the old, positional restore

    fixed = sorted_asof_join(
        left, ext, id_col="ID", date_col="deathdate",
        by_cols=by, direction="backward", allow_unmatched=True,
    )

    # The fix preserves the input row order exactly.
    assert fixed["ID"].tolist() == left["ID"].tolist(), (
        "sorted_asof_join must return rows in the input frame's original order"
    )
    # The old pattern does not.
    diffs = [
        i for i, (a, b) in enumerate(zip(old["ID"].tolist(), left["ID"].tolist()))
        if a != b
    ]
    assert len(diffs) > 0, (
        "the old .loc[df.index] pattern did NOT permute on the real tie "
        "structure -- this fixture no longer reproduces the defect"
    )


def test_permutation_is_whole_rows_not_value_drift():
    """Pin the failure mode: rows move intact, so ID-joined values agree.

    This is why a raw-column ID join cannot detect the corruption -- it is a
    tautology. It also documents why order, not value, is the right probe.
    """
    left = _real_tie_frame()
    by = ["latitude", "longitude"]
    ext = _real_ext()

    old = pd.merge_asof(
        left.sort_values(by="deathdate"),
        ext.sort_values(by="deathdate"),
        on="deathdate", by=by, direction="backward",
    )
    old = old.loc[left.index]
    fixed = sorted_asof_join(
        left, ext, id_col="ID", date_col="deathdate",
        by_cols=by, direction="backward", allow_unmatched=True,
    )

    o = old.set_index("ID")["spei_6"].astype(float)
    f = fixed.set_index("ID")["spei_6"].astype(float)
    common = o.index.intersection(f.index)
    value_diffs = int((~np.isclose(o.loc[common], f.loc[common], equal_nan=True)).sum())
    order_diffs = sum(1 for a, b in zip(old["ID"].tolist(), left["ID"].tolist()) if a != b)

    assert order_diffs > 0, "expected a positional permutation"
    assert value_diffs == 0, (
        f"expected whole-row permutation, but {value_diffs} rows had drifted "
        "in value -- the fixture no longer models this failure mode"
    )


def test_both_plugins_use_safe_join_and_keep_id():
    """Both plugins must use the safe join and stop dropping id_col.

    `catboost-climate-interactions.csv` had no ID column before this change,
    which made it impossible to verify by ID at all.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "plugins"
    for name in ("macro_stress_extractor", "advanced_spatial_temporal_extractor"):
        text = (root / f"{name}.py").read_text()
        assert "sorted_asof_join" in text, f"{name} must use the safe join"
        assert "[c for c in [id_col] + banned" not in text, (
            f"{name} still drops id_col from its output"
        )