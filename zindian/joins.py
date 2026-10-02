"""Order-safe as-of join utilities.

Background
----------
`pd.merge_asof` is a left join that requires the *left* frame to be sorted
on the join key. The natural way to satisfy that is to sort the left frame,
merge, then restore the original row order afterwards, typically with::

    merged = merged.loc[df.index]

That restore is **incorrect whenever the left key contains ties**. Under a
tie, `merge_asof` can emit rows in an order that no longer corresponds to the
left frame, and `.loc[<RangeIndex>]` resolves each label positionally rather
than by identity. The result is a silent, local row-scrambling: labels stay
attached to the
wrong feature values while every row count and every file-level ID check
still passes.

This happened for real: a left-side date tie across 56 duplicate
``(date, lat, lon)`` keys scrambled 368 of 2505 training rows in one
competition, corrupting 14 downstream branches.

The fix used here is to make row order **irrelevant** to correctness. The
left frame's identifier is carried through the join and the result is
restored by ``reindex`` on that identifier. ``reindex`` is a label-based
lookup and does not care what order ``merge_asof`` happened to emit.

Every function in this module fails loudly rather than returning a
plausible-but-wrong frame.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd


__all__ = ["sorted_asof_join"]


def _validate_unique_ids(df: pd.DataFrame, id_col: str, role: str) -> None:
    if id_col not in df.columns:
        raise KeyError(
            f"{role} frame has no identifier column {id_col!r}. "
            f"Available columns: {list(df.columns)}"
        )
    dupes = df[id_col].duplicated()
    if bool(dupes.any()):
        sample = df.loc[dupes, id_col].head(5).tolist()
        raise ValueError(
            f"{role} frame column {id_col!r} must be unique for an "
            f"order-safe as-of join, but {int(dupes.sum())} duplicate(s) "
            f"were found, e.g. {sample}. Deduplicate before joining."
        )


def sorted_asof_join(
    df: pd.DataFrame,
    ext_df: pd.DataFrame,
    *,
    id_col: str,
    date_col: str,
    by_cols: Sequence[str] = (),
    tolerance=None,
    direction: str = "backward",
    allow_unmatched: bool = False,
) -> pd.DataFrame:
    """Left-join ``ext_df`` onto ``df`` by nearest matching date.

    ``df`` must carry a unique ``id_col``. The result is guaranteed to have
    exactly ``len(df)`` rows in exactly ``df``'s original order, with a
    fresh ``RangeIndex``.

    ``direction`` defaults to ``"backward"`` — the most recent reference date
    at or before the left date — which is the leakage-safe direction for
    time-ordered features. ``"forward"`` and ``"nearest"`` pass through.

    ``allow_unmatched`` defaults to ``False``: a left row with no match
    raises rather than silently emitting NaN. Rows earlier than the first
    reference date have no backward match and are the usual cause.

    Raises
    ------
    KeyError, ValueError, AssertionError
        On duplicate identifiers, missing columns, unmatched rows, or any
        failure of the post-join identity checks.
    """
    if direction not in ("backward", "forward", "nearest"):
        raise ValueError(
            f"direction must be 'backward', 'forward' or 'nearest', "
            f"got {direction!r}"
        )

    _validate_unique_ids(df, id_col, "left")

    for col, frame, role in ((date_col, df, "left"), (date_col, ext_df, "right")):
        if col not in frame.columns:
            raise KeyError(
                f"{role} frame has no date column {col!r}. "
                f"Available columns: {list(frame.columns)}"
            )
    for col in by_cols:
        if col not in df.columns:
            raise KeyError(f"left frame has no group column {col!r}")
        if col not in ext_df.columns:
            raise KeyError(f"right frame has no group column {col!r}")

    n_left = len(df)
    # Snapshot identifiers BEFORE any sorting. Everything below is allowed to
    # reorder rows; this vector is the only source of truth for final order.
    raw_ids = df[id_col].copy()

    left = df.copy()
    right = ext_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(left[date_col]):
        left[date_col] = pd.to_datetime(left[date_col])
    if not pd.api.types.is_datetime64_any_dtype(right[date_col]):
        right[date_col] = pd.to_datetime(right[date_col])

    # mergesort is a stable sort: it preserves the original relative order of
    # rows that tie on date_col, which keeps the merge deterministic.
    left = left.sort_values(date_col, kind="mergesort")
    right = right.sort_values(date_col, kind="mergesort")

    merged = pd.merge_asof(
        left,
        right,
        on=date_col,
        by=list(by_cols) or None,
        direction=direction,
        tolerance=tolerance,
    )

    # --- Order-independent restore ---------------------------------------
    # This is the whole point: restore by label, not by position. Whether
    # merge_asof emitted a tie-scrambled order or a clean one is irrelevant.
    if merged[id_col].duplicated().any():
        raise AssertionError(
            "sorted_asof_join produced duplicate identifiers. The merge is "
            "ambiguous and the result cannot be safely restored."
        )

    out = merged.set_index(id_col).reindex(pd.Index(raw_ids, name=id_col))

    # --- Identity checks --------------------------------------------------
    if len(out) != n_left:
        raise AssertionError(
            f"join changed row count: expected {n_left}, got {len(out)}"
        )
    # After set_index the identifier IS the index, so this single comparison
    # carries both the ordering guarantee and the value-integrity guarantee.
    if not out.index.equals(pd.Index(raw_ids, name=id_col)):
        raise AssertionError(
            "join did not round-trip the left identifiers (values or order); "
            "refusing to return a mis-aligned frame."
        )

    out = out.reset_index()
    out.index = pd.RangeIndex(len(out))

    if not allow_unmatched:
        newly_added = [
            c for c in out.columns
            if c not in df.columns and c not in set(by_cols) and c != date_col
        ]
        unmatched = pd.Series(False, index=out.index)
        for col in newly_added:
            unmatched |= out[col].isna()
        if bool(unmatched.any()):
            offenders = out.loc[unmatched, id_col].head(5).tolist()
            raise ValueError(
                f"{int(unmatched.sum())} left row(s) had no {direction} match "
                f"in the reference frame for columns {newly_added}, e.g. IDs "
                f"{offenders}. Pass allow_unmatched=True to accept this "
                f"explicitly (they will be NaN and should be imputed)."
            )

    return out

