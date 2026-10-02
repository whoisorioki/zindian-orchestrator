"""CV strategy helpers for Zindian SoT compliance.

Provides a small compatibility layer so skills and shared training
functions can obtain a CV splitter or explicit splits from the
competition `challenge_config.json` `cv_strategy` block.

The helpers do NOT write to `challenge_config.json` — they only read
and return splitter objects or split iterators.
"""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    GroupKFold,
    TimeSeriesSplit,
)

from .config import ChallengeConfig, get_seed


def _read_strategy(config: ChallengeConfig | None = None) -> dict:
    if config is None:
        try:
            config = ChallengeConfig.load()
        except Exception:
            # Return empty strategy if config not available (e.g., unit tests)
            return {}
    return config.get("cv_strategy", {}) or {}


def make_cv_splitter(
    cv_strategy: dict | None = None,
    n_splits: int | None = None,
    random_seed: int | None = None,
):
    """Return an sklearn splitter instance according to `cv_strategy`.

    Supported shapes in `cv_strategy`:
      - {"type": "stratified", "n_splits": 5}
      - {"type": "group", "n_splits": 5}
      - {"type": "kfold", "n_splits": 5}
    Falls back to StratifiedKFold when unspecified.
    """
    strat = cv_strategy or _read_strategy(None)
    # Configs in the wild use both conventions -- "StratifiedKFold"/"GroupKFold"
    # and "stratified"/"group" -- so normalise case before dispatch. Without
    # this, class-style names fell through the branches below and reached the
    # fail-loud path, which would break legitimate configs that used to work.
    ctype = str(strat.get("type", "stratified")).strip()
    ctype_norm = ctype.lower()
    n = n_splits or strat.get("n_splits", 5)
    # Resolve seed: prefer caller-provided `random_seed`, then strategy values,
    # finally fall back to the canonical `reproducibility.seed` via `get_seed()`.
    seed: int
    if random_seed is not None:
        seed = random_seed
    else:
        val = strat.get("random_seed", strat.get("seed", None))
        seed = int(val) if val is not None else get_seed()

    if ctype_norm in ("stratified", "strat", "stratify", "stratifiedkfold"):
        return StratifiedKFold(n_splits=int(n), shuffle=True, random_state=int(seed))
    if ctype_norm in ("group", "groupkfold"):
        return GroupKFold(n_splits=int(n))
    if ctype_norm in ("kfold", "random", "timeseriessplit"):
        if ctype_norm == "timeseriessplit":
            return TimeSeriesSplit(n_splits=int(n))
        return KFold(n_splits=int(n), shuffle=True, random_state=int(seed))

    # SoT S-4: fail loud on an unhandled strategy. This previously fell
    # through to a shuffled KFold for ANY unrecognised type -- including
    # "BufferedSpatialCV" -- so a spatial competition silently ran random
    # folds and wrote them tagged with the spatial cv_strategy_id. A silent
    # substitution contradicts config and destroys provenance; there is no
    # safe default, so refuse.
    raise NotImplementedError(
        f"Unsupported cv_strategy.type {ctype!r}. Known types: "
        f"'stratified', 'group', 'kfold', 'TimeSeriesSplit'. Refusing to "
        f"substitute a random KFold: doing so would record a different "
        f"strategy than configured (SoT S-4/S-5). For spatially buffered CV, "
        f"load the explicit splits persisted by skill_05 via "
        f"load_explicit_cv_splits() rather than constructing a splitter here."
    )


def get_cv_splits(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None = None,
    cv_strategy: dict | None = None,
    random_seed: int | None = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, val_idx) pairs according to the cv strategy.

    If `cv_strategy` indicates a group CV, `groups` must be provided.
    """
    splitter = make_cv_splitter(cv_strategy=cv_strategy, random_seed=random_seed)
    if isinstance(splitter, GroupKFold) and groups is None:
        raise ValueError("Group CV requires `groups` to be provided")
    return splitter.split(X, y, groups) if groups is not None else splitter.split(X, y)


def materialize_cv_splits(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None = None,
    cv_strategy: dict | None = None,
    random_seed: int | None = None,
) -> list[list[list[int]]]:
    """Return concrete train/validation split arrays for the active strategy."""
    return [
        [
            np.asarray(train_idx, dtype=np.int64).tolist(),
            np.asarray(val_idx, dtype=np.int64).tolist(),
        ]
        for train_idx, val_idx in get_cv_splits(
            X,
            y,
            groups=groups,
            cv_strategy=cv_strategy,
            random_seed=random_seed,
        )
    ]


def load_explicit_cv_splits(
    state: dict | None,
) -> list[tuple[np.ndarray, np.ndarray]] | None:
    """Load concrete CV splits persisted by Skill 05, if available."""
    if not isinstance(state, dict):
        return None

    raw_splits = state.get("cv_split_indices")
    if not isinstance(raw_splits, list) or not raw_splits:
        return None

    explicit_splits: list[tuple[np.ndarray, np.ndarray]] = []
    for split in raw_splits:
        if not isinstance(split, (list, tuple)) or len(split) != 2:
            return None
        train_idx, val_idx = split
        explicit_splits.append(
            (
                np.asarray(train_idx, dtype=np.int64),
                np.asarray(val_idx, dtype=np.int64),
            )
        )
    return explicit_splits
