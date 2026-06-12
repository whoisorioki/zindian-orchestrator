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
)

from .config import ChallengeConfig, get_seed


def _read_strategy(config: ChallengeConfig | None = None) -> dict:
    if config is None:
        config = ChallengeConfig.load()
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
    ctype = strat.get("type", "stratified")
    n = n_splits or strat.get("n_splits", 5)
    # Resolve seed: prefer caller-provided `random_seed`, then strategy values,
    # finally fall back to the canonical `reproducibility.seed` via `get_seed()`.
    seed: int
    if random_seed is not None:
        seed = random_seed
    else:
        val = strat.get("random_seed", strat.get("seed", None))
        seed = int(val) if val is not None else get_seed()

    if ctype in ("stratified", "strat", "stratify"):
        return StratifiedKFold(n_splits=int(n), shuffle=True, random_state=int(seed))
    if ctype in ("group", "groupkfold"):
        return GroupKFold(n_splits=int(n))
    return KFold(n_splits=int(n), shuffle=True, random_state=int(seed))


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
