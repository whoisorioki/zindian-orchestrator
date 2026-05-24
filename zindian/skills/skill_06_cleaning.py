"""
Skill 06 — Cleaning / Preprocessing

Implements lightweight, competition-agnostic cleaning steps used by later skills.
Follows the repository style: reads processed feature CSVs, writes cleaned CSVs,
and updates `SKILL_STATE.json` with cleaning metadata.

Usage: python -m zindian.skills.skill_06_cleaning
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

from zindian.paths import resolve_competition_paths
from zindian.state import SkillStateStore


def _fill_numeric(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        if df[c].isnull().any():
            median = df[c].median()
            df[c] = df[c].fillna(median)
    return df


def _drop_constant(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    nunique = df.nunique(dropna=False)
    const_cols = [str(c) for c, n in nunique.items() if n <= 1]
    return df.drop(columns=const_cols), const_cols


def run(dry_run: bool = False) -> Dict[str, object]:
    print("\n" + "=" * 60)
    print("SKILL 06 — Cleaning / Preprocessing")
    print("=" * 60 + "\n")

    paths = resolve_competition_paths(require_competition=True)
    store = SkillStateStore(paths.state_path)
    state = store.read()

    proc_dir = paths.data_processed_dir
    out_train = proc_dir / "features_train_clean.csv"
    out_test = proc_dir / "features_test_clean.csv"

    train_in = proc_dir / "features_train.csv"
    test_in = proc_dir / "features_test.csv"

    if not train_in.exists() or not test_in.exists():
        raise FileNotFoundError("Processed feature files not found. Run Skill 07 first.")

    train = pd.read_csv(train_in)
    test = pd.read_csv(test_in)

    # Basic cleaning: fill numeric NaNs with median, drop constant cols
    train = _fill_numeric(train)
    test = _fill_numeric(test)

    train, const_train = _drop_constant(train)
    test, const_test = _drop_constant(test)

    # Ensure ID and target preserved ordering
    if "ID" not in train.columns:
        raise RuntimeError("ID column missing from training features")

    if not dry_run:
        proc_dir.mkdir(parents=True, exist_ok=True)
        train.to_csv(out_train, index=False)
        test.to_csv(out_test, index=False)

        state_patch = {
            "cleaning_completed_at": datetime.now(timezone.utc).isoformat(),
            "cleaning_dropped_constants_train": const_train,
            "cleaning_dropped_constants_test": const_test,
        }
        store.update(**state_patch)

    return {
        "status": "OK",
        "train_out": str(out_train),
        "test_out": str(out_test),
        "dropped_train": const_train,
        "dropped_test": const_test,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2))
