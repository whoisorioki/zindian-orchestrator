"""
Skill 14 — Inference / Post-processing

Applies optional post-processing rules to a saved submission file. Intended for
non-training transformations such as group-based smoothing, tie-breaking, and
final format validation. Always run after Skill 13 (fusion) and before Skill 16.

Usage: python -m zindian.skills.skill_14_inference <submission.csv>
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from zindian.paths import resolve_competition_paths
from zindian.state import SkillStateStore
from zindian.config import ChallengeConfig


def _ensure_format(df: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    # Reindex and enforce column order
    df = df.set_index("ID").reindex(sample["ID"]).reset_index()
    # Ensure column names match sample
    df.columns = sample.columns
    return df


def run(submission_path: str, *, dry_run: bool = False) -> Dict[str, object]:
    print("\n" + "=" * 60)
    print("SKILL 14 — Inference / Post-processing")
    print("=" * 60 + "\n")

    paths = resolve_competition_paths(require_competition=True)
    config = ChallengeConfig.load()
    store = SkillStateStore(paths.state_path)
    state = store.read()

    sub_path = Path(submission_path)
    if not sub_path.exists():
        raise FileNotFoundError(f"Submission file not found: {sub_path}")

    sample = pd.read_csv(paths.data_raw_dir / "SampleSubmission.csv")
    sub = pd.read_csv(sub_path)

    # Basic format enforcement
    corrected = _ensure_format(sub, sample)

    # Placeholder: potential smoothing hooks go here (no-op by default)
    # e.g., group-level prevalence correction, but must be competition-compliant

    out_path = sub_path.parent / f"post_{sub_path.name}"
    if not dry_run:
        corrected.to_csv(out_path, index=False)
        store.update(last_inference_path=str(out_path), last_inference_at=datetime.now(timezone.utc).isoformat())

    return {"status": "OK", "out_path": str(out_path)}


if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python -m zindian.skills.skill_14_inference <submission.csv>")
        raise SystemExit(1)
    print(json.dumps(run(sys.argv[1]), indent=2))
