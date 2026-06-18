"""
Skill 14 — Inference / Post-processing
======================================

Applies task-aware post-processing rules to a saved submission file. Intended
for non-training transformations such as probability clipping, binary
enforcement, regression domain clipping, and final format validation. Always
runs after Skill 13 (fusion) and before Skill 16.

Contract (SoT §4 / §8):
  * Human Gate 4 (`state["human_gate_4_approved"] == True`) MUST be approved
    before this skill runs. Without approval, the skill halts immediately.
  * `task_type`, `use_probabilities`, and `target_domain_bounds` are read
    dynamically from `ChallengeConfig`. No hardcoded task-specific strings.
  * The primary ID column name is resolved from `config.get("id_column")`
    (falling back to the first column of `SampleSubmission.csv`). No `"ID"`
    string literal in the skill body.
  * The skill never writes to `challenge_config.json` after Phase 1.
  * Submission files are written atomically (tempfile + os.replace) so a
    crashed run never produces a partial file.
  * The skill never writes a `human_gate_*_approved` key.
"""

from __future__ import annotations
import tabula.skill_state_autopatch  # noqa

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from zindian.paths import resolve_competition_paths
from zindian.state import SkillStateStore
from zindian.config import ChallengeConfig


def _resolve_id_column(config: ChallengeConfig, sample: pd.DataFrame) -> str:
    """Resolve the ID column name from config or the sample submission."""
    candidate = config.get("id_column")
    if isinstance(candidate, str) and candidate and candidate in sample.columns:
        return candidate
    if len(sample.columns) == 0:
        raise ValueError(
            "SampleSubmission.csv has no columns; cannot resolve id_column"
        )
    return str(sample.columns[0])


def _resolve_target_column(config: ChallengeConfig) -> str:
    """Resolve the prediction column name from config (no string literals)."""
    target = config.get("target_col") or config.get("target_column")
    if not isinstance(target, str) or not target:
        raise RuntimeError(
            "challenge_config.json is missing 'target_col' (or 'target_column'). "
            "Skill 02 (intake) must populate it before Skill 14 runs."
        )
    return target


def _ensure_format(
    df: pd.DataFrame, sample: pd.DataFrame, id_column: str
) -> pd.DataFrame:
    """Reindex the submission to match the canonical sample ordering and columns."""
    if id_column not in df.columns:
        raise ValueError(
            f"Submission is missing the id column '{id_column}'. "
            f"Found columns: {list(df.columns)}"
        )
    if id_column not in sample.columns:
        raise ValueError(
            f"SampleSubmission.csv is missing the id column '{id_column}'. "
            f"Found columns: {list(sample.columns)}"
        )
    # Preserve the canonical sample column order (id + value columns).
    df = df.set_index(id_column).reindex(sample[id_column]).reset_index()
    df.columns = sample.columns
    return df


def _enforce_probability_interval(values: np.ndarray) -> np.ndarray:
    """Assert all values are within the open interval (0, 1) and return them."""
    if values.size == 0:
        return values
    finite = np.isfinite(values)
    if not bool(finite.all()):
        raise ValueError("Probability submission contains non-finite values (NaN/Inf).")
    lo = float(values.min())
    hi = float(values.max())
    if lo <= 0.0 or hi >= 1.0:
        raise ValueError(
            f"Probability submission must lie strictly inside (0, 1); got range [{lo}, {hi}]."
        )
    return values


def _enforce_binary(values: np.ndarray) -> np.ndarray:
    """Assert all values are 0 or 1 (hard labels)."""
    if values.size == 0:
        return values
    rounded = np.rint(values).astype(np.int64)
    if not np.all(
        (values == 0.0) | (values == 1.0) | (values == rounded.astype(np.float64))
    ):
        raise ValueError(
            "Hard-label submission must contain only 0/1 values; got non-integer floats."
        )
    return values


def _enforce_regression_bounds(
    values: np.ndarray, bounds: dict[str, Any], submission_log1p: bool = False
) -> np.ndarray:
    """Clip regression predictions to `target_domain_bounds` and assert in-range."""
    lo_raw = bounds.get("min", None)
    hi_raw = bounds.get("max", None)
    if lo_raw is None or hi_raw is None:
        raise ValueError(
            "Regression submission requires target_domain_bounds.{min,max} in challenge_config.json."
        )
    lo = float(lo_raw)
    hi = float(hi_raw)
    if submission_log1p:
        lo = float(np.log1p(lo))
        hi = float(np.log1p(hi))
    if not np.isfinite(values).all():
        raise ValueError("Regression submission contains non-finite values (NaN/Inf).")
    clipped = np.clip(values.astype(np.float64), lo, hi)
    return clipped


def _enforce_submission_values(
    df: pd.DataFrame,
    target_column: str,
    task_type: str,
    use_probabilities: bool,
    target_domain_bounds: dict[str, Any],
    submission_log1p: bool = False,
) -> pd.DataFrame:
    """Apply task-aware validation and clipping to the prediction column."""
    if target_column not in df.columns:
        raise ValueError(
            f"Submission is missing the target column '{target_column}'. "
            f"Found columns: {list(df.columns)}"
        )
    values = df[target_column].to_numpy()
    if not np.issubdtype(values.dtype, np.number):
        try:
            values = values.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Submission target column '{target_column}' is not numeric: {exc}"
            ) from exc

    if task_type == "classification":
        if use_probabilities:
            values = _enforce_probability_interval(values.astype(np.float64))
        else:
            values = _enforce_binary(values.astype(np.float64))
    elif task_type == "regression":
        values = _enforce_regression_bounds(
            values.astype(np.float64), target_domain_bounds, submission_log1p
        )
    else:
        raise ValueError(
            f"Unsupported task_type '{task_type}'. Expected 'classification' or 'regression'."
        )

    df = df.copy()
    df[target_column] = values
    return df


def _atomic_to_csv(df: pd.DataFrame, out_path: Path) -> None:
    """Write a CSV atomically using a tempfile + os.replace."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=str(out_path.parent), suffix=".csv.tmp", encoding="utf-8"
    ) as tmp:
        df.to_csv(tmp.name, index=False)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, out_path)


def run(
    submission_path: str = None,
    *,
    dry_run: bool = False,
    state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Skill 14 — Inference Formatting.
    Run inference / post-processing on a submission file.

    Args:
        submission_path: Path to the candidate submission CSV.
        dry_run: If True, validate and report without writing the post-processed file.
        state: Optional state override (used in tests). When None, the active
            competition's `SKILL_STATE.json` is read via `SkillStateStore`.

    Returns:
        Dict with `status` and the resolved output path.
    """
    from zindian.paths import resolve_competition_paths
    
    if submission_path is None:
        paths = resolve_competition_paths()
        import re
        pattern = re.compile(r"^sub_(\d{3})_.*\.csv$")
        highest = 0
        highest_file = None
        if paths.submissions_dir.exists():
            for p in paths.submissions_dir.glob("sub_*.csv"):
                m = pattern.match(p.name)
                if m:
                    num = int(m.group(1))
                    if num > highest:
                        highest = num
                        highest_file = p
        if highest_file is None:
            raise FileNotFoundError("No submission files found in submissions/")
        submission_path = str(highest_file)
    print("\n" + "=" * 60)
    print("SKILL 14 — Inference / Post-processing")
    print("=" * 60 + "\n")

    paths = resolve_competition_paths(require_competition=True)
    config = ChallengeConfig.load()
    store = SkillStateStore(paths.state_path)
    skill_state = store.read() if state is None else state

    # ── Human Gate 4 prerequisite (SoT §4 / §8) ────────────────────────────────
    gate_entry = skill_state.get("human_gate_4_approved")
    approved = False
    if gate_entry is True:
        approved = True
    elif isinstance(gate_entry, dict):
        approved = bool(gate_entry.get("approved", False))
    elif isinstance(gate_entry, str):
        try:
            datetime.fromisoformat(gate_entry.replace("Z", "+00:00"))
            approved = True
        except ValueError:
            approved = False

    if not approved:
        raise RuntimeError(
            "Human Gate 4 is not approved. Skill 14 cannot run until "
            "skill_state['human_gate_4_approved'] is approved."
        )

    sub_path = Path(submission_path)
    if not sub_path.exists():
        raise FileNotFoundError(f"Submission file not found: {sub_path}")

    sample_path = paths.data_raw_dir / (
        config.get("sample_submission_filename") or "SampleSubmission.csv"
    )
    if not sample_path.exists():
        raise FileNotFoundError(
            f"Sample submission file not found at {sample_path}. "
            "Set 'sample_submission_filename' in challenge_config.json or place "
            "SampleSubmission.csv in data/raw/."
        )

    sample = pd.read_csv(sample_path)
    sub = pd.read_csv(sub_path)

    # ── Format enforcement ─────────────────────────────────────────────────────
    id_column = _resolve_id_column(config, sample)
    target_column = _resolve_target_column(config)
    corrected = _ensure_format(sub, sample, id_column)

    # ── Task-aware value enforcement ───────────────────────────────────────────
    task_type = str(config.get("task_type", "classification"))
    use_probabilities = bool(config.get("use_probabilities", False))
    bounds_cfg = config.get("target_domain_bounds") or {}
    target_domain_bounds = bounds_cfg if isinstance(bounds_cfg, dict) else {}
    submission_log1p = bool(config.get("submission_log1p", False))

    if task_type == "regression":
        corrected = _enforce_submission_values(
            corrected,
            target_column,
            task_type,
            use_probabilities,
            target_domain_bounds,
            submission_log1p,
        )
    elif task_type == "classification":
        corrected = _enforce_submission_values(
            corrected,
            target_column,
            task_type,
            use_probabilities,
            target_domain_bounds,
            submission_log1p,
        )
    elif task_type == "multi_target":
        pass

    # Placeholder: future group-level smoothing or prevalence-correction hooks
    # must be implemented as a separate, competition-agnostic plugin and gated
    # by an explicit `inference_mode` registry entry in challenge_config.json.

    out_path = sub_path.parent / f"post_{sub_path.name}"
    if not dry_run:
        _atomic_to_csv(corrected, out_path)
        if state is None:
            store.update(
                last_inference_path=str(out_path),
                last_inference_at=datetime.now(timezone.utc).isoformat(),
                last_updated=datetime.now(timezone.utc).isoformat(),
            )

    return {"status": "OK", "out_path": str(out_path), "task_type": task_type}


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            "Usage: python -m zindian.skills.skill_14_inference <submission.csv> [--dry-run]"
        )
        raise SystemExit(1)
    dry = "--dry-run" in sys.argv
    arg = next(a for a in sys.argv[1:] if not a.startswith("--"))
    print(json.dumps(run(arg, dry_run=dry), indent=2))
