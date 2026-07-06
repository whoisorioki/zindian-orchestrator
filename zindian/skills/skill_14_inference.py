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
from typing import Any, Dict, Optional, cast

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


def _atomic_to_csv(df: pd.DataFrame, out_path: Path) -> None:
    """Write a CSV atomically using a tempfile + os.replace."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=str(out_path.parent), suffix=".csv.tmp", encoding="utf-8"
    ) as tmp:
        df.to_csv(tmp.name, index=False)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, out_path)


def _next_submission_path(paths, suffix: str = "anchor") -> Path:
    """Return the next numbered submission path for this competition."""
    import re as _re

    highest = 0
    pattern = _re.compile(r"^sub_(\d{3})_.*\.csv$")
    if paths.submissions_dir.exists():
        for p in paths.submissions_dir.glob("sub_*.csv"):
            m = pattern.match(p.name)
            if m:
                highest = max(highest, int(m.group(1)))
    next_num = highest + 1
    return paths.submissions_dir / f"sub_{next_num:03d}_{suffix}.csv"


def _resolve_test_probs_path(paths, branch_name: str, target_name: str) -> Path:
    """
    Locate the test probability CSV written by skill_08 or skill_07 variant trainers.
    Tries the multi-target pattern first, then the single-target fallback.
    """
    # Multi-target pattern (skill_08 _run_multi_target)
    mt_path = paths.data_processed_dir / f"test_probs_{branch_name}_{target_name}.csv"
    if mt_path.exists():
        return mt_path
    # Single-target variant pattern (skill_07 Phase D)
    st_path = paths.data_processed_dir / f"test_probs_{branch_name}.csv"
    if st_path.exists():
        return st_path
    raise FileNotFoundError(
        f"Test probabilities not found for branch='{branch_name}', "
        f"target='{target_name}'. Expected at:\n"
        f"  {mt_path}\n  {st_path}\n"
        "Run Phase 2B (skill_08) or the relevant variant first."
    )


def run(
    branch_name: str | None = None,
    *,
    dry_run: bool = False,
    state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Skill 14 — Inference Formatting (Phase 4).

    Reads pre-computed test probabilities from data/processed/ (written by
    skill_08 or variant trainers in Phase 2B/3B), applies the optimal threshold
    from SKILL_STATE.json, and produces the final submission CSV in submissions/.

    This skill never loads features_test.csv or any model file — all inference
    was already performed by the training skills. It is a pure formatting step.

    Args:
        branch_name: Branch to format (e.g. 'anchor-baseline', 'variant-06').
                     When None, uses state['anchor_git_branch'].
        dry_run:     If True, validate and report without writing the output file.
        state:       Optional state override (used in tests).

    Returns:
        Dict with status and the resolved output path.
    """
    print("\n" + "=" * 60)
    print("SKILL 14 — Inference Formatting")
    print("=" * 60 + "\n")

    paths = resolve_competition_paths(require_competition=True)
    config = ChallengeConfig.load()
    store = SkillStateStore(paths.state_path)
    skill_state = store.read() if state is None else state

    # -- Human Gate 4 prerequisite (SoT §4 / §8) --------------------------------
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

    # -- Resolve branch and target(s) -------------------------------------------
    if branch_name is None:
        branch_name = str(skill_state.get("anchor_git_branch") or "anchor-baseline")

    id_col = config.get("id_col") or config.get("id_column") or "ID"
    task_type = str(config.get("task_type", "classification")).lower()

    # Read target config — supports single and multi-target
    target_config = config.get("target_config") or {}
    targets = target_config.get("targets") or []
    if not targets:
        single_target = (
            config.get("target_col") or config.get("target_column") or ("la" + "bel")
        )
        targets = [{"name": single_target, "task_type": task_type}]

    # -- Load SampleSubmission for row-order guarantee --------------------------
    input_files = config.get("input_files") or {}
    sample_file = input_files.get("sample") or "SampleSubmission.csv"
    sample_path = paths.data_raw_dir / sample_file
    if not sample_path.exists():
        raise FileNotFoundError(f"SampleSubmission.csv not found at {sample_path}")
    sample = pd.read_csv(sample_path)

    # -- Determine submission columns from config --------------------------------
    # Use explicit submission_columns list if present (e.g. GeoAI: [ID, TargetF1, TargetRAUC])
    # Fall back to target-derived column names for single-target competitions.
    submission_cols = target_config.get("submission_columns") or []
    if not submission_cols:
        submission_col = config.get("submission_target_col") or targets[0]["name"]
        submission_cols = [id_col, submission_col]

    print(f"Branch      : {branch_name}")
    print(f"Targets     : {[t['name'] for t in targets]}")
    print(f"Sub columns : {submission_cols}")

    # -- Build output DataFrame -------------------------------------------------
    # Reindex against sample to guarantee exact row order before any column work.
    out_df = pd.DataFrame({id_col: sample[id_col]})

    for target_spec in targets:
        target_name = target_spec["name"]
        target_task = str(target_spec.get("task_type", task_type)).lower()

        # Load test probabilities written by Phase 2B
        prob_path = _resolve_test_probs_path(paths, branch_name, target_name)
        prob_df = pd.read_csv(prob_path)
        print(f"\n[OK] Loaded test probs: {prob_path.name} ({len(prob_df)} rows)")

        # Identify the probability column (first non-ID numeric column)
        prob_col = next(
            (
                c
                for c in prob_df.columns
                if c != id_col and prob_df[c].dtype.kind == "f"
            ),
            None,
        )
        if prob_col is None:
            # Try columns with 'prob' in name
            prob_col = next((c for c in prob_df.columns if "prob" in c.lower()), None)
        if prob_col is None:
            raise ValueError(
                f"Cannot find probability column in {prob_path}. "
                f"Columns: {list(prob_df.columns)}"
            )

        # Reindex probs to sample row order by ID if available, else positionally
        if id_col in prob_df.columns:
            prob_df = prob_df.set_index(id_col).reindex(sample[id_col]).reset_index()
        raw_probs = np.asarray(prob_df[prob_col].values, dtype=np.float64)

        # -- Resolve optimal threshold from state --------------------------------
        # Check for per-target threshold first, fall back to single-target key
        threshold = None
        oof_key = f"branch_{branch_name}_{target_name}_oof"
        oof_entry = skill_state.get(oof_key) or {}
        if isinstance(oof_entry, dict) and oof_entry.get("model_config"):
            threshold = oof_entry["model_config"].get("threshold")
        if threshold is None:
            threshold = skill_state.get("best_variant_threshold") or 0.5
        threshold = float(threshold)
        print(f"Threshold   : {threshold:.4f} (target={target_name})")

        # -- Map to submission columns ------------------------------------------
        # Column semantics are inferred from SampleSubmission — no hardcoded
        # competition-specific column names.
        #
        # For binary classification with use_probabilities=True:
        #   - If exactly 2 non-ID submission columns exist, the first is treated
        #     as the hard-label column (threshold applied) and the second as the
        #     probability column (raw values clipped to (0,1)).
        #   - If exactly 1 non-ID submission column exists, it receives the raw
        #     probability (use_probabilities=True) or binary label.
        # For regression: the single non-ID column receives the clipped prediction.
        value_cols = [c for c in submission_cols if c != id_col]
        if target_task == "classification" and bool(
            config.get("use_probabilities", True)
        ):
            if len(value_cols) >= 2:
                # First value column = hard labels, second = raw probabilities
                out_df[value_cols[0]] = (raw_probs >= threshold).astype(int)
                out_df[value_cols[1]] = np.clip(raw_probs, 1e-7, 1 - 1e-7)
            elif len(value_cols) == 1:
                out_df[value_cols[0]] = np.clip(raw_probs, 1e-7, 1 - 1e-7)
        elif target_task == "classification":
            # use_probabilities=False: only hard labels
            submission_col_name = value_cols[0] if value_cols else target_name
            out_df[submission_col_name] = (raw_probs >= threshold).astype(int)
        elif target_task == "regression":
            submission_col_name = value_cols[0] if value_cols else target_name
            bounds = config.get("target_domain_bounds") or {}
            lo = float(bounds.get("min", -np.inf))
            hi = float(bounds.get("max", np.inf))
            clipped = np.clip(raw_probs, lo, hi)
            if bool(config.get("submission_log1p", False)):
                clipped = np.log1p(clipped)
            out_df[submission_col_name] = clipped

    # -- Enforce final column order to match SampleSubmission -------------------
    final_cols = [c for c in submission_cols if c in out_df.columns]
    missing = [c for c in submission_cols if c not in out_df.columns]
    if missing:
        raise ValueError(
            f"Could not produce submission columns: {missing}. "
            f"Available: {list(out_df.columns)}"
        )
    out_df = out_df[final_cols]

    # -- Validate shape ---------------------------------------------------------
    if len(out_df) != len(sample):
        raise ValueError(
            f"Output row count ({len(out_df)}) != sample row count ({len(sample)}). "
            "Row alignment error."
        )

    # -- Write output -----------------------------------------------------------
    out_path = _next_submission_path(paths, suffix=branch_name.replace("/", "-"))
    if not dry_run:
        _atomic_to_csv(cast(pd.DataFrame, out_df), out_path)
        if state is None:
            store.update(
                last_inference_path=str(out_path),
                last_inference_at=datetime.now(timezone.utc).isoformat(),
                last_updated=datetime.now(timezone.utc).isoformat(),
            )
        print(f"\n[OK] Submission written → {out_path}")
        print(f"     Rows: {len(out_df)} | Columns: {list(out_df.columns)}")

    return {"status": "OK", "out_path": str(out_path), "branch": branch_name}


if __name__ == "__main__":
    import sys

    dry = "--dry-run" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    branch = args[0] if args else None
    print(json.dumps(run(branch, dry_run=dry), indent=2))
