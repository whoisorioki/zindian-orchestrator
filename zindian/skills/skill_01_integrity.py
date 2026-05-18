"""
Skill 01 — Integrity Audit
Computes and locks the MD5 hash of the target column in Training_Data.csv.
Must run after data download and before any data transformation.
Halts if hash shifts on re-run — indicates data tampering or corruption.
"""

import os
import json
import hashlib
import tempfile
import pandas as pd
from pandas.util import hash_pandas_object
from pathlib import Path
from datetime import datetime, timezone

from zindian.paths import resolve_competition_paths
from zindian.config import ChallengeConfig


# Default target column names (can be overridden per-competition)
TARGET_COL = "Occurrence Status"
SUBMISSION_TARGET_COL = "Target"

def compute_md5(series: pd.Series) -> str:
    """Compute MD5 hash of a pandas Series values.

    Use a stable string-concatenation method (canonical across scripts):
    join stringified values with commas, encode, then MD5. This matches
    the project-side fix instruction and avoids cross-method mismatches.
    """
    as_bytes = series.astype(str).str.cat(sep=",").encode("utf-8")
    return hashlib.md5(as_bytes).hexdigest()


def compute_file_md5(filepath: str) -> str:
    """Compute MD5 hash of an entire file."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_hash(current: str, locked: str, name: str) -> bool:
    """Compare current hash to locked hash. Halt if mismatch."""
    if current != locked:
        raise RuntimeError(
            f"\n❌ MD5 MISMATCH DETECTED — {name}\n"
            f"  Locked : {locked}\n"
            f"  Current: {current}\n"
            f"  Data may have been corrupted or tampered with.\n"
            f"  Re-download from data/raw/ and restart from Skill 01."
        )
    return True


def update_skill_state(integrity: dict, state_path: Path) -> None:
    if not state_path.exists():
        raise FileNotFoundError(f"SKILL_STATE.json not found at {state_path}")
    with state_path.open("r", encoding="utf-8") as f:
        state = json.load(f)

    state["md5_target_hash"] = integrity["md5_target_hash"]
    state["md5_train_file"] = integrity["md5_train_file"]
    state["md5_test_file"] = integrity["md5_test_file"]
    state["md5_sample_sub_file"] = integrity["md5_sample_sub_file"]
    # Do not downgrade dag_phase if already beyond phase_1
    current_phase = state.get("dag_phase")
    if current_phase in (None, "uninitialized", "phase_0_foundation"):
        state["dag_phase"] = "phase_1_complete"
    state["last_updated"] = datetime.now(timezone.utc).isoformat()

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json", dir=str(state_path.parent)) as tmp:
        json.dump(state, tmp, indent=2)
        tmp_path = tmp.name
    os.replace(tmp_path, state_path)
    print(f"✅ {state_path} updated with MD5 hashes")


def run(re_verify: bool = False) -> dict:
    """
    Main entry point.

    First run (re_verify=False):
      - Loads Training_Data.csv
      - Computes MD5 of target column and all raw files
      - Locks hashes in SKILL_STATE.json

    Subsequent runs (re_verify=True):
      - Recomputes hashes
      - Compares to locked values
      - Halts if any mismatch detected
    """
    print(f"\n{'='*60}")
    print(f"SKILL 01 — Integrity Audit")
    print(f"Mode: {'RE-VERIFY' if re_verify else 'INITIAL LOCK'}")
    print(f"{'='*60}\n")

    # Resolve competition-aware paths
    paths = resolve_competition_paths()
    train_path = paths.data_raw_dir / "Training_Data.csv"
    test_path = paths.data_raw_dir / "Test.csv"
    sample_path = paths.data_raw_dir / "SampleSubmission.csv"
    state_path = paths.state_path
    print(f"Loading Training_Data.csv from: {train_path}")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sub = pd.read_csv(sample_path)

    # Load ChallengeConfig to override default column names if present
    try:
        cfg = ChallengeConfig.load()
        tc = cfg.get("target_column") or cfg.get("target_col")
        sc = cfg.get("submission_target_column") or cfg.get("submission_target_col")
        target_col = tc if tc else TARGET_COL
        submission_target_col = sc if sc else SUBMISSION_TARGET_COL
    except Exception:
        target_col = TARGET_COL
        submission_target_col = SUBMISSION_TARGET_COL

    print(f"  Train shape : {train.shape}")
    print(f"  Test shape  : {test.shape}")
    print(f"  Sample sub  : {sub.shape}")

    # Validate expected columns exist — be permissive for generality
    if target_col not in train.columns:
        raise AssertionError(f"❌ Target column '{target_col}' not found in Training_Data.csv")
    if "ID" not in train.columns:
        raise AssertionError("❌ ID column missing from train")
    # Latitude/Longitude are helpful but not mandatory across competitions
    if "Latitude" not in train.columns or "Longitude" not in train.columns:
        print("⚠️ Latitude/Longitude columns missing — continuing (not mandatory)")
    if submission_target_col not in sub.columns:
        raise AssertionError(f"❌ '{submission_target_col}' not found in SampleSubmission.csv")
    print("✅ Required columns present (warnings may have been emitted)")

    # Validate target values
    # Validate target values; be permissive but warn if non-binary
    unique_targets = sorted(pd.unique(train[target_col].astype(str)).tolist())
    is_binary_numeric = set(train[target_col].dropna().unique()) <= {0, 1}
    if not is_binary_numeric:
        print(f"⚠️ Target values are not strictly 0/1: {unique_targets} — will hash canonical string form")
    else:
        print(f"✅ Target values confirmed numeric binary: {[0,1]}")

    # Print class distribution
    counts = train[TARGET_COL].value_counts().to_dict()
    total = len(train)
    print(f"\n  Class distribution:")
    print(f"    Absent  (0): {counts.get(0, 0):,} ({counts.get(0,0)/total*100:.1f}%)")
    print(f"    Present (1): {counts.get(1, 0):,} ({counts.get(1,0)/total*100:.1f}%)")

    # Compute hashes
    print("\nComputing MD5 hashes...")
    md5_target = compute_md5(train[target_col])
    md5_train_file = compute_file_md5(str(train_path))
    md5_test_file = compute_file_md5(str(test_path))
    md5_sample_sub = compute_file_md5(str(sample_path))

    print(f"  Target column MD5 : {md5_target}")
    print(f"  Train file MD5    : {md5_train_file}")
    print(f"  Test file MD5     : {md5_test_file}")
    print(f"  Sample sub MD5    : {md5_sample_sub}")

    integrity = {
        "md5_target_hash": md5_target,
        "md5_train_file": md5_train_file,
        "md5_test_file": md5_test_file,
        "md5_sample_sub_file": md5_sample_sub,
    }

    # If re-verifying, compare against locked hashes
    if re_verify:
        with state_path.open("r", encoding="utf-8") as f:
            state = json.load(f)

        print("\nVerifying against locked hashes...")
        verify_hash(md5_target, state.get("md5_target_hash"), "target column")
        verify_hash(md5_train_file, state.get("md5_train_file"), "train file")
        verify_hash(md5_test_file, state.get("md5_test_file"), "test file")
        verify_hash(md5_sample_sub, state.get("md5_sample_sub_file"), "sample submission")
        print("✅ All hashes match — data integrity confirmed")
    else:
        # First run — lock the hashes (do not downgrade dag_phase if beyond)
        update_skill_state(integrity, state_path)
        print("\n✅ Hashes locked in SKILL_STATE.json")

    # Derive feature columns (exclude ID and target)
    raw_feature_cols = [c for c in train.columns if c not in {"ID", target_col}]
    summary = {
        "status": "OK",
        "train_rows": len(train),
        "test_rows": len(test),
        "n_features_raw": len(raw_feature_cols),
        "target_col": target_col,
        "submission_target_col": submission_target_col,
        "task": "binary_classification",
        "class_distribution": counts,
        "feature_cols": raw_feature_cols,
        "note": "Do not assume raw features beyond what is present in training file",
        **integrity
    }

    print(f"\n--- Dataset Summary ---")
    print(f"Task         : {summary['task']}")
    print(f"Train rows   : {summary['train_rows']:,}")
    print(f"Test rows    : {summary['test_rows']:,}")
    print(f"Raw features : {summary['feature_cols']}")
    print(f"Note         : {summary['note']}")

    return summary
