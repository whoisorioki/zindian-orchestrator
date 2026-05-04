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


# Default target column names (can be overridden per-competition)
TARGET_COL = "Occurrence Status"
SUBMISSION_TARGET_COL = "Target"

def compute_md5(series: pd.Series) -> str:
    """Compute MD5 hash of a pandas Series values."""
    # Use pandas stable hashing to support extension/categorical dtypes.
    hashed = hash_pandas_object(series, index=False).to_numpy(dtype="uint64").tobytes()
    return hashlib.md5(hashed).hexdigest()


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

    print(f"  Train shape : {train.shape}")
    print(f"  Test shape  : {test.shape}")
    print(f"  Sample sub  : {sub.shape}")

    # Validate expected columns exist
    assert TARGET_COL in train.columns, \
        f"❌ Target column '{TARGET_COL}' not found in Training_Data.csv"
    assert "ID" in train.columns, "❌ ID column missing from train"
    assert "Latitude" in train.columns, "❌ Latitude missing from train"
    assert "Longitude" in train.columns, "❌ Longitude missing from train"
    assert SUBMISSION_TARGET_COL in sub.columns, \
        f"❌ '{SUBMISSION_TARGET_COL}' not found in SampleSubmission.csv"
    print("✅ All expected columns present")

    # Validate target values
    unique_targets = sorted(train[TARGET_COL].unique().tolist())
    assert unique_targets == [0, 1], \
        f"❌ Unexpected target values: {unique_targets} (expected [0, 1])"
    print(f"✅ Target values confirmed: {unique_targets}")

    # Print class distribution
    counts = train[TARGET_COL].value_counts().to_dict()
    total = len(train)
    print(f"\n  Class distribution:")
    print(f"    Absent  (0): {counts.get(0, 0):,} ({counts.get(0,0)/total*100:.1f}%)")
    print(f"    Present (1): {counts.get(1, 0):,} ({counts.get(1,0)/total*100:.1f}%)")

    # Compute hashes
    print("\nComputing MD5 hashes...")
    md5_target = compute_md5(train[TARGET_COL])
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
        # First run — lock the hashes
        update_skill_state(integrity, state_path)
        print("\n✅ Hashes locked in SKILL_STATE.json")

    # Dataset summary for agent
    summary = {
        "status": "OK",
        "train_rows": len(train),
        "test_rows": len(test),
        "n_features_raw": len(train.columns) - 2,  # exclude ID and target
        "target_col": TARGET_COL,
        "submission_target_col": SUBMISSION_TARGET_COL,
        "task": "binary_classification",
        "class_distribution": counts,
        "feature_cols": ["Latitude", "Longitude"],
        "note": "Raw features are only Lat/Lon — climate enrichment needed via TerraClimate",
        **integrity
    }

    print(f"\n--- Dataset Summary ---")
    print(f"Task         : {summary['task']}")
    print(f"Train rows   : {summary['train_rows']:,}")
    print(f"Test rows    : {summary['test_rows']:,}")
    print(f"Raw features : {summary['feature_cols']}")
    print(f"Note         : {summary['note']}")

    return summary
