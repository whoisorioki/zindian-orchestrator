"""
Skill 01 — Integrity Audit
Computes and locks the MD5 hash of the target column in Training_Data.csv.
Must run after data download and before any data transformation.
Halts if hash shifts on re-run — indicates data tampering or corruption.
"""

import tabula.skill_state_autopatch  # noqa
import hashlib
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

from zindian.paths import resolve_competition_paths
from zindian.config import ChallengeConfig
from zindian.state import SkillStateStore

# Default target column names (can be overridden per-competition)
TARGET_COL = "target"
SUBMISSION_TARGET_COL = "target"


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
            f"\n[FAIL] MD5 MISMATCH DETECTED — {name}\n"
            f"  Locked : {locked}\n"
            f"  Current: {current}\n"
            f"  Data may have been corrupted or tampered with.\n"
            f"  Re-download from data/raw/ and restart from Skill 01."
        )
    return True


def update_skill_state(integrity: dict, state_path: Path) -> None:
    """Persist integrity hashes into SKILL_STATE.json using SkillStateStore.

    Uses safe update semantics and schema validation.
    """
    store = SkillStateStore(state_path)
    state = store.read()
    # Prepare updates dict
    updates = {
        "md5_target_hash": integrity["md5_target_hash"],
        "md5_train_file": integrity["md5_train_file"],
        "md5_test_file": integrity["md5_test_file"],
        "md5_sample_sub_file": integrity["md5_sample_sub_file"],
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }
    # Do not downgrade dag_phase if already beyond phase_1
    current_phase = state.get("dag_phase")
    if current_phase in (None, "uninitialized", "phase_0_foundation"):
        updates["dag_phase"] = "phase_1_complete"
    store.update(**updates)
    print(f"[OK] {state_path} updated with MD5 hashes via SkillStateStore")


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
    print(f"\n{'=' * 60}")
    print("SKILL 01 — Integrity Audit")
    print(f"Mode: {'RE-VERIFY' if re_verify else 'INITIAL LOCK'}")
    print(f"{'=' * 60}\n")

    # Resolve competition-aware paths
    paths = resolve_competition_paths()

    # Load ChallengeConfig to check for input_files override and target cols
    task_type = "classification"
    target_col: str | None = TARGET_COL
    submission_target_col: str | None = SUBMISSION_TARGET_COL
    id_col = "ID"
    domain = ""
    input_files: dict[str, str] = {}

    try:
        cfg = ChallengeConfig.load()
        tc = cfg.get("target_column") or cfg.get("target_col")
        sc = cfg.get("submission_target_column") or cfg.get("submission_target_col")
        ic = cfg.get("id_column") or cfg.get("id_col")
        if tc:
            target_col = tc
        if sc:
            submission_target_col = sc
        else:
            submission_target_col = target_col
        if ic:
            id_col = ic
        task_type = cfg.get("task_type") or "classification"
        domain = cfg.get("domain", "")
        input_files = cfg.get("input_files", {}) or {}
    except Exception:
        pass

    train_file = input_files.get("train", "Training_Data.csv")
    test_file = input_files.get("test", "Test.csv")
    sample_file = input_files.get("sample", "SampleSubmission.csv")

    train_path = paths.data_raw_dir / train_file
    test_path = paths.data_raw_dir / test_file
    sample_path = paths.data_raw_dir / sample_file
    state_path = paths.state_path

    print(f"Loading {train_file} from: {train_path}")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sub = pd.read_csv(sample_path)

    print(f"  Train shape : {train.shape}")
    print(f"  Test shape  : {test.shape}")
    print(f"  Sample sub  : {sub.shape}")

    # Check if we're in INIT mode (no config exists yet)
    config_exists = paths.config_path.exists()

    # INIT mode: Skip target validation - skill_02 will detect multi-target structure
    if not config_exists:
        print("[INFO]  INIT mode detected - skipping target column validation")
        print("   skill_02 will detect target structure from SampleSubmission.csv")
        target_col = None  # Signal to skip target-dependent operations
    else:
        # ENFORCE mode: Validate expected columns
        if target_col not in train.columns:
            # Check if this is a multi-target competition
            try:
                target_config = cfg.get("target_config")
                if target_config:
                    print(
                        "[INFO]  Multi-target competition detected - skipping single target validation"
                    )
                    target_col = None  # Skip single-target operations
                else:
                    raise AssertionError(
                        f"[FAIL] Target column '{target_col}' not found in {train_file}"
                    )
            except Exception:
                raise AssertionError(
                    f"[FAIL] Target column '{target_col}' not found in {train_file}"
                )

        if id_col not in train.columns:
            raise AssertionError(f"[FAIL] ID column '{id_col}' missing from train")

        # Latitude/Longitude check only makes sense for geospatial challenges
        if domain.lower() == "geospatial":
            if "Latitude" not in train.columns or "Longitude" not in train.columns:
                print(
                    "[WARN] Latitude/Longitude columns missing — continuing (not mandatory)"
                )

        if submission_target_col not in sub.columns:
            raise AssertionError(
                f"[FAIL] '{submission_target_col}' not found in SampleSubmission.csv"
            )
        print("[OK] Required columns present (warnings may have been emitted)")

    # Validate target values (skip in INIT mode)
    if target_col is None:
        print("[INFO]  Skipping target validation in INIT mode")
        counts: dict = {}
    elif task_type == "regression":
        # Continuous descriptive metrics

        t_arr = train[target_col].dropna().to_numpy()
        t_min = float(t_arr.min()) if t_arr.size else 0.0
        t_max = float(t_arr.max()) if t_arr.size else 0.0
        t_mean = float(t_arr.mean()) if t_arr.size else 0.0
        t_std = float(t_arr.std()) if t_arr.size else 0.0

        print("\n  Continuous target statistics:")
        print(f"    Min  : {t_min:.5f}")
        print(f"    Max  : {t_max:.5f}")
        print(f"    Mean : {t_mean:.5f}")
        print(f"    Std  : {t_std:.5f}")

        counts = {}
    else:
        # Validate target values; be permissive but warn if non-binary
        unique_targets = sorted(pd.unique(train[target_col].astype(str)).tolist())
        is_binary_numeric = set(train[target_col].dropna().unique()) <= {0, 1}
        if not is_binary_numeric:
            print(
                f"[WARN] Target values are not strictly 0/1: {unique_targets} — will hash canonical string form"
            )
        else:
            print(f"[OK] Target values confirmed numeric binary: {[0, 1]}")

        # Print class distribution
        counts = train[target_col].value_counts().to_dict()
        total = len(train)
        print("\n  Class distribution:")
        print(
            f"    Absent  (0): {counts.get(0, 0):,} ({counts.get(0, 0) / total * 100:.1f}%)"
        )
        print(
            f"    Present (1): {counts.get(1, 0):,} ({counts.get(1, 0) / total * 100:.1f}%)"
        )

    # Compute hashes
    print("\nComputing MD5 hashes...")
    # In INIT mode, skip target column hash (will be computed by skill_02)
    if target_col is not None:
        md5_target = compute_md5(train[target_col])
    else:
        md5_target = "pending_skill_02"  # Placeholder for INIT mode

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
        store = SkillStateStore(state_path)
        state = store.read()

        print("\nVerifying against locked hashes...")
        locked = state.get("md5_target_hash")
        if not locked:
            raise RuntimeError(
                "No locked md5_target_hash found in SKILL_STATE.json for verification"
            )
        verify_hash(md5_target, locked, "target column")

        locked = state.get("md5_train_file")
        if not locked:
            raise RuntimeError(
                "No locked md5_train_file found in SKILL_STATE.json for verification"
            )
        verify_hash(md5_train_file, locked, "train file")

        locked = state.get("md5_test_file")
        if not locked:
            raise RuntimeError(
                "No locked md5_test_file found in SKILL_STATE.json for verification"
            )
        verify_hash(md5_test_file, locked, "test file")

        locked = state.get("md5_sample_sub_file")
        if not locked:
            raise RuntimeError(
                "No locked md5_sample_sub_file found in SKILL_STATE.json for verification"
            )
        verify_hash(md5_sample_sub, locked, "sample submission")

        print("[OK] All hashes match — data integrity confirmed")
    else:
        # First run — lock the hashes (do not downgrade dag_phase if beyond)
        update_skill_state(integrity, state_path)
        print("\n[OK] Hashes locked in SKILL_STATE.json")

    # Derive feature columns (exclude ID and target)
    if target_col is not None:
        raw_feature_cols = [c for c in train.columns if c not in {id_col, target_col}]
    else:
        # INIT mode: Just exclude ID column
        raw_feature_cols = [c for c in train.columns if c != id_col]
    summary = {
        "status": "OK",
        "train_rows": len(train),
        "test_rows": len(test),
        "n_features_raw": len(raw_feature_cols),
        "target_col": target_col if target_col else "pending_skill_02",
        "submission_target_col": (
            submission_target_col if target_col else "pending_skill_02"
        ),
        "task": "regression" if task_type == "regression" else "binary_classification",
        "class_distribution": counts,
        "feature_cols": raw_feature_cols,
        "note": "Do not assume raw features beyond what is present in training file",
        **integrity,
    }

    print("\n--- Dataset Summary ---")
    print(f"Task         : {summary['task']}")
    print(f"Train rows   : {summary['train_rows']:,}")
    print(f"Test rows    : {summary['test_rows']:,}")
    print(f"Raw features : {summary['feature_cols']}")
    print(f"Note         : {summary['note']}")

    return summary
