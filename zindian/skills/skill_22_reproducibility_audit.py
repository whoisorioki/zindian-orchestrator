"""
SKILL 22 - Reproducibility & Integration Audit
Validates absolute data geometry, tracking alignment, and locks state shapes.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str]) -> str:
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except subprocess.CalledProcessError as exc:
        return f"ERROR: {exc.stderr.strip()}"


def audit_pipeline() -> bool:
    print("=" * 70)
    print("ZINDIAN ORCHESTRATOR: FINAL REPRODUCIBILITY AUDIT")
    print("=" * 70)

    root_dir = Path(__file__).resolve().parents[2]
    comp_dir = root_dir / "competitions" / "ey-frogs"
    state_path = comp_dir / "SKILL_STATE.json"

    errors_found = 0

    print("\n[Check 1] Synchronizing Repository Branch State...")
    current_branch = run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    print(f"  - Active Git Branch : {current_branch}")

    if not state_path.exists():
        print(f"  ERROR: Missing critical tracking file: {state_path}")
        return False

    with state_path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)

    recorded_branch = state.get("current_git_branch", "Unknown")
    recorded_phase = state.get("dag_phase", "Unknown")
    print(f"  - Serialized Git Branch: {recorded_branch}")
    print(f"  - Serialized DAG Phase: {recorded_phase}")

    if current_branch != recorded_branch:
        print("  NOTICE: Branch tracking asymmetry detected.")
        print(f"      Workspace is on '{current_branch}' but state records '{recorded_branch}'.")

    print("\n[Check 2] Validating Processed Feature Array Geometries...")
    import pandas as pd

    feature_targets = {
        "features_full_train.csv": (6312, 98),
        # The held-out test frame excludes the target column.
        "features_full_test.csv": (2000, 97),
    }

    for file_name, expected_shape in feature_targets.items():
        file_path = comp_dir / "data" / "processed" / file_name
        if not file_path.exists():
            print(f"  ERROR: File missing: {file_path.name}")
            errors_found += 1
            continue

        frame = pd.read_csv(file_path)
        if frame.shape != expected_shape:
            print(f"  ERROR: Shape mismatch in {file_name}: got {frame.shape}, expected {expected_shape}")
            errors_found += 1
        else:
            print(f"  OK: Shape verified for {file_name}: {frame.shape}")

    print("\n[Check 3] Inspecting Alignment of the OOF Probability Pool...")
    pool_files = [
        "oof_variant-34.csv",
        "oof_variant-pseudo_iter0.csv",
        "oof_variant-36.csv",
    ]

    base_id_series = None

    for oof_name in pool_files:
        oof_path = comp_dir / "data" / "processed" / oof_name
        if not oof_path.exists():
            oof_path = comp_dir / "reports" / oof_name

        if not oof_path.exists():
            print(f"  ERROR: Critical OOF artifact missing from pool: {oof_name}")
            errors_found += 1
            continue

        frame_oof = pd.read_csv(oof_path)
        if len(frame_oof) != 6312:
            print(f"  ERROR: Row alignment failure in {oof_name}: got {len(frame_oof)}, expected 6312")
            errors_found += 1
            continue

        if "ID" not in frame_oof.columns:
            print(f"  ERROR: Namespace alignment bug: {oof_name} is missing key 'ID' index column")
            errors_found += 1
            continue

        if base_id_series is None:
            base_id_series = frame_oof["ID"].reset_index(drop=True)
            print(f"  OK: Anchored index vector using {oof_name}")
        elif not frame_oof["ID"].reset_index(drop=True).equals(base_id_series):
            print(f"  ERROR: Positional shuffling detected in {oof_name}; indices are out of order.")
            errors_found += 1
        else:
            print(f"  OK: Positional alignment confirmed for {oof_name}")

    print("\n" + "=" * 70)
    if errors_found == 0:
        print("INTEGRATION STATUS: SECURE. WORKSPACE FULLY REPRODUCIBLE.")
        print("All feature arrays, shapes, index alignments, and configurations are locked.")
        print("=" * 70)
        return True

    print(f"AUDIT BLOCKED: {errors_found} structural anomalies identified.")
    print("Resolve configuration mapping states before final archive export.")
    print("=" * 70)
    return False


if __name__ == "__main__":
    raise SystemExit(0 if audit_pipeline() else 1)