#!/usr/bin/env python3
"""SOT Alignment Check: Automates verification of docs/source_of_truth.md statuses against codebase reality."""

import re
from typing import Any
import sys
import argparse
from pathlib import Path

# Paths
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
SOT_PATH = WORKSPACE_DIR / "docs/source_of_truth.md"
ZINDIAN_DIR = WORKSPACE_DIR / "zindian"

# Definitions of code checks for each S-item.
# Each item has a list of check tuples: (relative_file_path, search_pattern, must_exist)
# If must_exist is True, the pattern must exist in the file.
# If must_exist is False, the pattern must NOT exist in the file.
# Special check: if the file itself does not exist, the check fails (unless must_exist is False and file is missing).
SOT_CHECKS: dict[str, dict[str, Any]] = {
    "1": {
        "name": "S1/S9 — Bessel's Correction & Absolute Promotion Margins",
        "checks": [
            ("zindian/skills/skill_12_metric.py", "fold_score_variance_nb", True),
            ("zindian/skills/skill_12_metric.py", "se_oof", True),
            ("zindian/skills/skill_11_gate.py", "se_oof", True),
            (
                "zindian/skills/skill_11_gate.py",
                "effective_margin = max(effective_margin, 1.0 * se_oof)",
                True,
            ),
        ],
    },
    "2": {
        "name": "S2 — MAPE Zero-Target Bias & MASE",
        "checks": [
            ("zindian/state.py", "zero_fraction", True),
            ("zindian/state.py", "mase", True),
            ("zindian/skills/skill_08_anchor.py", "mae_naive_baseline", True),
        ],
    },
    "3": {
        "name": "S3 — Non-Uniform Metric Scaling / Composite Weighting",
        "checks": [
            (
                "zindian/skills/skill_12_metric.py",
                "use_inverse_variance_weighting",
                True,
            ),
            ("zindian/skills/skill_11_gate.py", "_effective_target_weight", True),
        ],
    },
    "4": {
        "name": "S4 — Correlation-Based Pruning",
        "checks": [
            ("zindian/oracle_fusion_core.py", "y_true=y_true", True),
        ],
    },
    "5": {
        "name": "S5 — Target Covariance Breakdown",
        "checks": [],  # Deferred, no active checks needed
    },
    "6": {
        "name": "S6 — Multicollinear Leakage Splitting / Systematic MI Audit",
        "checks": [
            ("zindian/skills/skill_10_shap.py", "leakage_mi_advisory", True),
            ("zindian/skills/skill_11_gate.py", "leakage_mi_advisory", True),
            ("zindian/skills/skill_10_shap.py", "mi_max_samples", True),
        ],
    },
    "7": {
        "name": "S7 — Spatial Autocorrelation Bias",
        "checks": [
            ("zindian/skills/skill_05_cv.py", "spatial_buffer_km", True),
            ("zindian/skills/skill_05_cv.py", "_apply_spatial_buffer", True),
            ("zindian/skills/skill_05_cv.py", "build_spatial_splits", True),
        ],
    },
    "8": {
        "name": "S8 — Fixed Pseudo-label Thresholding & Adaptive Quantiles",
        "checks": [
            # S8 is "Decision recorded" — the class-wise quantile mechanism is
            # specified but NOT yet implemented. skill_21 still uses fixed
            # absolute thresholds (CONF_POS_DEFAULT=0.85, CONF_NEG_DEFAULT=0.15).
            # The check below verifies the current fixed-threshold implementation
            # exists, NOT the future quantile mechanism.
            ("zindian/skills/skill_21_pseudo_label.py", "CONF_POS_DEFAULT", True),
            ("zindian/skills/skill_21_pseudo_label.py", "CONF_NEG_DEFAULT", True),
        ],
    },
    "10": {
        "name": "S10 — Floating-Point Integrity limits",
        "checks": [
            (
                "zindian/skills/skill_22_reproducibility_audit.py",
                "derived_artifact_fingerprints",
                True,
            ),
            (
                "zindian/skills/skill_22_reproducibility_audit.py",
                "_audit_derived_artifact_fingerprints",
                True,
            ),
        ],
    },
}


def parse_sot_statuses(sot_content: str) -> dict:
    """Parses Section 9 of the SoT to determine the declared status of S1-S10."""
    statuses = {}

    # Locate Section 9
    sec9_match = re.search(r"## 9\. Known Gaps Registry.*", sot_content, re.DOTALL)
    if not sec9_match:
        print(
            "[ERROR] Could not locate '## 9. Known Gaps Registry' in source_of_truth.md"
        )
        sys.exit(1)

    sec9_text = sec9_match.group(0)

    # Find S-item bullet points and their status lines
    # Pattern matches: - **S<num> ...**: followed by *Status:* <status_text>
    # Handles combined entries like "S1 — ... & S9 — ..." by extracting ALL
    # S-numbers from the bold text and assigning the same status to each.
    entry_pattern = r"-\s*\*\*((?:S\d+[^*]*?)+)\*\*:\s*\n\s*\*Status:\*\s*([^\n]+)"
    entries = re.findall(entry_pattern, sec9_text)

    for bold_text, status_line in entries:
        # Extract all S-numbers from the bold text (e.g., "S1" and "S9" from
        # "S1 — Bessel's Correction Underestimation & S9 — Absolute Promotion Margins")
        s_nums = re.findall(r"S(\d+)", bold_text)
        status_line_clean = status_line.strip()

        if "[Pending]" in status_line_clean:
            status = "Pending"
        elif "[Deferred]" in status_line_clean:
            status = "Deferred"
        else:
            status = "Implemented"

        for num in s_nums:
            statuses[num] = {"status": status, "line": status_line_clean}

    return statuses


def run_code_check(file_rel_path: str, pattern: str, must_exist: bool) -> bool:
    """Runs a single code check against the workspace."""
    file_path = WORKSPACE_DIR / file_rel_path
    if not file_path.exists():
        return (
            not must_exist
        )  # If it shouldn't exist, missing file is a pass. Otherwise a fail.

    try:
        content = file_path.read_text(encoding="utf-8")
        found = pattern in content
        return found if must_exist else not found
    except Exception as e:
        print(f"[ERROR] Failed to read {file_rel_path}: {e}")
        return False


def check_claim_code_coupling(sot_statuses: dict) -> list[str]:
    """Scans all Python files for claim-code coupling comments and verifies SoT matches them."""
    errors = []
    coupling_pattern = re.compile(r"#\s*(S\d+)\s*-\s*implemented\s*([\d-]+)")

    for py_file in WORKSPACE_DIR.glob("zindian/**/*.py"):
        if ".venv" in py_file.parts or "tests" in py_file.parts:
            continue
        try:
            content = py_file.read_text(encoding="utf-8")
            for match in coupling_pattern.finditer(content):
                s_num = match.group(1)[1:]  # e.g., "6" from "S6"
                date_str = match.group(2)

                # Verify in SoT
                if s_num not in sot_statuses:
                    errors.append(
                        f"Code comment in {py_file.relative_to(WORKSPACE_DIR)} claims {match.group(1)} "
                        f"implemented on {date_str}, but no status for S{s_num} found in SoT Section 9."
                    )
                else:
                    sot_status_info = sot_statuses[s_num]
                    if sot_status_info["status"] != "Implemented":
                        errors.append(
                            f"Code comment in {py_file.relative_to(WORKSPACE_DIR)} claims {match.group(1)} "
                            f"implemented on {date_str}, but SoT Section 9 claims it is {sot_status_info['status']}."
                        )
                    elif date_str not in sot_status_info["line"]:
                        errors.append(
                            f"Code comment in {py_file.relative_to(WORKSPACE_DIR)} claims {match.group(1)} "
                            f"implemented on {date_str}, but the SoT Section 9 status line for S{s_num} "
                            f"does not reference this date/commit."
                        )
        except Exception as e:
            print(f"[ERROR] Failed to scan {py_file} for coupling: {e}")

    return errors


def main():
    parser = argparse.ArgumentParser(description="Check SoT alignment with codebase.")
    parser.add_argument(
        "--fail-on-misaligned",
        action="store_true",
        help="Exit with 1 on MISALIGNED items.",
    )
    parser.add_argument(
        "--fail-on-code-ahead",
        action="store_true",
        help="Exit with 1 on CODE_AHEAD items.",
    )
    args = parser.parse_args()

    if not SOT_PATH.exists():
        print(f"[ERROR] source_of_truth.md not found at {SOT_PATH}")
        sys.exit(1)

    sot_content = SOT_PATH.read_text(encoding="utf-8")
    sot_statuses = parse_sot_statuses(sot_content)

    align_count = 0
    misaligned_count = 0
    code_ahead_count = 0

    print("=" * 80)
    # Highlight title with standard format
    print("ZINDIAN ORCHESTRATOR SOT-CODE ALIGNMENT AUDIT")
    print("=" * 80)

    for s_num, item_info in SOT_CHECKS.items():
        name = item_info["name"]
        checks = item_info["checks"]

        # Get SoT status
        sot_info = sot_statuses.get(s_num)
        if not sot_info:
            print(f"[WARNING] S{s_num} not found in SoT Section 9.")
            continue

        sot_status = sot_info["status"]

        # Evaluate code checks
        all_passed = True
        failed_checks = []
        for file_rel, pattern, must_exist in checks:
            passed = run_code_check(file_rel, pattern, must_exist)
            if not passed:
                all_passed = False
                failed_checks.append((file_rel, pattern, must_exist))

        # Determine alignment
        if sot_status in ("Pending", "Deferred"):
            if all_passed and len(checks) > 0:
                print(f"[CODE_AHEAD] S{s_num} — {name}")
                print(f"  SoT claims: {sot_status}")
                print("  Code:       All implementation patterns are present.")
                code_ahead_count += 1
            else:
                print(f"[ALIGN]      S{s_num} — {name} (Sync: {sot_status})")
                align_count += 1
        elif sot_status == "Implemented":
            if all_passed:
                print(f"[ALIGN]      S{s_num} — {name} (Sync: Implemented)")
                align_count += 1
            else:
                print(f"[MISALIGNED] S{s_num} — {name}")
                print("  SoT claims: Implemented")
                print("  Code:       Missing implementation patterns:")
                for file_rel, pattern, must_exist in failed_checks:
                    mode = "must contain" if must_exist else "must NOT contain"
                    print(f"    - {file_rel}: {mode} '{pattern}'")
                misaligned_count += 1

    # Run claim-code coupling check
    print("-" * 80)
    print("CLAIM-CODE COUPLING AUDIT")
    print("-" * 80)
    coupling_errors = check_claim_code_coupling(sot_statuses)
    if coupling_errors:
        for err in coupling_errors:
            print(f"[MISALIGNED] {err}")
            misaligned_count += 1
    else:
        print("[ALIGN]      No claim-code coupling errors found.")

    # Print Summary
    print("=" * 80)
    print("AUDIT SUMMARY")
    print("=" * 80)
    print(f"Aligned:      {align_count}")
    print(f"Misaligned:   {misaligned_count}")
    print(f"Code Ahead:   {code_ahead_count}")
    print("=" * 80)

    # Exit codes
    if misaligned_count > 0 and args.fail_on_misaligned:
        sys.exit(1)
    if code_ahead_count > 0 and args.fail_on_code_ahead:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
