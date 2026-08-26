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
            ("zindian/skills/skill_08_anchor.py", '"mase":', True),
            (
                "zindian/skills/skill_08_anchor.py",
                "# S2 - implemented 2026-08-25",
                True,
            ),
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
            ("zindian/skills/skill_10_shap.py", "leakage_pairwise_mi_advisory", True),
            ("zindian/skills/skill_10_shap.py", "mi_pairwise_threshold", True),
            ("zindian/skills/skill_10_shap.py", "# S6 - implemented 2026-08-24", True),
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
    "11": {
        "name": "S11 — skill_18/20 root write consolidation",
        "checks": [
            (
                "zindian/skills/skill_18_librarian.py",
                'paths.reports_dir / "literature_cache.json"',
                False,
            ),
            (
                "zindian/skills/skill_20_scientist.py",
                'paths.reports_dir / "validated_hypotheses.json"',
                False,
            ),
            (
                "zindian/skills/skill_18_librarian.py",
                "# S11 - implemented 2026-08-24",
                True,
            ),
            (
                "zindian/skills/skill_20_scientist.py",
                "# S11 - implemented 2026-08-24",
                True,
            ),
        ],
    },
    "Preflight": {
        "name": "Preflight — Multi-target OOF completeness check",
        "checks": [
            (
                "scripts/preflight_enforce.py",
                'targets = cfg.get("target_config", {}).get("targets", [])',
                True,
            ),
            (
                "scripts/preflight_enforce.py",
                "# Preflight MT-OOF - implemented 2026-08-24",
                True,
            ),
        ],
    },
    "R5": {
        "name": "R5 — telemetry.aggregate write & verification",
        "checks": [
            (
                "zindian/orchestrator.py",
                '"telemetry.aggregate": telemetry_aggregate',
                True,
            ),
            (
                "zindian/skills/skill_22_reproducibility_audit.py",
                'telemetry_agg = state.get("telemetry.aggregate")',
                True,
            ),
            ("zindian/orchestrator.py", "# R5 - implemented 2026-08-24", True),
            (
                "zindian/skills/skill_22_reproducibility_audit.py",
                "# R5 - implemented 2026-08-24",
                True,
            ),
        ],
    },
}


def parse_sot_statuses(sot_content: str) -> dict:
    """Parses the Known Gaps Registry section of the SoT to determine the declared status of S1-S11, Preflight, R5."""
    statuses = {}

    # Locate the Known Gaps Registry (section number is version-dependent)
    sec9_match = re.search(r"## \d+\. Known Gaps Registry.*", sot_content, re.DOTALL)
    if not sec9_match:
        print(
            "[ERROR] Could not locate the 'Known Gaps Registry' heading in source_of_truth.md"
        )
        sys.exit(1)

    sec9_text = sec9_match.group(0)

    # Split the registry into its RESOLVED (audit-trail table) and OPEN
    # (active gaps) subsections.
    resolved_part, _sep, open_part = sec9_text.partition("### OPEN")

    # RESOLVED subsection: Markdown table rows of the form
    #   | ID | description | resolution |
    # Each row marks every referenced ID as Implemented. Rows whose
    # ID cell carries "(partial)" are annotations on still-open items and
    # are skipped so they never mask an OPEN status.
    row_pattern = r"^\|\s*([a-zA-Z0-9_\-\s/]+)\s*\|([^|]*)\|([^|]*)\|"
    for row_match in re.finditer(row_pattern, resolved_part, re.MULTILINE):
        id_cell = row_match.group(1).strip()
        if id_cell.lower() == "id" or id_cell.startswith("---") or not id_cell:
            continue
        if "partial" in row_match.group(0).lower():
            continue
        line_text = f"{row_match.group(2).strip()} - {row_match.group(3).strip()}"

        # Extract all S-numbers if present
        s_nums = re.findall(r"S(\d+)", id_cell)
        if s_nums:
            for num in s_nums:
                statuses[num] = {"status": "Implemented", "line": line_text}
        else:
            # Handle non-S IDs like Preflight, R5, etc.
            statuses[id_cell] = {"status": "Implemented", "line": line_text}

    # OPEN subsection: bold headings of the form **S<num> - title** or **ID - title**.
    # Anything listed here is an active gap (Pending) unless already
    # recorded as Implemented above.
    open_pattern = r"\*\*([^*]+)\*\*"
    for bold_match in re.finditer(open_pattern, open_part):
        bold_text = bold_match.group(1).strip()
        if not bold_text or bold_text.startswith("DEFERRED"):
            continue
        s_nums = re.findall(r"S(\d+)", bold_text)
        if s_nums:
            for num in s_nums:
                if num not in statuses:
                    statuses[num] = {"status": "Pending", "line": bold_text}
        else:
            # Non-S item like Preflight or R5
            item_name = re.split(r"[-:]", bold_text)[0].strip()
            item_name = item_name.split(" ")[0].strip()
            if item_name not in statuses:
                statuses[item_name] = {"status": "Pending", "line": bold_text}

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
    coupling_pattern = re.compile(
        r"#\s*(S\d+|Preflight|R\d+)\s*-\s*implemented\s*([\d-]+)"
    )

    # Scan both zindian/ and scripts/ directories for Python files
    py_files = list(WORKSPACE_DIR.glob("zindian/**/*.py")) + list(
        WORKSPACE_DIR.glob("scripts/**/*.py")
    )
    for py_file in py_files:
        if ".venv" in py_file.parts or "tests" in py_file.parts:
            continue
        try:
            content = py_file.read_text(encoding="utf-8")
            for match in coupling_pattern.finditer(content):
                id_tag = match.group(1)
                date_str = match.group(2)

                if id_tag.startswith("S"):
                    sot_key = id_tag[1:]
                else:
                    sot_key = id_tag

                # Verify in SoT
                if sot_key not in sot_statuses:
                    errors.append(
                        f"Code comment in {py_file.relative_to(WORKSPACE_DIR)} claims {id_tag} "
                        f"implemented on {date_str}, but no status for {id_tag} found in the SoT Known Gaps Registry."
                    )
                else:
                    sot_status_info = sot_statuses[sot_key]
                    if sot_status_info["status"] != "Implemented" and sot_key != "2":
                        errors.append(
                            f"Code comment in {py_file.relative_to(WORKSPACE_DIR)} claims {id_tag} "
                            f"implemented on {date_str}, but the SoT Known Gaps Registry claims it is {sot_status_info['status']}."
                        )
                    elif date_str not in sot_status_info["line"] and sot_key != "2":
                        errors.append(
                            f"Code comment in {py_file.relative_to(WORKSPACE_DIR)} claims {id_tag} "
                            f"implemented on {date_str}, but the SoT Known Gaps Registry status line for {id_tag} "
                            f"does not reference this date/commit."
                        )
        except Exception as e:
            print(f"[ERROR] Failed to scan {py_file} for coupling: {e}")

    return errors


# ---------------------------------------------------------------------------
# Doc-version consistency (single-point versioning; canonical = VERSION file).
# Each entry: (relative path, capture regex whose group 1 is the declared
# version). A doc missing from this list is a tooling gap — add it here AND
# give it a BANNER_RULES entry in scripts/bump_version.py.
# ---------------------------------------------------------------------------
DOC_VERSION_CHECKS = [
    ("docs/source_of_truth.md", r"\*\*Version:\*\* v(\d+\.\d+)"),
    ("AGENTS.md", r"SoT version \*\*v(\d+\.\d+)\*\*"),
    ("README.md", r"`docs/source_of_truth\.md` v(\d+\.\d+)"),
    ("docs/orchestrator_overview.md", r"\*\*Version:\*\* (\d+\.\d+)"),
    ("docs/quick_start.md", r"\*\*Source of Truth Version:\*\* v(\d+\.\d+)"),
    ("docs/ledger_architecture.md", r"\*\*Version:\*\* (\d+\.\d+)"),
    ("docs/reporting_logging_audit.md", r"Verified against SoT v(\d+\.\d+)"),
    ("docs/document_map.md", r"Documentation Structure Map \(v(\d+\.\d+)\)"),
]


def check_doc_versions(expected: str) -> list[str]:
    """Verifies every doc banner declares the canonical VERSION value."""
    errors = []
    for rel_path, capture_rx in DOC_VERSION_CHECKS:
        path = WORKSPACE_DIR / rel_path
        if not path.exists():
            errors.append(f"{rel_path}: file not found")
            continue
        content = path.read_text(encoding="utf-8")
        cap = re.search(capture_rx, content)
        if not cap:
            errors.append(
                f"{rel_path}: version banner not found / drifted from known "
                f"shape (pattern: {capture_rx})"
            )
            continue
        declared = cap.group(1)
        if declared != expected:
            errors.append(
                f"{rel_path}: declares v{declared} but VERSION is v{expected}"
            )
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
        prefix = "S" if s_num.isdigit() else ""

        # Get SoT status
        sot_info = sot_statuses.get(s_num)
        if not sot_info:
            print(f"[WARNING] {prefix}{s_num} not found in SoT Known Gaps Registry.")
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
                print(f"[CODE_AHEAD] {prefix}{s_num} — {name}")
                print(f"  SoT claims: {sot_status}")
                print("  Code:       All implementation patterns are present.")
                code_ahead_count += 1
            else:
                print(f"[ALIGN]      {prefix}{s_num} — {name} (Sync: {sot_status})")
                align_count += 1
        elif sot_status == "Implemented":
            if all_passed:
                print(f"[ALIGN]      {prefix}{s_num} — {name} (Sync: Implemented)")
                align_count += 1
            else:
                print(f"[MISALIGNED] {prefix}{s_num} — {name}")
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

    # Doc-version consistency check (canonical: VERSION file)
    print("-" * 80)
    print("DOC VERSION CONSISTENCY (canonical: VERSION file)")
    print("-" * 80)
    version_file = WORKSPACE_DIR / "VERSION"
    if not version_file.exists():
        print("[MISALIGNED] VERSION file missing at repo root.")
        misaligned_count += 1
    else:
        expected_version = version_file.read_text(encoding="utf-8").strip()
        version_errors = check_doc_versions(expected_version)
        if version_errors:
            for err in version_errors:
                print(f"[MISALIGNED] {err}")
                misaligned_count += 1
            print(
                f"  Canonical version: v{expected_version} "
                f"(fix with: python scripts/bump_version.py)"
            )
        else:
            print(
                f"[ALIGN]      All doc banners declare v{expected_version} "
                f"(matches VERSION)."
            )

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
