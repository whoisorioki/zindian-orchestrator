#!/usr/bin/env python3
"""Test/Demo Script for Phase 1 Skills (Integrity + Intake + Reporter)"""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("PHASE 1 DEMO — Integrity + Intake + Reporter")
print("=" * 80)

# Test data: Mock Zindi API response for challenge_config population
MOCK_ZINDI_CHALLENGE_DATA = {
    "id": 1,
    "title": "Zindi Financial Inclusion in Africa ✨ - Gain Skills",
    "slug": "financial-inclusion-in-africa",
    "evaluation_metric": "mean_absolute_error",
    "submission_format": "unique_id,bank_account",
    "daily_submission_limit": 10,
    "total_submission_limit": None,
    "allowed_external_data": False,
    "automl_permitted": False,
    "data_modality": "tabular",
    "team_allowed": True,
    "public_split_pct": None,
    "private_split_pct": None,
}

# Verify SKILL_STATE.json exists
skill_state_path = Path("SKILL_STATE.json")
if not skill_state_path.exists():
    print("ERROR: SKILL_STATE.json not found in current directory")
    print("Run this script from the project root: cd /path/to/zindian_orchestrator")
    sys.exit(1)

print("\n" + "=" * 80)
print("[1/3] SKILL 01 — Integrity Audit (MD5 Hash Lock)")
print("=" * 80)

try:
    from zindian.skills.skill_01_integrity import run as skill_01_run

    result = skill_01_run()

    print(f"\nStatus: {result['status']}")
    if result["status"] == "GO":
        print(f"✓ MD5 Hash: {result['md5_target_hash'][:16]}...")
        print(f"✓ Rows Locked: {result['rows_locked']}")
    else:
        print(f"✗ Error: {result['message']}")
        if "traceback" in result:
            print(result["traceback"])

except Exception as e:
    print(f"✗ Skill 01 failed to run: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 80)
print("[2/3] SKILL 02 — Challenge Intake (Config Populator)")
print("=" * 80)

try:
    from zindian.skills.skill_02_intake import run as skill_02_run

    result = skill_02_run(slug=str(MOCK_ZINDI_CHALLENGE_DATA["slug"]))

    print(f"\nStatus: {result['status']}")
    if result["status"] == "GO":
        config = result["config"]
        print(f"✓ Competition: {config['slug']}")
        print(f"✓ Metric: {config['metric']}")
        print(f"✓ Domain: {config['domain']}")
        print(f"✓ Daily Limit: {config['daily_limit']}")
        print(f"✓ Use Probabilities: {config['use_probabilities']}")
    else:
        print(f"✗ Error: {result['message']}")
        if "traceback" in result:
            print(result["traceback"])

except Exception as e:
    print(f"✗ Skill 02 failed to run: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 80)
print("[3/3] SKILL 15 — Reporter (DuckDB Initialization)")
print("=" * 80)

try:
    from zindian.skills.skill_15_reporter import run as skill_15_run

    result = skill_15_run()

    print(f"\nStatus: {result['status']}")
    if result["status"] == "GO":
        print(f"✓ Ledger: {result['ledger_path']}")
        print(f"✓ Experiments Table Rows: {result['experiments_count']}")
        print(f"✓ Submissions Table Rows: {result['submissions_count']}")
        print(f"✓ Phase 1 Summary: {result['phase_1_summary_path']}")
    else:
        print(f"✗ Error: {result['message']}")
        if "traceback" in result:
            print(result["traceback"])

except Exception as e:
    print(f"✗ Skill 15 failed to run: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 80)
print("PHASE 1 DEMO COMPLETE")
print("=" * 80)

# Print summary of generated files
print("\nGenerated/Updated Files:")
for file_path in [
    "SKILL_STATE.json",
    "challenge_config.json",
    "reports/integrity_audit.json",
    "reports/skill_02_summary.json",
    "reports/phase_1_summary.json",
    "reports/experiments.db",
]:
    path = Path(file_path)
    if path.exists():
        size = path.stat().st_size
        print(f"  ✓ {file_path} ({size} bytes)")
    else:
        print(f"  · {file_path} (not yet created)")

print("\n" + "=" * 80)
print("Next Steps:")
print("  1. Review generated files in reports/ directory")
print("  2. Proceed to Phase 2: Create EDA skill and Anchor baseline")
print("=" * 80)
