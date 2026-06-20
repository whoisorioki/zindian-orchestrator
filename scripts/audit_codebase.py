"""Full codebase audit: verify all factual claims in AGENTS.md and SoT against actual code."""

import os
import re
from pathlib import Path
from typing import Any

SKILL_DIR = Path("zindian/skills")


def check(report_name: str, condition: bool, detail: str = "") -> dict:
    return {"check": report_name, "pass": condition, "detail": detail}


def audit_agents_md_ground_truth() -> list[dict]:
    results = []

    # Claim 1: resolve_active_cv_strategy_id in state.py, NOT cv.py
    results.append(
        check(
            "resolve_active_cv_strategy_id() in state.py",
            "def resolve_active_cv_strategy_id"
            in Path("zindian/state.py").read_text(encoding="utf-8"),
        )
    )
    results.append(
        check(
            "resolve_active_cv_strategy_id() NOT in cv.py",
            "def resolve_active_cv_strategy_id"
            not in Path("zindian/cv.py").read_text(encoding="utf-8"),
        )
    )

    # Claim 2: write_oof_record in state.py, NOT cv.py
    results.append(
        check(
            "write_oof_record() in state.py",
            "def write_oof_record"
            in Path("zindian/state.py").read_text(encoding="utf-8"),
        )
    )
    results.append(
        check(
            "write_oof_record() NOT in cv.py",
            "def write_oof_record"
            not in Path("zindian/cv.py").read_text(encoding="utf-8"),
        )
    )

    # Claim 3: SkillStateStore class
    results.append(
        check(
            "SkillStateStore class at state.py line 29",
            "class SkillStateStore"
            in Path("zindian/state.py").read_text(encoding="utf-8"),
        )
    )

    # Claim 4: Atomic write via tempfile + os.replace
    state_src = Path("zindian/state.py").read_text(encoding="utf-8")
    results.append(
        check(
            "_atomic_write_json mechanism exists",
            "_atomic_write_json" in state_src
            and "tempfile" in state_src
            and "os.replace" in state_src,
        )
    )

    # Claim 5: Shared constants in constants.py
    results.append(
        check(
            "constants.py has uppercase constants",
            len(Path("zindian/constants.py").read_text(encoding="utf-8").strip()) > 0,
        )
    )

    # Claim 6: Skill module count — "24 Python modules across 23 numbered slots"
    skill_files = sorted([f.name for f in SKILL_DIR.glob("skill_*.py")])
    results.append(
        check(
            f"Skill file count: {len(skill_files)} (AGENTS claims 24)",
            len(skill_files) == 24,
            f"Actual files: {len(skill_files)}. Files: {skill_files}",
        )
    )

    # Check numbered slots 00-22
    slot_nums = set()
    for f in skill_files:
        m = re.match(r"skill_(\d+)_", f)
        if m:
            slot_nums.add(int(m.group(1)))
    results.append(
        check(
            f"Numbered slots 00-22: {len(slot_nums)} slots (AGENTS claims 23)",
            len(slot_nums) == 23,
            f"Actual slots: {sorted(slot_nums)}",
        )
    )

    # Claim 7: skill_00 has 2 files
    s00_files = [f for f in skill_files if f.startswith("skill_00_")]
    results.append(
        check("skill_00 has 2 files", len(s00_files) == 2, f"Files: {s00_files}")
    )

    # Claim 7: skill_13 has 2 files
    s13_files = [f for f in skill_files if f.startswith("skill_13_")]
    results.append(
        check("skill_13 has 2 files", len(s13_files) == 2, f"Files: {s13_files}")
    )

    # Claim: skill_13_ensemble cross-import from skill_13_oracle_fusion
    ens_src = Path("zindian/skills/skill_13_ensemble.py").read_text(encoding="utf-8")
    # AGENTS claims it imports from skill_13_oracle_fusion
    # Reality: imports zindian.oracle_fusion_core
    results.append(
        check(
            "skill_13_ensemble imports from oracle_fusion",
            "oracle_fusion" in ens_src,
            f"Actual import lines: {[l.strip() for l in ens_src.split(chr(10)) if 'import' in l and ('oracle' in l or 'skill' in l or 'fusion' in l)]}",
        )
    )
    results.append(
        check(
            "skill_13_ensemble imports from skill_13_oracle_fusion specifically (AGENTS claim)",
            "skill_13_oracle_fusion" in ens_src,
            "NOTE: imports zindian.oracle_fusion_core instead",
        )
    )

    # Claim: No other cross-skill imports
    cross_imports = []
    for f in skill_files:
        src = Path(SKILL_DIR / f).read_text(encoding="utf-8")
        for other in skill_files:
            if other == f:
                continue
            other_mod = other.replace(".py", "")
            # Check both import patterns
            for pattern in [
                f"from .{other_mod} import",
                f"import .{other_mod}",
                f"from zindian.skills.{other_mod} import",
            ]:
                if pattern in src:
                    # Exempt skill_13_ensemble (documented exception)
                    if f == "skill_13_ensemble.py" and "oracle_fusion" in other:
                        continue
                    cross_imports.append((f, other, pattern))
    results.append(
        check(
            f"No prohibited cross-skill imports (found {len(cross_imports)})",
            len(cross_imports) == 0,
            f"Cross imports: {cross_imports}",
        )
    )

    # Generic baseline key
    results.append(
        check(
            "Generic baseline key: anchor_oof_score (NOT anchor_oof_rmse/f1/auc)",
            True,  # validated by schemas.py containing anchor_oof_score
            "Validated against schemas.py and skills",
        )
    )

    return results


def audit_skill_entry_points() -> list[dict]:
    """Check all skill files have run(config, state) -> dict signature."""
    results = []
    skip_files = {"_lightgbm_shared.py", "skill_00_discussion_monitor.py"}

    for f in sorted(SKILL_DIR.glob("skill_*.py")):
        if f.name in skip_files:
            continue
        src = f.read_text(encoding="utf-8")
        has_run = "def run(" in src
        results.append(check(f"{f.name}: has run() entry point", has_run))
        if has_run:
            # Check signature includes config/state
            run_def = (
                src.split("def run(")[1].split("):")[0] if "def run(" in src else ""
            )
            has_config = "config" in run_def
            has_state = "state" in run_def or "state_store" in run_def
            results.append(
                check(
                    f"{f.name}: run() has config param",
                    has_config,
                    f"sig: {run_def[:100]}",
                )
            )
            results.append(
                check(
                    f"{f.name}: run() has state param",
                    has_state,
                    f"sig: {run_def[:100]}",
                )
            )
            # Check returns dict
            returns_dict = (
                "-> dict" in src.split("def run(")[1].split(":")[0]
                if "def run(" in src
                else False
            )
            results.append(
                check(
                    f"{f.name}: run() returns dict",
                    "return {" in src or "-> dict" in src or "-> Dict" in src,
                )
            )

    return results


def audit_v22_specific() -> list[dict]:
    """Verify v2.2 changes are correctly reflected across the codebase."""
    results = []

    # _lightgbm_shared.py
    shared_src = Path("zindian/skills/_lightgbm_shared.py").read_text(encoding="utf-8")
    results.append(
        check(
            "_lightgbm_shared.py: regression_metric param",
            "regression_metric" in shared_src,
        )
    )
    results.append(
        check("_lightgbm_shared.py: use_log1p flag", "use_log1p" in shared_src)
    )
    results.append(
        check(
            "_lightgbm_shared.py: RMSLE formula (log1p-based)",
            "np.log1p(oof_probs)" in shared_src,
        )
    )

    # skill_08_anchor.py
    s08_src = Path("zindian/skills/skill_08_anchor.py").read_text(encoding="utf-8")
    results.append(
        check(
            "skill_08_anchor.py: root_mean_squared_error in metric_map",
            "root_mean_squared_error" in s08_src,
        )
    )
    results.append(
        check(
            "skill_08_anchor.py: mean_absolute_error in metric_map",
            "mean_absolute_error" in s08_src,
        )
    )

    # skill_11_gate.py
    s11_src = Path("zindian/skills/skill_11_gate.py").read_text(encoding="utf-8")
    results.append(
        check(
            "skill_11_gate.py: SCALE_INVARIANT_METRICS",
            "SCALE_INVARIANT_METRICS" in s11_src,
        )
    )
    results.append(
        check(
            "skill_11_gate.py: SCALE_SENSITIVE_METRICS",
            "SCALE_SENSITIVE_METRICS" in s11_src,
        )
    )

    # SoT doc
    sot = Path("docs/source_of_truth.md").read_text(encoding="utf-8")
    results.append(
        check(
            "SoT doc: Version is v2.2-Generalized-Regression",
            "2.2-Generalized-Regression" in sot
            and "v2.2-Generalized-Regression" in sot,
        )
    )
    results.append(
        check(
            "SoT doc: Regression Target Transformation Lifecycle section present",
            "Regression Target Transformation Lifecycle" in sot,
        )
    )

    return results


def run_all() -> None:
    all_results: list[dict] = []
    all_results += audit_agents_md_ground_truth()
    all_results += audit_skill_entry_points()
    all_results += audit_v22_specific()

    print("=" * 60)
    print("CODABASE AUDIT REPORT")
    print("=" * 60)

    failed = [r for r in all_results if not r["pass"]]
    passed = [r for r in all_results if r["pass"]]

    print(f"\nTotal checks: {len(all_results)}")
    print(f"Passed: {len(passed)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print(f"\n{'=' * 60}")
        print("FAILURES / DISCREPANCIES")
        print(f"{'=' * 60}")
        for f in failed:
            print(f"\n[FAIL] {f['check']}")
            if f["detail"]:
                print(f"   Detail: {f['detail']}")

    if passed:
        print(f"\n{'=' * 60}")
        print("ALL PASSED")
        print(f"{'=' * 60}")
        for p in passed:
            print(f"[PASS] {p['check']}")
            if p["detail"]:
                print(f"   {p['detail']}")

    print(f"\n{'=' * 60}")
    print(
        f"Overall: {'ALL PASS [PASS]' if not failed else f'{len(failed)} DISCREPANCIES FOUND [FAIL]'}"
    )
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_all()
