#!/usr/bin/env python3
"""Preflight ENFORCE check for reproducibility and environment lock.

Checks (SoT Section 3 ENFORCE mode):
- requirements.in and requirements.txt present
- challenge_config.json contains all required SoT v2.0.1 schema fields
- SKILL_STATE.json contains all required human gate keys (using flat boolean format)
- OOF cv_strategy_id tagging validation (via AST parsing)
- Cross-skill import static scan (via AST parsing)
- AutoML import static scan (via AST parsing)

Usage:
  python scripts/preflight_enforce.py --competition competitions/ey-frogs

Exits 0 on success, 1 on failure.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
import argparse


def fail(msg: str):
    print(f"ERROR: {msg}")
    sys.exit(1)


def ok(msg: str):
    print(f"OK: {msg}")


# ---------------------------------------------------------------------------
# Configuration Completeness Checks
# ---------------------------------------------------------------------------

REQUIRED_CONFIG_KEYS = {
    "name",
    "slug",
    "task_type",
    "metric",
    "metric_direction",
    "use_probabilities",
    "drift_threshold",
    "submission_format",
    "submission_budget",
    "daily_limit",
    "total_limit",
    "public_split_pct",
    "private_split_pct",
    "team_allowed",
    "code_review_tier",
    "allowed_external_data",
    "automl_permitted",
    "data_modality",
    "data_shape",
    "phase_skill_map",
    "reproducibility",
    "shap_leak_threshold",
    "variance_gate_threshold",
    "gate_margin",
    "cv_strategy",
    "variants",
    "community_signals",
    "policy_filters",
    "file_hashes",
}

VALID_METRIC_DIRECTIONS = {"maximize", "minimize"}


def check_config_completeness(cfg: dict) -> None:
    """Validate challenge_config.json fields per SoT v2.0.1 schema."""
    # Check top-level keys
    missing_keys = REQUIRED_CONFIG_KEYS - set(cfg.keys())
    if missing_keys:
        fail(f"challenge_config.json is missing required keys: {sorted(missing_keys)}")
    ok("challenge_config.json contains all top-level keys")

    # Nested block checks: reproducibility
    repro = cfg.get("reproducibility")
    if not isinstance(repro, dict):
        fail("challenge_config.json `reproducibility` must be a dictionary")
    seed = repro.get("seed")
    if seed is None or not isinstance(seed, int):
        fail("reproducibility.seed is missing or not an int")
    ok(f"reproducibility.seed: {seed}")

    # Nested block checks: cv_strategy
    cv = cfg.get("cv_strategy")
    if not isinstance(cv, dict):
        fail("challenge_config.json `cv_strategy` must be a dictionary")
    for subfield in (
        "type",
        "n_splits",
        "shuffle",
        "random_state",
        "group_col",
        "stratify_col",
        "selection_reason",
    ):
        if subfield not in cv:
            fail(f"cv_strategy block is missing `{subfield}`")
    if not isinstance(cv.get("n_splits"), int):
        fail("cv_strategy.n_splits must be an int")
    if not cv.get("type"):
        fail("cv_strategy.type must be specified")
    ok(f"cv_strategy block complete ({cv.get('type')})")

    # Numeric/Float checks
    for name, expected_types in [
        ("drift_threshold", (int, float)),
        ("shap_leak_threshold", (int, float)),
        ("variance_gate_threshold", (int, float)),
        ("gate_margin", (int, float)),
    ]:
        val = cfg.get(name)
        if val is None or not isinstance(val, expected_types):
            fail(f"challenge_config.json `{name}` is missing or not a number")
        ok(f"challenge_config.json `{name}`: {val}")

    # Boolean checks
    for name in (
        "use_probabilities",
        "allowed_external_data",
        "automl_permitted",
        "team_allowed",
    ):
        val = cfg.get(name)
        if val is None or not isinstance(val, bool):
            fail(f"challenge_config.json `{name}` is missing or not a boolean")
        ok(f"challenge_config.json `{name}`: {val}")

    # Metric direction check
    direction = cfg.get("metric_direction")
    if direction not in VALID_METRIC_DIRECTIONS:
        fail(
            f"challenge_config.json `metric_direction` must be one of {VALID_METRIC_DIRECTIONS}, got '{direction}'"
        )
    ok(f"challenge_config.json `metric_direction`: {direction}")


def check_human_gate_keys(state: dict, cfg: dict) -> None:
    """Validate standard 5 human gate keys and flat branch booleans."""
    required_gates = [
        "human_gate_1_approved",
        "human_gate_3_approved",
        "human_gate_4_approved",
        "human_gate_5_selection",
    ]
    missing = []
    for key in required_gates:
        val = state.get(key)
        if val is None:
            missing.append(key)
        else:
            expected_type = list if key == "human_gate_5_selection" else bool
            if not isinstance(val, expected_type):
                fail(
                    f"SKILL_STATE.{key} has wrong type: {type(val).__name__}, expected {expected_type.__name__}"
                )

    # Check flat branch approvals using variants list from config
    variants = cfg.get("variants", [])
    if variants:
        for branch in variants:
            gate_key = f"human_gate_2_{branch}_approved"
            val = state.get(gate_key)
            if val is None:
                missing.append(gate_key)
            elif not isinstance(val, bool):
                fail(
                    f"SKILL_STATE.{gate_key} has wrong type: {type(val).__name__}, expected bool"
                )
    else:
        # Fallback: scan for any keys matching human_gate_2_{branch}_approved in state
        gate2_keys = [
            k
            for k in state
            if k.startswith("human_gate_2_") and k.endswith("_approved")
        ]
        for gk in gate2_keys:
            if not isinstance(state[gk], bool):
                fail(
                    f"SKILL_STATE.{gk} has wrong type: {type(state[gk]).__name__}, expected bool"
                )

    # Reject legacy human_gate_2_by_branch container
    if "human_gate_2_by_branch" in state:
        fail(
            "SKILL_STATE contains legacy container 'human_gate_2_by_branch' which is prohibited"
        )

    if missing:
        fail(f"Missing required human gate keys: {missing}")

    ok("SKILL_STATE.json contains all required human gate keys with correct types")


# ---------------------------------------------------------------------------
# Anchor OOF Score Presence Check
# ---------------------------------------------------------------------------

# DAG phase strings written by skill_08 and skill_11 that indicate the anchor
# run should already have completed. anchor_oof_score being null in any of
# these phases is anomalous and worth surfacing as a preflight warning.
_POST_ANCHOR_PHASES = frozenset({
    "phase_2_anchor_confirmed",   # written by skill_08
    "phase_3_gate_blocked",       # written by skill_11 on gate fail
    "phase_3_anchor_promoted",    # written by skill_11 on gate pass
    "phase_3_features",           # written by skill_07 and skill_05
})


def check_anchor_oof_score(state: dict) -> None:
    """Warn (non-blocking) when anchor_oof_score is null post-anchor phases."""
    dag_phase = state.get("dag_phase", "uninitialized")
    if dag_phase not in _POST_ANCHOR_PHASES:
        # Phase hasn't reached the anchor run yet — null is expected.
        return
    if state.get("anchor_oof_score") is None:
        print(
            f"WARN: anchor_oof_score is null at dag_phase='{dag_phase}'. "
            "Phase 2 anchor may not have written state correctly. "
            "Run skill_08_anchor before proceeding to Phase 3."
        )
    else:
        ok(f"anchor_oof_score present at dag_phase='{dag_phase}'")



# ---------------------------------------------------------------------------
# AST Static Code Auditing
# ---------------------------------------------------------------------------


def scan_oof_cv_strategy_tags(skills_dir: Path) -> None:
    """Scan skill modules that call write_oof_record() and assert cv_strategy_id is passed."""
    violations = []
    for skill_file in sorted(skills_dir.glob("skill_*.py")):
        try:
            tree = ast.parse(
                skill_file.read_text(encoding="utf-8"), filename=str(skill_file)
            )
        except Exception as e:
            fail(f"Failed to parse AST of {skill_file.name}: {e}")

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr

                if func_name == "write_oof_record":
                    has_cv_strategy_id = any(
                        kw.arg == "cv_strategy_id" for kw in node.keywords
                    )
                    if not has_cv_strategy_id:
                        violations.append(
                            f"{skill_file.name} calls write_oof_record() at line {node.lineno} "
                            f"without cv_strategy_id keyword argument"
                        )

    if violations:
        for v in violations:
            print(f"  WARN: {v}")
        fail(f"OOF cv_strategy_id tag violations found: {len(violations)}")
    ok("All write_oof_record() calls include cv_strategy_id= kwarg")


def get_ast_imports(filepath: Path) -> list[tuple[int, str]]:
    """Parse python file and extract all imported module names and their line numbers."""
    try:
        tree = ast.parse(filepath.read_text(encoding="utf-8"), filename=str(filepath))
    except Exception as e:
        fail(f"Failed to parse AST of {filepath.name}: {e}")

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                imports.append((node.lineno, name.name))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            level = node.level
            if level > 0:
                imports.append((node.lineno, f"{'.' * level}{module}"))
            else:
                imports.append((node.lineno, module))
    return imports


def check_is_cross_skill_import(module_name: str) -> bool:
    name = module_name.lstrip(".")
    if name.startswith("zindian.skills.skill_"):
        return True
    if name.startswith("skills.skill_"):
        return True
    if name.startswith("skill_") and re.match(r"^skill_\d{2}", name):
        return True
    return False


AUTOML_PACKAGES = {
    "autosklearn",
    "tpot",
    "h2o",
    "autogluon",
    "auto-sklearn",
    "auto_ml",
    "mljar",
}


def check_is_automl_import(module_name: str) -> bool:
    name = module_name.lstrip(".").split(".")[0].lower()
    name_alt = name.replace("-", "_")
    return name in AUTOML_PACKAGES or name_alt in AUTOML_PACKAGES


def scan_cross_skill_imports(skills_dir: Path) -> None:
    """Flag any cross-skill import across the skill modules using AST parsing."""
    violations = []
    for skill_file in sorted(skills_dir.glob("skill_*.py")):
        imports = get_ast_imports(skill_file)
        for lineno, module in imports:
            if check_is_cross_skill_import(module):
                violations.append(
                    f"{skill_file.name} line {lineno}: imports '{module}'"
                )

    if violations:
        for v in violations:
            print(f"  VIOLATION: {v}")
        fail(f"Cross-skill import violations found: {len(violations)}")
    ok("No cross-skill imports detected")


def scan_automl_imports(skills_dir: Path) -> None:
    """Flag any AutoML library import in any skill body using AST parsing."""
    violations = []
    for skill_file in sorted(skills_dir.glob("*.py")):
        imports = get_ast_imports(skill_file)
        for lineno, module in imports:
            if check_is_automl_import(module):
                violations.append(
                    f"{skill_file.name} line {lineno}: imports AutoML package '{module}'"
                )

    if violations:
        for v in violations:
            print(f"  VIOLATION: {v}")
        fail(f"AutoML import violations found: {len(violations)}")
    ok("No AutoML library imports detected")


# ---------------------------------------------------------------------------
# Main Execution Flow
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--competition",
        default=None,
        help="Path to competition folder (e.g., competitions/ey-frogs)",
    )
    args = parser.parse_args()

    root = Path.cwd()
    skills_dir = root / "zindian" / "skills"

    # AST Static scans
    scan_automl_imports(skills_dir)
    scan_cross_skill_imports(skills_dir)
    scan_oof_cv_strategy_tags(skills_dir)

    # Requirements files checks
    req_in = root / "requirements.in"
    req_txt = root / "requirements.txt"
    if not req_in.exists():
        fail("requirements.in missing at project root")
    ok("requirements.in present")

    if not req_txt.exists():
        fail(
            "requirements.txt missing — run: pip-compile requirements.in --output-file requirements.txt"
        )
    ok("requirements.txt present")

    # Competition selection
    if args.competition:
        comp_path = Path(args.competition)
    else:
        comps = [p for p in (root / "competitions").iterdir() if p.is_dir()]
        if not comps:
            fail("No competitions/ folder or empty")
        comp_path = comps[0]

    # Load and validate challenge_config.json
    config_path = comp_path / "challenge_config.json"
    if not config_path.exists():
        fail(f"challenge_config.json not found in {comp_path}")

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    ok("challenge_config.json loaded")
    check_config_completeness(cfg)

    # Load and validate SKILL_STATE.json
    state_path = comp_path / "SKILL_STATE.json"
    if not state_path.exists():
        fail(f"SKILL_STATE.json not found in {comp_path}")

    state = json.loads(state_path.read_text(encoding="utf-8"))
    ok("SKILL_STATE.json loaded")
    check_human_gate_keys(state, cfg)
    check_anchor_oof_score(state)


    print("\nPRELIGHT ENFORCE: ALL CHECKS PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
