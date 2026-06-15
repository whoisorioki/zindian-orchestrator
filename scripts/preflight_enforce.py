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


def verify_section_1_assumptions(comp_path: Path, cfg: dict, state: dict, skills_dir: Path, root: Path) -> None:
    """Programmatically verify SoT Section 1 Assumptions (A1-A10)."""
    print("\n--- Verifying Section 1 Assumptions ---")
    
    # A1 - Single competition at a time
    import os
    env_slug = os.environ.get("COMPETITION_SLUG")
    cfg_slug = cfg.get("slug")
    if env_slug and cfg_slug and env_slug != cfg_slug:
        fail(f"[A1 Scoping Violation] Environment slug '{env_slug}' does not match challenge config slug '{cfg_slug}'")
    ok("A1 check: COMPETITION_SLUG matches challenge_config slug")

    # A2 - Tabular data only
    INVALID_A2_EXTENSIONS = {
        '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp',
        '.mp3', '.wav', '.flac', '.ogg', '.m4a',
        '.mp4', '.avi', '.mkv', '.mov', '.webm',
        '.gml', '.graphml', '.h5', '.hdf5', '.xml'
    }
    raw_dir = comp_path / "data" / "raw"
    if raw_dir.exists():
        invalid_files = []
        for p in raw_dir.glob("**/*"):
            if p.is_file() and p.suffix.lower() in INVALID_A2_EXTENSIONS:
                invalid_files.append(p.name)
        if invalid_files:
            fail(f"[A2 Tabular Signature Violation] Non-tabular files found: {invalid_files}")
    ok("A2 check: No non-tabular file extensions in raw data folder")

    # A3 - Zindi platform conventions
    budget = cfg.get("submission_budget")
    if isinstance(budget, dict):
        total_budget = budget.get("total")
    else:
        total_budget = budget
    if total_budget is None or not isinstance(total_budget, (int, float)) or total_budget > 30:
        fail(f"[A3 Zindi Limits Violation] submission_budget total must be <= 30, got {total_budget}")
    if cfg.get("automl_permitted") is not False:
        fail(f"[A3 Zindi Limits Violation] automl_permitted must be False, got {cfg.get('automl_permitted')}")
    ok("A3 check: submission_budget <= 30 and automl_permitted is False")

    # A4 - Supervised learning only
    train_file_name = cfg.get("input_files", {}).get("train") or "Train.csv"
    train_path = comp_path / "data" / "raw" / train_file_name
    if not train_path.exists():
        train_path = comp_path / "data" / "raw" / "Train.csv"
    if not train_path.exists():
        fail(f"[A4 Target In Schema Violation] Raw training data not found at {train_path}")
    target_col = cfg.get("target_col") or cfg.get("target_column")
    if not target_col:
        fail("[A4 Target In Schema Violation] target_col is not defined in challenge_config.json")
    
    header = []
    if train_path.suffix.lower() == ".csv":
        try:
            with open(train_path, "r", encoding="utf-8") as f:
                first_line = f.readline()
                header = [col.strip().strip('"').strip("'") for col in first_line.split(",")]
        except Exception as e:
            fail(f"Failed to read CSV header from {train_path}: {e}")
    elif train_path.suffix.lower() == ".parquet":
        try:
            import pandas as pd
            df_sample = pd.read_parquet(train_path, columns=[])
            header = list(df_sample.columns)
        except Exception as e:
            try:
                import pyarrow.parquet as pq
                meta = pq.read_metadata(train_path)
                header = meta.schema.names
            except Exception as e2:
                fail(f"Failed to read Parquet header from {train_path}: {e2}")
    if header:
        if not any(h.lower() == target_col.lower() for h in header):
            fail(f"[A4 Target In Schema Violation] Target column '{target_col}' not found in raw training columns: {header}")
    ok(f"A4 check: Target '{target_col}' is present in training data schema")

    # A5 - No hardcoded competition-specific values anywhere
    prohibited_strings = set()
    if cfg_slug:
        prohibited_strings.add(cfg_slug.lower())
    if target_col:
        prohibited_strings.add(target_col.lower())
    cv = cfg.get("cv_strategy") or {}
    group_col = cv.get("group_col")
    if group_col and str(group_col).strip():
        prohibited_strings.add(str(group_col).lower())
    stratify_col = cv.get("stratify_col")
    if stratify_col and str(stratify_col).strip():
        prohibited_strings.add(str(stratify_col).lower())

    violations_a5 = []
    for skill_file in sorted(skills_dir.glob("skill_*.py")):
        try:
            tree = ast.parse(skill_file.read_text(encoding="utf-8"), filename=str(skill_file))
        except Exception as e:
            fail(f"Failed to parse AST of {skill_file.name}: {e}")
        for node in ast.walk(tree):
            val = None
            if isinstance(node, ast.Constant):
                val = node.value
            elif node.__class__.__name__ == "Str":
                val = node.s
            if isinstance(val, str):
                val_lower = val.lower()
                if val_lower in prohibited_strings and len(val_lower) > 2:
                    violations_a5.append(
                        f"{skill_file.name} line {node.lineno}: contains hardcoded competition-specific string literal '{val}'"
                    )
    if violations_a5:
        for v in violations_a5:
            print(f"  VIOLATION: {v}")
        fail(f"A5 Hardcoded competition value violations found: {len(violations_a5)}")
    ok("A5 check: No hardcoded competition-specific strings in skills")

    # A6 - SKILL_STATE.json is the single source of truth for execution state
    state_py = root / "zindian" / "state.py"
    if not state_py.exists():
        fail("zindian/state.py not found")
    state_content = state_py.read_text(encoding="utf-8")
    if "os.replace" not in state_content:
        fail("[A6 Isolation Violation] Atomic state write mechanism (os.replace) not found in zindian/state.py")
    ok("A6 check: Atomic state write mechanism present in state.py")

    # A7 - The OOF contract is universal
    oof_keys = [k for k in state if k.startswith("branch_") and k.endswith("_oof")]
    for k in oof_keys:
        record = state[k]
        if isinstance(record, dict):
            cv_id = record.get("cv_strategy_id")
            if not cv_id:
                fail(f"[A7 OOF contract Violation] '{k}' is missing 'cv_strategy_id' tag in state")
    ok("A7 check: All OOF records carry a cv_strategy_id tag")

    # A8 - Spatial signals are group signals
    spatial = cfg.get("spatial_signal", {})
    cv_strategy = cfg.get("cv_strategy", {})
    if spatial.get("present", False) or spatial.get("group_col"):
        cv_type = cv_strategy.get("type")
        if cv_type != "GroupKFold":
            fail(f"[A8 Spatial Route Violation] Spatial signal is present but cv_strategy.type is '{cv_type}', expected 'GroupKFold'")
    ok("A8 check: Spatial structures route strictly to GroupKFold")

    # A9 - The research sidecar is non-blocking at every consumption point
    unsafe_keys = {"sidecar_recommendations", "cv_strategy_override", "pseudo_label_result", "anchor_challenge", "eda"}
    violations_a9 = []
    for skill_file in sorted(skills_dir.glob("skill_*.py")):
        try:
            tree = ast.parse(skill_file.read_text(encoding="utf-8"), filename=str(skill_file))
        except Exception as e:
            fail(f"Failed to parse AST of {skill_file.name}: {e}")
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript):
                slice_val = None
                if isinstance(node.slice, ast.Constant):
                    slice_val = node.slice.value
                elif node.slice.__class__.__name__ == "Str":
                    slice_val = node.slice.s
                elif isinstance(node.slice, ast.Index): # Python < 3.9
                    if isinstance(node.slice.value, ast.Constant):
                        slice_val = node.slice.value
                    elif node.slice.value.__class__.__name__ == "Str":
                        slice_val = node.slice.value.s
                if slice_val in unsafe_keys:
                    violations_a9.append(
                        f"{skill_file.name} line {node.lineno}: direct bracket access on unsafe key '{slice_val}'"
                    )
    if violations_a9:
        for v in violations_a9:
            print(f"  VIOLATION: {v}")
        fail(f"A9 Unsafe state key bracket access violations found: {len(violations_a9)}")
    ok("A9 check: All research sidecar / optional state reads use safe .get() patterns")

    # A10 - Python environment is stable and reproducible
    req_txt = root / "requirements.txt"
    if req_txt.exists():
        content = req_txt.read_text(encoding="utf-8")
        if "autogenerated by pip-compile" not in content:
            fail("[A10 Pip Signatures Violation] requirements.txt is missing the pip-compile autogenerated signature header")
    else:
        fail("requirements.txt missing")
    ok("A10 check: requirements.txt has pip-compile signature header")


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

    # Programmatic Section 1 Assertions (A1-A10)
    verify_section_1_assumptions(comp_path, cfg, state, skills_dir, root)

    print("\nPRELIGHT ENFORCE: ALL CHECKS PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
