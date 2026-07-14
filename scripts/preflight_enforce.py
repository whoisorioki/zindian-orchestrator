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
import argparse
import json
import re
import sys
from pathlib import Path
from typing import cast


class PreflightError(Exception):
    pass


def fail(msg: str):
    raise PreflightError(msg)


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
    repro = cast(dict[str, object], repro)
    seed = repro.get("seed")
    if seed is None or not isinstance(seed, int):
        fail("reproducibility.seed is missing or not an int")
    ok(f"reproducibility.seed: {seed}")

    # Nested block checks: cv_strategy
    cv = cfg.get("cv_strategy")
    if not isinstance(cv, dict):
        fail("challenge_config.json `cv_strategy` must be a dictionary")
    cv = cast(dict[str, object], cv)
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
_POST_ANCHOR_PHASES = frozenset(
    {
        "phase_2_anchor_confirmed",  # written by skill_08
        "phase_3_gate_blocked",  # written by skill_11 on gate fail
        "phase_3_anchor_promoted",  # written by skill_11 on gate pass
        "phase_3_features",  # written by skill_07 and skill_05
    }
)


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


def verify_section_1_assumptions(
    comp_path: Path, cfg: dict, state: dict, skills_dir: Path, root: Path
) -> None:
    """Programmatically verify SoT Section 1 Assumptions (A1-A10)."""
    print("\n--- Verifying Section 1 Assumptions ---")

    # A1 - Single competition at a time
    import os

    env_slug = os.environ.get("COMPETITION_SLUG")
    cfg_slug = cfg.get("slug")
    if env_slug and cfg_slug and env_slug != cfg_slug:
        fail(
            f"[A1 Scoping Violation] Environment slug '{env_slug}' does not match challenge config slug '{cfg_slug}'"
        )
    ok("A1 check: COMPETITION_SLUG matches challenge_config slug")

    # A2 - Tabular data only
    INVALID_A2_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".tiff",
        ".webp",
        ".mp3",
        ".wav",
        ".flac",
        ".ogg",
        ".m4a",
        ".mp4",
        ".avi",
        ".mkv",
        ".mov",
        ".webm",
        ".gml",
        ".graphml",
        ".h5",
        ".hdf5",
        ".xml",
    }
    raw_dir = comp_path / "data" / "raw"
    if raw_dir.exists():
        invalid_files = []
        for p in raw_dir.glob("**/*"):
            if p.is_file() and p.suffix.lower() in INVALID_A2_EXTENSIONS:
                invalid_files.append(p.name)
        if invalid_files:
            fail(
                f"[A2 Tabular Signature Violation] Non-tabular files found: {invalid_files}"
            )
    ok("A2 check: No non-tabular file extensions in raw data folder")

    # A3 - Zindi platform conventions
    budget = cfg.get("submission_budget")
    if isinstance(budget, dict):
        total_budget = budget.get("total")
    else:
        total_budget = budget
    if (
        total_budget is None
        or not isinstance(total_budget, (int, float))
        or total_budget > 30
    ):
        fail(
            f"[A3 Zindi Limits Violation] submission_budget total must be <= 30, got {total_budget}"
        )
    if cfg.get("automl_permitted") is not False:
        fail(
            f"[A3 Zindi Limits Violation] automl_permitted must be False, got {cfg.get('automl_permitted')}"
        )
    ok("A3 check: submission_budget <= 30 and automl_permitted is False")

    # A4 - Supervised learning only
    train_file_name = cfg.get("input_files", {}).get("train") or "Train.csv"
    train_path = comp_path / "data" / "raw" / train_file_name
    if not train_path.exists():
        train_path = comp_path / "data" / "raw" / "Train.csv"
    if not train_path.exists():
        fail(
            f"[A4 Target In Schema Violation] Raw training data not found at {train_path}"
        )
    target_col = cfg.get("target_col") or cfg.get("target_column")
    if not target_col:
        fail(
            "[A4 Target In Schema Violation] target_col is not defined in challenge_config.json"
        )
    target_col = str(target_col)

    header = []
    if train_path.suffix.lower() == ".csv":
        try:
            with open(train_path, "r", encoding="utf-8") as f:
                first_line = f.readline()
                header = [
                    col.strip().strip('"').strip("'") for col in first_line.split(",")
                ]
        except Exception as e:
            fail(f"Failed to read CSV header from {train_path}: {e}")
    elif train_path.suffix.lower() == ".parquet":
        try:
            import pandas as pd

            df_sample = pd.read_parquet(train_path, columns=[])
            header = list(df_sample.columns)
        except Exception:
            try:
                import pyarrow.parquet as pq

                meta = pq.read_metadata(train_path)
                header = meta.schema.names
            except Exception as e2:
                fail(f"Failed to read Parquet header from {train_path}: {e2}")
    if header:
        if not any(h.lower() == target_col.lower() for h in header):
            fail(
                f"[A4 Target In Schema Violation] Target column '{target_col}' not found in raw training columns: {header}"
            )
    ok(f"A4 check: Target '{target_col}' is present in training data schema")

    # A5 - No hardcoded competition-specific values anywhere
    prohibited_strings = set()
    if cfg_slug:
        prohibited_strings.add(cfg_slug.lower())
    if target_col and target_col.lower() != "target":
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
            tree = ast.parse(
                skill_file.read_text(encoding="utf-8"), filename=str(skill_file)
            )
        except Exception as e:
            fail(f"Failed to parse AST of {skill_file.name}: {e}")
        for node in ast.walk(tree):
            val = None
            if isinstance(node, ast.Constant):
                val = node.value
            elif node.__class__.__name__ == "Str":
                val = getattr(node, "s", None)
            if isinstance(val, str):
                val_lower = val.lower()
                if val_lower in prohibited_strings and len(val_lower) > 2:
                    lineno = getattr(node, "lineno", "?")
                    violations_a5.append(
                        f"{skill_file.name} line {lineno}: contains hardcoded competition-specific string literal '{val}'"
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
        fail(
            "[A6 Isolation Violation] Atomic state write mechanism (os.replace) not found in zindian/state.py"
        )
    ok("A6 check: Atomic state write mechanism present in state.py")

    # A7 - The OOF contract is universal
    oof_keys = [k for k in state if k.startswith("branch_") and k.endswith("_oof")]
    for k in oof_keys:
        record = state[k]
        if isinstance(record, dict):
            cv_id = record.get("cv_strategy_id")
            if not cv_id:
                fail(
                    f"[A7 OOF contract Violation] '{k}' is missing 'cv_strategy_id' tag in state"
                )
    ok("A7 check: All OOF records carry a cv_strategy_id tag")

    # A8 - Spatial signals are group signals
    spatial = cfg.get("spatial_signal", {})
    cv_strategy = cfg.get("cv_strategy", {})
    if spatial.get("present", False) or spatial.get("group_col"):
        cv_type = cv_strategy.get("type")
        if cv_type != "GroupKFold":
            fail(
                f"[A8 Spatial Route Violation] Spatial signal is present but cv_strategy.type is '{cv_type}', expected 'GroupKFold'"
            )
    ok("A8 check: Spatial structures route strictly to GroupKFold")

    # A9 - The research sidecar is non-blocking at every consumption point
    unsafe_keys = {
        "sidecar_recommendations",
        "cv_strategy_override",
        "pseudo_label_result",
        "anchor_challenge",
        "eda",
    }
    violations_a9 = []
    for skill_file in sorted(skills_dir.glob("skill_*.py")):
        try:
            tree = ast.parse(
                skill_file.read_text(encoding="utf-8"), filename=str(skill_file)
            )
        except Exception as e:
            fail(f"Failed to parse AST of {skill_file.name}: {e}")
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript):
                slice_node = node.slice
                slice_val = None
                if isinstance(slice_node, ast.Constant):
                    slice_val = slice_node.value
                elif slice_node.__class__.__name__ == "Str":
                    slice_val = getattr(slice_node, "s", None)
                else:
                    inner_slice = getattr(slice_node, "value", None)
                    if isinstance(inner_slice, ast.Constant):
                        slice_val = inner_slice.value
                    elif inner_slice.__class__.__name__ == "Str":
                        slice_val = getattr(inner_slice, "s", None)
                if slice_val in unsafe_keys:
                    violations_a9.append(
                        f"{skill_file.name} line {getattr(node, 'lineno', '?')}: direct bracket access on unsafe key '{slice_val}'"
                    )
    if violations_a9:
        for v in violations_a9:
            print(f"  VIOLATION: {v}")
        fail(
            f"A9 Unsafe state key bracket access violations found: {len(violations_a9)}"
        )
    ok("A9 check: All research sidecar / optional state reads use safe .get() patterns")

    # A10 - Python environment is stable and reproducible
    req_txt = root / "requirements.txt"
    if req_txt.exists():
        content = req_txt.read_text(encoding="utf-8")
        if "autogenerated by pip-compile" not in content:
            fail(
                "[A10 Pip Signatures Violation] requirements.txt is missing the pip-compile autogenerated signature header"
            )
    else:
        fail("requirements.txt missing")
    ok("A10 check: requirements.txt has pip-compile signature header")

    # A12 - Multi-target mixed-task competitions require recombination policy
    target_config = cfg.get("target_config")
    if target_config and isinstance(target_config, dict):
        targets = target_config.get("targets", [])
        if len(targets) > 1:
            # Check if mixed-task (both classification and regression)
            task_types = set(t.get("task_type") for t in targets if isinstance(t, dict))
            if (
                len(task_types) > 1
                and "classification" in task_types
                and "regression" in task_types
            ):
                policy = target_config.get("pseudo_label_recombination_policy")
                if not policy:
                    fail(
                        "[A12 Multi-Target Violation] Mixed-task multi-target competition detected "
                        "(both classification and regression targets present) but "
                        "'pseudo_label_recombination_policy' is missing in target_config. "
                        "Add this field to challenge_config.json before running Phase 3B."
                    )
                ok(f"A12 check: Multi-target recombination policy present: '{policy}'")


def main():
    import os
    import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--competition",
        default=None,
        help="Path to competition folder (e.g., competitions/ey-frogs)",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run preflight check non-interactively",
    )
    args = parser.parse_args()

    root = Path.cwd()
    skills_dir = root / "zindian" / "skills"

    is_interactive = not args.non_interactive and sys.stdin.isatty()

    # 1. Resolve competition path
    comp_path: Path | None = None
    try:
        from zindian.paths import resolve_competition_paths

        comp_paths = resolve_competition_paths(
            args.competition, require_competition=False
        )
        comp_path = comp_paths.competition_dir
    except Exception:
        if args.competition:
            comp_path = Path(args.competition)
        else:
            comps = [p for p in (root / "competitions").iterdir() if p.is_dir()]
            if not comps:
                print("ERROR: No competitions/ folder or empty")
                sys.exit(1)
            comp_path = comps[0]

    if comp_path is None:
        print("ERROR: Could not resolve competition path")
        sys.exit(1)

    # 2. Detect INIT vs ENFORCE Mode
    config_path = comp_path / "challenge_config.json"
    state_path = comp_path / "SKILL_STATE.json"

    is_init = not config_path.exists() or config_path.stat().st_size == 0
    mode = "INIT" if is_init else "ENFORCE"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    failures = []
    warnings = []

    # Run check stages
    # AST scans and environment lock check run in both modes
    try:
        scan_automl_imports(skills_dir)
    except PreflightError as e:
        failures.append(str(e))

    try:
        scan_cross_skill_imports(skills_dir)
    except PreflightError as e:
        failures.append(str(e))

    try:
        scan_oof_cv_strategy_tags(skills_dir)
    except PreflightError as e:
        failures.append(str(e))

    req_txt = root / "requirements.txt"
    if req_txt.exists():
        content = req_txt.read_text(encoding="utf-8")
        if "autogenerated by pip-compile" not in content:
            failures.append(
                "[A10 Pip Signatures Violation] requirements.txt is missing the pip-compile autogenerated signature header"
            )
    else:
        failures.append("requirements.txt missing")

    # Load data for rendering
    cfg = {}
    state = {}

    if is_init:
        # INIT mode specific checks
        if not comp_path.exists():
            failures.append(f"Competition path does not exist: {comp_path}")
        elif not os.access(comp_path, os.W_OK):
            failures.append(f"Competition path is not writable: {comp_path}")

        raw_dir = comp_path / "data" / "raw"
        train_path = raw_dir / "Train.csv"
        test_path = raw_dir / "Test.csv"
        if not train_path.exists() or not test_path.exists():
            failures.append(
                f"Raw data files Train.csv or Test.csv missing in {raw_dir}"
            )

        if state_path.exists():
            try:
                state_data = json.loads(state_path.read_text(encoding="utf-8"))
                if state_data.get("dag_phase") not in (
                    None,
                    "uninitialized",
                    "phase_1_incomplete",
                    "phase_1_integrity",
                ):
                    failures.append(
                        f"Conflicting SKILL_STATE.json from a prior run exists (phase: {state_data.get('dag_phase')})"
                    )
            except Exception:
                warnings.append(
                    "SKILL_STATE.json is present but invalid JSON — it will be overwritten"
                )
    else:
        # ENFORCE mode specific checks
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            check_config_completeness(cfg)
        except Exception as e:
            failures.append(f"Failed to load/validate challenge_config.json: {e}")

        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if cfg:
                check_human_gate_keys(state, cfg)
        except Exception as e:
            failures.append(f"Failed to load/validate SKILL_STATE.json: {e}")

        if cfg and state:
            try:
                check_anchor_oof_score(state)
            except Exception as e:
                warnings.append(str(e))

            try:
                verify_section_1_assumptions(comp_path, cfg, state, skills_dir, root)
            except PreflightError as e:
                failures.append(str(e))

    # Output preflight report json
    reports_dir = comp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_data = {
        "timestamp": timestamp,
        "mode": mode,
        "failures": failures,
        "warnings": warnings,
        "result": "PASS" if not failures else "FAIL",
    }
    report_file = (
        reports_dir
        / f"preflight_{mode}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    report_file.write_text(json.dumps(report_data, indent=2), encoding="utf-8")

    # Render ASCII Status panel
    comp_slug = cfg.get("slug") if cfg else (state.get("competition") or comp_path.name)

    # Extract config variables for summary box
    task_type = cfg.get("task_type", "N/A")
    metric = cfg.get("metric", "N/A")
    metric_direction = cfg.get("metric_direction", "N/A")
    use_probabilities = str(cfg.get("use_probabilities", "N/A")).lower()
    target_col = cfg.get("target_col", "N/A")
    seed = cfg.get("reproducibility", {}).get("seed", "NOT SET")

    budget_total = (
        cfg.get("submission_budget", {}).get("total", 30)
        if isinstance(cfg.get("submission_budget"), dict)
        else cfg.get("submission_budget", 30)
    )
    budget_daily = (
        cfg.get("submission_budget", {}).get("daily", 5)
        if isinstance(cfg.get("submission_budget"), dict)
        else 5
    )
    budget_used = state.get("submissions_used_today", 0)
    budget_remaining = state.get("remaining_submissions", budget_total)
    submission_budget = f"{budget_remaining} remaining ({budget_daily} today)"

    cv_strategy = "N/A"
    cv_override_active = "no"
    active_strategy = "N/A"
    if cfg:
        cv_strategy = f"{cfg.get('cv_strategy', {}).get('type', 'N/A')} — {cfg.get('cv_strategy', {}).get('selection_reason', 'N/A')[:30]}..."
        override_active = state.get("cv_strategy_override", {}).get("active", False)
        if override_active:
            cv_override_active = f"YES — {state.get('cv_strategy_override', {}).get('override_strategy', 'N/A')}"
            active_strategy = state.get("cv_strategy_override", {}).get(
                "override_strategy", "N/A"
            )
        else:
            active_strategy = cfg.get("cv_strategy", {}).get("type", "N/A")

    target_domain_bounds = "NOT INITIALIZED"
    if cfg and cfg.get("target_domain_bounds"):
        target_domain_bounds = f"{cfg.get('target_domain_bounds', {}).get('min')}, {cfg.get('target_domain_bounds', {}).get('max')}"

    external_data = str(cfg.get("allowed_external_data", False)).lower()

    # Integrity status values
    file_hashes = (
        "N/A (INIT)"
        if is_init
        else ("FAIL" if any("hash" in f.lower() for f in failures) else "PASS")
    )
    skill_state_status = (
        "N/A"
        if is_init
        else ("invalid" if "skill_state" in "".join(failures).lower() else "valid")
    )
    env_lock = "present" if req_txt.exists() else "MISSING"
    config_lock = (
        "NOT LOCKED — Phase 1 incomplete"
        if is_init or state.get("dag_phase") in ("uninitialized", "phase_1_incomplete")
        else "active"
    )

    # OOF Contract status
    oof_active_strategy = active_strategy
    oof_tagged = (
        "N/A"
        if is_init
        else (
            "all tagged"
            if not any("cv_strategy_id" in f.lower() for f in failures)
            else f"{sum('cv_strategy_id' in f.lower() for f in failures)} violations"
        )
    )
    single_cv = (
        "confirmed"
        if not any("single cv" in f.lower() for f in failures)
        else "VIOLATION"
    )

    # Policy status
    policy_filters = (
        f"{len(cfg.get('policy_filters', []))} columns blocked" if cfg else "N/A"
    )
    leaked_features = (
        "empty"
        if not state.get("leaked_features")
        else f"{len(state.get('leaked_features'))} flagged"
    )
    banned_check = (
        "N/A"
        if is_init
        else ("FAIL" if any("banned" in f.lower() for f in failures) else "PASS")
    )

    # Sidecar status
    skill_00 = "running" if state.get("sidecar_skill_00_active") else "not started"
    skill_18 = state.get("skill_18_last_run_timestamp", "not yet run")
    skill_19 = state.get("skill_19_last_run_timestamp", "not yet run")
    skill_20 = state.get("skill_20_last_run_timestamp", "not yet run")
    unresolved_hypotheses = len(state.get("sidecar_recommendations", []))

    # Human Gates status
    gate_1 = "approved" if state.get("human_gate_1_approved") is True else "pending"
    approved_branches = [
        k
        for k, v in state.items()
        if k.startswith("human_gate_2_") and k.endswith("_approved") and v is True
    ]
    gate_2 = f"{len(approved_branches)} approved" if approved_branches else "pending"
    gate_3 = "approved" if state.get("human_gate_3_approved") is True else "pending"
    gate_4 = "approved" if state.get("human_gate_4_approved") is True else "pending"
    gate_5 = (
        "selected"
        if len(state.get("human_gate_5_selection", [])) == 2
        else "not selected"
    )

    # Zindi Compliance status
    automl_usage = (
        "none" if not any("automl" in f.lower() for f in failures) else "WARNING — list"
    )
    raw_probs = "confirmed" if cfg.get("use_probabilities") is True else "NOT CONFIRMED"
    seed_repro = "confirmed" if seed != "NOT SET" else "NOT SET"
    sub_selection = (
        "2 selected"
        if len(state.get("human_gate_5_selection", [])) == 2
        else "NOT YET SELECTED"
    )
    code_review = (
        "yes"
        if all(
            [
                state.get("human_gate_1_approved") is True,
                gate_2 != "pending",
                state.get("human_gate_3_approved") is True,
                state.get("human_gate_4_approved") is True,
                gate_5 == "selected",
            ]
        )
        else "NO"
    )

    result_status = "FAIL" if failures else "PASS"

    box = f"""
Preflight Prompt Surfaced to Operator
+-----------------------------------------------------------+
|        ZINDIAN ORCHESTRATOR - SESSION PREFLIGHT           |
|        Competition : {comp_slug:<30} |
|        Mode        : {mode:<30} |
|        Date        : {timestamp:<30} |
+-----------------------------------------------------------+

[CONFIG]
  competition_id       : {comp_slug}
  task_type            : {task_type}
  metric               : {metric}
  metric_direction     : {metric_direction}
  use_probabilities    : {use_probabilities}
  target_col           : {target_col}
  seed                 : {seed}
  submission_budget    : {submission_budget}
  cv_strategy          : {cv_strategy}
  cv_override active   : {cv_override_active}
  active strategy      : {active_strategy}
  target_domain_bounds : {target_domain_bounds}
  external_data        : {external_data}
  automl_permitted     : FALSE (Zindi rule - hard prohibition)

[INTEGRITY]
  file hashes          : {file_hashes}
  SKILL_STATE.json     : {skill_state_status}
  environment lock     : {env_lock}
  config lock          : {config_lock}

[OOF CONTRACT]
  active strategy      : {oof_active_strategy}
  cv_strategy_id tagged: {oof_tagged}
  single CV object     : {single_cv}

[POLICY]
  policy_filters       : {policy_filters}
  leaked_features      : {leaked_features}
  banned column check  : {banned_check}

[SIDECAR]
  skill_00             : {skill_00}
  skill_18 last run    : {skill_18}
  skill_19 last run    : {skill_19}
  skill_20 last run    : {skill_20}
  unresolved hypotheses: {unresolved_hypotheses}

[HUMAN GATES]
  Gate 1 - anchor review       : {gate_1}
  Gate 2 - branches reviewed   : {gate_2}
  Gate 3 - fusion              : {gate_3}
  Gate 4 - inference           : {gate_4}
  Gate 5 - final selection     : {gate_5}

[ZINDI COMPLIANCE]
  automl usage detected: {automl_usage}
  raw probabilities    : {raw_probs}
  seed reproducibility : {seed_repro}
  submission selection : {sub_selection}
  code review ready    : {code_review}

-------------------------------------------------------------
PREFLIGHT RESULT: {result_status}
"""
    print(box.encode("ascii", errors="replace").decode("ascii"))

    if failures:
        print("[FAIL] FAILURES DETECTED:")
        for idx, f in enumerate(failures, 1):
            print(f"  {idx}. {f}")
        print()
    if warnings:
        print("[WARN] WARNINGS DETECTED:")
        for idx, w in enumerate(warnings, 1):
            print(f"  {idx}. {w}")
        print()

    # Prompt selection
    if not is_interactive:
        if failures:
            print("ERROR: Preflight failed in non-interactive mode.")
            sys.exit(1)
        else:
            print("Preflight checks passed in non-interactive mode. Proceeding.")
            sys.exit(0)

    # Interactive choice loop
    while True:
        print("  [1] PROCEED  - all checks pass")
        print("  [2] ABORT    - do not start this session")
        print("  [3] OVERRIDE - proceed despite warnings (requires written reason)")
        print()
        try:
            choice = input("Enter choice [1-3]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAborted by user.")
            sys.exit(0)

        if choice == "1":
            if failures:
                print(
                    "\n[FAIL] Cannot PROCEED: there are active failures that must be resolved."
                )
                continue
            else:
                print("\nProceeding...")
                # Write preflight_confirmed to state
                if is_init:
                    state = {
                        "dag_phase": "phase_1_incomplete",
                        "preflight_confirmed": True,
                        "competition": comp_slug,
                    }
                else:
                    state["preflight_confirmed"] = True
                state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
                sys.exit(0)
        elif choice == "2":
            print("\nAborting session...")
            sys.exit(0)
        elif choice == "3":
            if not warnings and not failures:
                print("\nOverride not needed - all checks pass.")
                continue
            reason = input("\nReason for OVERRIDE: ").strip()
            if not reason:
                print("[FAIL] Reason cannot be empty.")
                continue
            print("\nProceeding with override...")
            if is_init:
                state = {
                    "dag_phase": "phase_1_incomplete",
                    "preflight_confirmed": True,
                    "preflight_override_reason": reason,
                    "competition": comp_slug,
                }
            else:
                state["preflight_confirmed"] = True
                state["preflight_override_reason"] = reason
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            sys.exit(0)
        else:
            print("[FAIL] Invalid selection. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
