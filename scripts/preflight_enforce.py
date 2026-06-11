#!/usr/bin/env python3
"""Preflight ENFORCE check for reproducibility and environment lock.

Checks (SoT Section 3 ENFORCE mode):
- requirements.in and requirements.txt present
- challenge_config.json contains all required SoT v2.0.1 schema fields
- SKILL_STATE.json contains the standard 5 human gate keys
- OOF cv_strategy_id tagging validation
- Cross-skill import static scan
- AutoML import static scan

Usage:
  python scripts/preflight_enforce.py --competition competitions/ey-frogs

Exits 0 on success, 1 on failure.
"""
from __future__ import annotations

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
# Static scans (run on source tree, not per-competition)
# ---------------------------------------------------------------------------

REQUIRED_CONFIG_FIELDS = {
    "reproducibility": ("reproducibility", dict),
    "reproducibility.seed": "seed",
    "cv_strategy": ("cv_strategy", dict),
    "cv_strategy.type": "type",
    "cv_strategy.n_splits": "n_splits",
    "cv_strategy.selection_reason": "selection_reason",
    "drift_threshold": ("drift_threshold", (int, float)),
    "use_probabilities": ("use_probabilities", bool),
    "metric_direction": ("metric_direction", str),
    "shap_leak_threshold": ("shap_leak_threshold", (int, float)),
    "variance_gate_threshold": ("variance_gate_threshold", (int, float)),
    "gate_margin": ("gate_margin", (int, float)),
}

VALID_METRIC_DIRECTIONS = {"maximize", "minimize"}


def check_config_completeness(cfg: dict) -> None:
    """Validate challenge_config.json fields per SoT v2.0.1 schema."""
    # Nested block checks
    for block_key, (key_path, expected_type) in [
        ("reproducibility", ("reproducibility", dict)),
        ("cv_strategy", ("cv_strategy", dict)),
    ]:
        block = cfg.get(key_path)
        if not block or not isinstance(block, expected_type):
            fail(f"challenge_config.json missing `{key_path}` block")
        ok(f"challenge_config.json `{key_path}` block present")

    # Nested field checks
    repro = cfg.get("reproducibility", {})
    seed = repro.get("seed")
    if seed is None or not isinstance(seed, int):
        fail("reproducibility.seed is missing or not an int")
    ok(f"reproducibility.seed: {seed}")

    cv = cfg.get("cv_strategy", {})
    for subfield in ("type", "n_splits", "selection_reason"):
        val = cv.get(subfield)
        if subfield == "n_splits" and (val is None or not isinstance(val, int)):
            fail(f"cv_strategy.{subfield} is missing or not an int")
        elif subfield != "n_splits" and not val:
            fail(f"cv_strategy.{subfield} is missing or empty")
    ok(f"cv_strategy block complete ({cv.get('type', '<unknown>')})")

    # Float threshold checks
    for name, (key_path, expected_type) in [
        ("drift_threshold", ("drift_threshold", (int, float))),
        ("shap_leak_threshold", ("shap_leak_threshold", (int, float))),
        ("variance_gate_threshold", ("variance_gate_threshold", (int, float))),
        ("gate_margin", ("gate_margin", (int, float))),
    ]:
        val = cfg.get(key_path)
        if val is None or not isinstance(val, expected_type):
            fail(f"challenge_config.json `{name}` is missing or not a number")
        ok(f"challenge_config.json `{name}`: {val}")

    # Bool checks
    use_probs = cfg.get("use_probabilities")
    if use_probs is None or not isinstance(use_probs, bool):
        fail("challenge_config.json `use_probabilities` is missing or not a bool")
    ok(f"challenge_config.json `use_probabilities`: {use_probs}")

    # Metric direction check
    direction = cfg.get("metric_direction")
    if not direction or not isinstance(direction, str):
        fail("challenge_config.json `metric_direction` is missing or not a string")
    if direction not in VALID_METRIC_DIRECTIONS:
        fail(f"challenge_config.json `metric_direction` must be one of {VALID_METRIC_DIRECTIONS}, got '{direction}'")
    ok(f"challenge_config.json `metric_direction`: {direction}")


HUMAN_GATE_KEYS: dict[str, type] = {
    "human_gate_1_approved": bool,
    "human_gate_2_by_branch": dict,
    "human_gate_3_approved": bool,
    "human_gate_4_approved": bool,
    "human_gate_5_selection": list,
}


def check_human_gate_keys(state: dict) -> None:
    """Validate standard 5 human gate keys and inner branch booleans."""
    missing = []
    for key, expected_type in HUMAN_GATE_KEYS.items():
        val = state.get(key)
        if val is None:
            missing.append(key)
        elif not isinstance(val, expected_type):
            fail(f"SKILL_STATE.{key} has wrong type: {type(val).__name__}, expected {expected_type.__name__}")

    if missing:
        fail(f"Missing required human gate keys: {missing}")

    # Validate inner branch booleans
    gate2 = state.get("human_gate_2_by_branch", {})
    for branch_key, branch_val in gate2.items():
        if not branch_key.endswith("_approved"):
            fail(f"human_gate_2_by_branch key '{branch_key}' does not end with '_approved'")
        if not isinstance(branch_val, bool):
            fail(f"human_gate_2_by_branch.{branch_key} must be a bool, got {type(branch_val).__name__}")

    ok("SKILL_STATE.json contains all 5 human gate keys with correct types")
    ok(f"human_gate_2_by_branch has {len(gate2)} branch entries, all bool")


def scan_oof_cv_strategy_tags(skills_dir: Path) -> None:
    """Scan skill modules that call write_oof_record() and assert cv_strategy_id is passed."""
    violations = []
    for skill_file in sorted(skills_dir.glob("skill_*.py")):
        code = skill_file.read_text()
        if "write_oof_record" not in code:
            continue
        # Check that cv_strategy_id= kwarg is passed in the call
        # Simple heuristic: find write_oof_record( calls and check for cv_strategy_id=
        calls = re.findall(r"write_oof_record\s*\([^)]*\)", code, re.DOTALL)
        for call in calls:
            if "cv_strategy_id=" not in call:
                violations.append(f"{skill_file.name} calls write_oof_record() without cv_strategy_id= kwarg")

    if violations:
        for v in violations:
            print(f"  WARN: {v}")
        fail(f"OOF cv_strategy_id tag violations found: {len(violations)}")
    ok("All write_oof_record() calls include cv_strategy_id= kwarg")


CROSS_SKILL_IMPORT_PATTERNS = [
    re.compile(r"from\s+\.skill_\d{2}"),
    re.compile(r"from\s+zindian\.skills\.skill_\d{2}"),
    re.compile(r"import\s+skill_\d{2}"),
]


def scan_cross_skill_imports(skills_dir: Path) -> None:
    """Flag any from .skill_NN_* or from zindian.skills.skill_NN_* import in a skill module."""
    violations = []
    for skill_file in sorted(skills_dir.glob("skill_*.py")):
        code = skill_file.read_text()
        for pattern in CROSS_SKILL_IMPORT_PATTERNS:
            matches = pattern.findall(code)
            for m in matches:
                violations.append(f"{skill_file.name}: {m.strip()}")

    if violations:
        for v in violations:
            print(f"  VIOLATION: {v}")
        fail(f"Cross-skill import violations: {len(violations)}")
    ok("No cross-skill imports detected")


AUTOML_PACKAGES = ["autosklearn", "tpot", "h2o", "autogluon", "auto-sklearn", "auto_ml", "mljar"]


def scan_automl_imports(skills_dir: Path) -> None:
    """Flag any AutoML library import in any skill body."""
    violations = []
    for skill_file in sorted(skills_dir.glob("*.py")):
        code = skill_file.read_text()
        for pkg in AUTOML_PACKAGES:
            pattern = re.compile(rf"(?:^|\n)\s*(?:import|from)\s+{re.escape(pkg)}")
            matches = pattern.findall(code)
            if matches:
                violations.append(f"{skill_file.name}: imports '{pkg}'")

    if violations:
        for v in violations:
            print(f"  VIOLATION: {v}")
        fail(f"AutoML import violations: {len(violations)}")
    ok("No AutoML library imports detected")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", default=None, help="Path to competition folder (e.g., competitions/ey-frogs)")
    args = parser.parse_args()

    root = Path.cwd()
    skills_dir = root / "zindian" / "skills"

    # Static scans (run once, not per-competition)
    scan_automl_imports(skills_dir)
    scan_cross_skill_imports(skills_dir)
    scan_oof_cv_strategy_tags(skills_dir)

    # ── Requirements files ──
    req_in = root / "requirements.in"
    req_txt = root / "requirements.txt"
    if not req_in.exists():
        fail("requirements.in missing at project root")
    ok("requirements.in present")

    if not req_txt.exists():
        fail("requirements.txt missing — run: pip-compile requirements.in --output-file requirements.txt")
    ok("requirements.txt present")

    # ── Competition directory ──
    if args.competition:
        comp_path = Path(args.competition)
    else:
        comps = [p for p in (root / "competitions").iterdir() if p.is_dir()]
        if not comps:
            fail("No competitions/ folder or empty")
        comp_path = comps[0]

    # ── challenge_config.json ──
    config_path = comp_path / "challenge_config.json"
    if not config_path.exists():
        fail(f"challenge_config.json not found in {comp_path}")

    cfg = json.loads(config_path.read_text())
    ok("challenge_config.json loaded")
    check_config_completeness(cfg)

    # ── SKILL_STATE.json ──
    state_path = comp_path / "SKILL_STATE.json"
    if not state_path.exists():
        fail(f"SKILL_STATE.json not found in {comp_path}")

    state = json.loads(state_path.read_text())
    ok("SKILL_STATE.json loaded")
    check_human_gate_keys(state)

    print("\nPRELIGHT ENFORCE: ALL CHECKS PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()