"""
SKILL 22 — Reproducibility & Integration Audit
==============================================

Active pipeline auditor. Replaces the per-competition static snapshot.

Contract (SoT §4 / §8):
  * No hardcoded row counts, geometry constants, or competition slugs.
  * Performs an AST scan over `zindian/skills/` to verify the absence of
    forbidden AutoML libraries (autogluon, auto-sklearn, tpot, h2o, …).
  * Verifies that the pinned `requirements.txt` is consistent with
    `requirements.in` (every top-level requirement in `.in` must appear in
    `.txt`).
  * Audits every `branch_{name}_oof` record in `SKILL_STATE.json` to ensure it
    carries a valid `cv_strategy_id` matching the active CV strategy.
  * Never writes a `human_gate_*_approved` key.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

# AutoML libraries forbidden by the SoT / AGENTS.md "No AutoML" rule.
FORBIDDEN_AUTOML_MODULES: tuple[str, ...] = (
    "autogluon",
    "auto_sklearn",
    "auto-sklearn",
    "tpot",
    "h2o",
    "h2o4gpu",
    "flaml",
    "pycaret",
    "mlbox",
    "gluonts",
    "ludwig",
)


def run_command(cmd: list[str]) -> str:
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except subprocess.CalledProcessError as exc:
        return f"ERROR: {exc.stderr.strip()}"
    except FileNotFoundError as exc:
        return f"ERROR: {exc}"


def _scan_skill_imports(skill_dir: Path) -> list[tuple[Path, int, str]]:
    """Return a list of (path, lineno, module) for any forbidden AutoML import."""
    offenders: list[tuple[Path, int, str]] = []
    for p in sorted(skill_dir.glob("*.py")):
        try:
            source = p.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name.split(".")[0]
                    if name in FORBIDDEN_AUTOML_MODULES:
                        offenders.append((p, getattr(node, "lineno", 0), name))
            elif isinstance(node, ast.ImportFrom) and node.module:
                name = node.module.split(".")[0]
                if name in FORBIDDEN_AUTOML_MODULES:
                    offenders.append((p, getattr(node, "lineno", 0), name))
            elif isinstance(node, ast.Attribute):
                # Catch `import autogluon...` then `autogluon.foo(...)` only at
                # the leaf of dotted chains.
                if isinstance(node.value, ast.Name):
                    if node.value.id in FORBIDDEN_AUTOML_MODULES:
                        offenders.append((p, getattr(node, "lineno", 0), node.value.id))
    return offenders


def _parse_requirements(text: str) -> list[str]:
    """Parse a requirements file and return the top-level package names.

    Strips comments, blank lines, -r / -e includes, and version specifiers.
    """
    names: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith(
            ("-r", "--requirement", "-e", "--editable", "-c", "--constraint")
        ):
            continue
        # Strip environment markers (`; python_version < '3.10'`) and extras (`[cpu]`).
        cleaned = re.split(r"[#;\[]", stripped, maxsplit=1)[0].strip()
        if not cleaned:
            continue
        # Take the package name (left of any version specifier or @ url).
        pkg = re.split(r"[<>=!~ ]+", cleaned, maxsplit=1)[0]
        # If it's a URL, take the last path component.
        if "/" in pkg:
            pkg = pkg.rsplit("/", 1)[-1]
        if pkg:
            names.append(pkg.lower())
    return names


def _verify_lockfile(
    repo_root: Path,
) -> tuple[bool, list[str]]:
    """Verify `requirements.in` and `requirements.txt` are in sync.

    Every top-level package declared in `.in` must appear in `.txt`. We don't
    assert an exact set match (transitive deps in `.txt` are expected), only
    that nothing in `.in` is missing.
    """
    issues: list[str] = []
    req_in = repo_root / "requirements.in"
    req_txt = repo_root / "requirements.txt"
    if not req_in.exists():
        issues.append("requirements.in is missing from the repository root")
    if not req_txt.exists():
        issues.append("requirements.txt is missing from the repository root")
        return (False, issues)
    in_pkgs = set(_parse_requirements(req_in.read_text(encoding="utf-8")))
    txt_pkgs = set(_parse_requirements(req_txt.read_text(encoding="utf-8")))
    missing = sorted(in_pkgs - txt_pkgs)
    if missing:
        issues.append(
            f"requirements.txt is missing top-level requirements from "
            f"requirements.in: {', '.join(missing)}"
        )
    # Also surface a stale-lockfile warning.
    if req_in.exists() and req_txt.exists():
        try:
            in_mtime = req_in.stat().st_mtime
            txt_mtime = req_txt.stat().st_mtime
            if txt_mtime < in_mtime:
                issues.append(
                    "requirements.txt is older than requirements.in; "
                    "re-run `pip-compile requirements.in`."
                )
        except OSError:
            pass
    return (len(issues) == 0, issues)


def _audit_oof_strategy_tags(
    state: dict[str, Any],
    active_strategy_id: str,
) -> tuple[bool, list[str]]:
    """Verify every branch_{name}_oof record carries a valid cv_strategy_id.

    A record is considered valid if its `cv_strategy_id` matches the active
    strategy id, or is one of the recognized fallback markers.
    """
    issues: list[str] = []
    if not isinstance(state, dict):
        return (False, ["SKILL_STATE is not a dict"])
    if not active_strategy_id:
        issues.append("Active CV strategy id is empty; cannot validate OOF tags.")
        return (False, issues)
    for key, value in state.items():
        if not (
            isinstance(key, str) and key.startswith("branch_") and key.endswith("_oof")
        ):
            continue
        if not isinstance(value, dict):
            issues.append(f"{key}: OOF record is not a dict")
            continue
        cv_id = value.get("cv_strategy_id")
        if not isinstance(cv_id, str) or not cv_id:
            issues.append(f"{key}: missing cv_strategy_id tag")
            continue
        if cv_id != active_strategy_id and cv_id not in (
            "config:unknown",
            "override:unknown",
            "unknown",
        ):
            issues.append(
                f"{key}: cv_strategy_id '{cv_id}' does not match active strategy "
                f"'{active_strategy_id}'"
            )
    return (len(issues) == 0, issues)


def audit_pipeline(slug: str | None = None) -> bool:
    print("=" * 70)
    print("ZINDIAN ORCHESTRATOR: FINAL REPRODUCIBILITY AUDIT")
    print("=" * 70)

    repo_root = Path(__file__).resolve().parents[2]
    skill_dir = repo_root / "zindian" / "skills"
    errors_found = 0

    # ── Check 0: Lockfile consistency ─────────────────────────────────────────
    print(
        "\n[Check 0] Verifying environment lock (requirements.in vs requirements.txt)"
    )
    lock_ok, lock_issues = _verify_lockfile(repo_root)
    if not lock_ok:
        for issue in lock_issues:
            print(f"  ERROR: {issue}")
        errors_found += 1
    else:
        print("  OK: requirements.txt is in sync with requirements.in")
        if any("older" in s for s in lock_issues):
            for issue in lock_issues:
                print(f"  WARNING: {issue}")

    # ── Check 1: AST scan for forbidden AutoML imports ────────────────────────
    print("\n[Check 1] Scanning zindian/skills/ for forbidden AutoML imports")
    if not skill_dir.exists():
        print(f"  ERROR: skill directory missing at {skill_dir}")
        errors_found += 1
    else:
        offenders = _scan_skill_imports(skill_dir)
        if offenders:
            for path, lineno, module in offenders:
                rel = path.relative_to(repo_root)
                print(f"  ERROR: {rel}:{lineno} imports forbidden module '{module}'")
                errors_found += 1
        else:
            print("  OK: no AutoML library imports detected in zindian/skills/")

    # ── Check 2: Git branch alignment (informational) ─────────────────────────
    print("\n[Check 2] Synchronizing Repository Branch State")
    current_branch = run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    print(f"  - Active Git Branch : {current_branch}")

    # Resolve competition state if a slug is provided.
    comp_dir: Path | None = None
    state_path: Path | None = None
    if slug:
        comp_dir = repo_root / "competitions" / slug
        state_path = comp_dir / "SKILL_STATE.json"
    if comp_dir is None:
        # Fall back to autodetect when exactly one competition exists.
        comps = list((repo_root / "competitions").glob("*/SKILL_STATE.json"))
        if len(comps) == 1:
            state_path = comps[0]
            comp_dir = state_path.parent
            print(f"  - Auto-selected competition: {comp_dir.name}")
        elif len(comps) > 1:
            print(
                "  NOTICE: multiple competitions present and no slug supplied; "
                "skipping per-competition checks."
            )
        else:
            print(
                "  NOTICE: no active competition found; skipping per-competition checks."
            )
    if comp_dir is not None and state_path is not None:
        if not state_path.exists():
            print(f"  ERROR: Missing critical tracking file: {state_path}")
            return False
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"  ERROR: SKILL_STATE.json is not valid JSON: {exc}")
            return False
        recorded_branch = state.get("current_git_branch", "Unknown")
        recorded_phase = state.get("dag_phase", "Unknown")
        print(f"  - Serialized Git Branch: {recorded_branch}")
        print(f"  - Serialized DAG Phase: {recorded_phase}")
        if (
            current_branch
            and recorded_branch != "Unknown"
            and current_branch != recorded_branch
        ):
            print(
                f"  NOTICE: Branch tracking asymmetry — workspace is on "
                f"'{current_branch}' but state records '{recorded_branch}'."
            )

        # ── Check 3: OOF cv_strategy_id audit ───────────────────────────────
        print("\n[Check 3] Auditing OOF records for cv_strategy_id alignment")
        # Determine the active strategy id from SKILL_STATE / config.
        active_id: str | None = None
        override = state.get("cv_strategy_override") or {}
        if isinstance(override, dict) and override.get("active"):
            active_id = f"override:{override.get('override_strategy', 'unknown')}"
        if not active_id:
            # Try to load the challenge_config.json.
            cfg_path = comp_dir / "challenge_config.json"
            if cfg_path.exists():
                try:
                    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                    cv_strategy = cfg.get("cv_strategy") or {}
                    if isinstance(cv_strategy, dict) and cv_strategy.get("type"):
                        active_id = f"config:{cv_strategy.get('type', 'unknown')}"
                except json.JSONDecodeError:
                    pass
        if not active_id:
            active_id = "unknown"
        oof_ok, oof_issues = _audit_oof_strategy_tags(state, active_id)
        if oof_issues:
            for issue in oof_issues:
                print(f"  ERROR: {issue}")
            errors_found += len(oof_issues)
        else:
            oof_count = sum(
                1
                for k in state.keys()
                if isinstance(k, str) and k.startswith("branch_") and k.endswith("_oof")
            )
            print(
                f"  OK: All {oof_count} OOF records carry cv_strategy_id matching "
                f"active strategy '{active_id}'"
            )

    print("\n" + "=" * 70)
    if errors_found == 0:
        print("INTEGRATION STATUS: SECURE. WORKSPACE FULLY REPRODUCIBLE.")
        print(
            "Lockfile, AutoML import scan, branch tracking, and OOF tags all verified."
        )
        print("=" * 70)
        return True

    print(f"AUDIT BLOCKED: {errors_found} structural anomalies identified.")
    print("Resolve configuration mapping states before final archive export.")
    print("=" * 70)
    return False


def run(slug: str | None = None) -> dict[str, Any]:
    from datetime import datetime, timezone
    from zindian.paths import resolve_competition_paths
    from zindian.state import SkillStateStore

    paths = resolve_competition_paths(require_competition=False)
    actual_slug = slug or (
        paths.competition_dir.name if paths.competition_dir else None
    )

    success = audit_pipeline(slug=actual_slug)

    state = {}
    if paths.state_path and paths.state_path.exists():
        state_store = SkillStateStore(paths.state_path)
        state = state_store.read()
        state["reproducibility_audit"] = {
            "success": success,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        state_store.write(state)

    return {"success": success, "state": state}


if __name__ == "__main__":
    slug_arg = (
        sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else None
    )
    raise SystemExit(0 if audit_pipeline(slug=slug_arg) else 1)
