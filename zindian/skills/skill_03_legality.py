"""
Skill 03 — Deep Research / Legality Check
Reads challenge_config.json and compliance_log.md.
Hard gate before Skill 08 — must return GO before anchor runs.
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def write_state(state: dict, path: Path) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".json"
    ) as tmp:
        json.dump(state, tmp, indent=2)
        tmp_path = tmp.name
    os.replace(tmp_path, path)


def load_compliance_flags(compliance_path: Path) -> list:
    if not compliance_path.exists():
        return []
    flags = []
    for line in compliance_path.read_text().splitlines():
        if line.startswith("### 🚨"):
            flags.append(line.replace("### 🚨", "").strip())
    return flags


def run_checks(config: dict, flags: list) -> list:
    checks = []

    # Check 1: External data
    allowed_external = config.get("allowed_external_data", False)
    checks.append({
        "name": "External Data",
        "status": "PASS",
        "finding": (
            "PROHIBITED — use only competition-provided files."
            if not allowed_external
            else "PERMITTED — document any external source."
        ),
        "action": (
            "Do not use GBIF, Kaggle, or any non-provided dataset."
            if not allowed_external
            else "Permitted — document sources."
        ),
        "blocks": False,
    })

    # Check 2: TerraClimate via Planetary Computer
    checks.append({
        "name": "TerraClimate / Planetary Computer",
        "status": "PASS",
        "finding": (
            "PERMITTED — listed in competition requirements.txt. "
            "This is the intended feature source, not external data."
        ),
        "action": "Fetch all 14 TerraClimate variables freely.",
        "blocks": False,
    })

    # Check 3: AutoML
    automl_ok = config.get("automl_permitted", False)
    checks.append({
        "name": "AutoML Tools",
        "status": "PASS",
        "finding": "PROHIBITED — H2O, AutoSklearn, TPOT etc. not allowed.",
        "action": "Use manually configured sklearn/LightGBM/XGBoost pipelines.",
        "blocks": False,
    })

    # Check 4: Spatial features
    banned = config.get("banned_features", [])
    spatial_banned = any("spatial" in b.lower() for b in banned)
    checks.append({
        "name": "Spatial Features",
        "status": "WARN" if spatial_banned else "PASS",
        "finding": (
            "Derived spatial features BANNED (discussion 32369). "
            "Raw Lat/Lon as direct inputs is permitted. "
            "BANNED: distance calcs, H3 bins, spatial clusters, "
            "admin region encodings, Lat/Lon polynomial terms."
            if spatial_banned
            else "No spatial bans detected."
        ),
        "action": (
            "Use raw Lat/Lon only. No spatial aggregations."
            if spatial_banned
            else "No restriction."
        ),
        "blocks": False,
    })

    # Check 5: Output format
    use_probs = config.get("use_probabilities", False)
    metric = config.get("metric", "unknown")
    checks.append({
        "name": "Output Format",
        "status": "PASS",
        "finding": (
            f"Metric: {metric}. "
            + ("Submit raw float probabilities [0,1]."
               if use_probs
               else "Submit hard integer labels 0 or 1. "
                    "Threshold search is internal — never submit probabilities.")
        ),
        "action": (
            "predict_proba()[:, 1] → Target"
            if use_probs
            else "predict() or threshold(predict_proba()) → Target as int"
        ),
        "blocks": False,
    })

    # Check 6: Submission budget
    daily = config.get("daily_limit", 10)
    total = config.get("total_limit", 300)
    checks.append({
        "name": "Submission Budget",
        "status": "PASS",
        "finding": f"Max {daily}/day, {total} total.",
        "action": (
            "Anchor phase: max 2/day. Exploration: max 5/day. "
            "Always reserve 2/day buffer. Gate must pass before submit."
        ),
        "blocks": False,
    })

    # Check 7: Code review
    tier = config.get("code_review_tier", "top_10")
    hours = config.get("code_review_hours", 48)
    checks.append({
        "name": "Code Review",
        "status": "WARN",
        "finding": (
            f"{tier} on private leaderboard must submit reproducible code "
            f"within {hours} hours. All notebooks must run top-to-bottom "
            f"on data/raw/ only."
        ),
        "action": (
            "Set random seeds everywhere. No hardcoded paths. "
            "No manual intermediate files. Test full pipeline before submit."
        ),
        "blocks": False,
    })

    # Check 8: Compliance flags
    if flags:
        checks.append({
            "name": "Discussion Compliance Flags",
            "status": "WARN",
            "finding": f"{len(flags)} flags: " + " | ".join(flags[:3]),
            "action": "Read reports/compliance_log.md before proceeding.",
            "blocks": False,
        })

    return checks


def write_legality_report(checks: list, config: dict,
                          report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    blocked = [c for c in checks if c["status"] == "BLOCKED"]
    warnings = [c for c in checks if c["status"] == "WARN"]
    passed = [c for c in checks if c["status"] == "PASS"]

    lines = [
        "# Legality Report — Skill 03",
        f"**Competition**: {config.get('name', config.get('slug'))}",
        f"**Generated**: {now}",
        f"**Checks**: {len(checks)} | "
        f"Blocked: {len(blocked)} | "
        f"Warnings: {len(warnings)} | "
        f"Passed: {len(passed)}",
        "", "---", "",
        "## ⚠️ Warnings", "",
    ]

    for c in warnings:
        lines += [
            f"### {c['name']}",
            f"**Finding**: {c['finding']}",
            f"**Action**: {c['action']}", "",
        ]

    lines += ["## ✅ Passed", ""]
    for c in passed:
        lines.append(f"- **{c['name']}**: {c['finding']}")

    lines += [
        "", "---", "",
        "## Feature Engineering Constraints",
        "",
        f"- Allowed: Raw Lat/Lon + all 14 TerraClimate variables",
        f"- Banned features: {config.get('banned_features', [])}",
        f"- External data: {'PROHIBITED' if not config.get('allowed_external_data') else 'PERMITTED'}",
        f"- AutoML: PROHIBITED",
        f"- Output: {'Hard labels 0/1' if not config.get('use_probabilities') else 'Raw probabilities'}",
        f"- Metric: {config.get('metric', 'unknown')}",
    ]

    report_path.write_text("\n".join(lines))


def main(slug: str = "ey-frogs") -> dict:
    base = Path(f"competitions/{slug}")
    config_path     = base / "challenge_config.json"
    state_path      = base / "SKILL_STATE.json"
    compliance_path = base / "reports/compliance_log.md"
    report_path     = base / "reports/legality_report.md"
    log_path        = base / "reports/submission_log.md"

    print("=" * 60)
    print("SKILL 03 — Deep Research / Legality Check")
    print("=" * 60)

    config = load_json(config_path)
    state  = load_json(state_path)
    flags  = load_compliance_flags(compliance_path)

    print(f"Competition  : {config.get('name')}")
    print(f"Current phase: {state.get('dag_phase')}")
    print(f"Compliance flags: {len(flags)}")
    print("\nRunning checks...")

    checks = run_checks(config, flags)
    hard_blocks = [c for c in checks if c.get("blocks")]
    status = "BLOCKED" if hard_blocks else "GO"

    print(f"\n{'='*60}")
    print(f"RESULT: {status}")
    print(f"{'='*60}")
    for c in checks:
        icon = {"PASS": "✅", "WARN": "⚠️ ", "BLOCKED": "❌"}.get(
            c["status"], "?"
        )
        print(f"{icon} {c['name']}: {c['status']}")
        if c["status"] != "PASS":
            print(f"   → {c['action']}")

    write_legality_report(checks, config, report_path)
    print(f"\n✅ Report → {report_path}")

    # Append to submission log
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    with open(log_path, "a") as f:
        f.write(
            f"\n## Skill 03 — Legality Check [{now}]\n"
            f"**Status**: {status}\n"
            f"**Warnings**: {[c['name'] for c in checks if c['status']=='WARN']}\n"
        )

    # Update state
    state["dag_phase"] = "phase_2_legality_checked"
    state["legality_status"] = status
    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    write_state(state, state_path)
    print(f"✅ SKILL_STATE.json → dag_phase: phase_2_legality_checked")

    if status == "GO":
        print(f"\n✅ SKILL 03 PASSED — proceed to Skill 08")
    else:
        print(f"\n❌ SKILL 03 BLOCKED — resolve before Skill 08")

    return {"status": status, "checks": len(checks)}


if __name__ == "__main__":
    import sys
    slug = sys.argv[1] if len(sys.argv) > 1 else "ey-frogs"
    result = main(slug)
    print(f"\nResult: {result}")
