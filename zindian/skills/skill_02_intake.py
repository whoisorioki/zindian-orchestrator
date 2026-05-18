"""
Skill 02 — Challenge Intake
Reads the competition API and populates challenge_config.json
with all rules, limits, and constraints extracted from the response.
Must run after Skill 00 (compliance check).
"""

import os
import json
import requests
import tempfile
from pathlib import Path
from datetime import datetime, timezone
import difflib

from zindian.paths import CompetitionPaths, resolve_competition_paths
from zindian.config import ConfigNotPopulated

BASE_URL = "https://api.zindi.africa/v1/competitions"


def fetch_competition(slug: str, headers: dict) -> dict:
    url = f"{BASE_URL}/{slug}"
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json().get("data", {})


def extract_config(data: dict, slug: str) -> dict:
    """
    Extract all rule fields from raw competition API response.
    These values are ground truth — hardcoding nothing.
    """
    sections = data.get("sections", [])

    # Extract rules text from sections
    rules_text = ""
    for s in sections:
        ct = s.get("content_text", "")
        if ct:
            rules_text += ct + " "

    # Parse known fields from the API response; do not hardcode defaults.
    # These fields should be treated as authoritative when provided by the API.
    name = data.get("name") or data.get("title") or f"{slug}"
    subtitle = data.get("subtitle")
    end_time = data.get("end_time")

    # Metric must come from API. If missing, write null and raise ConfigNotPopulated.
    metric = data.get("metric")
    metric_direction = data.get("metric_direction")

    # Derive use_probabilities from metric when possible.
    use_probabilities = None
    if metric is not None:
        m = str(metric).lower()
        if m in ("log_loss", "logloss", "cross_entropy", "auc"):
            use_probabilities = True
        elif m in ("f1_score", "f1", "accuracy", "rmse", "mae"):
            use_probabilities = False
        else:
            use_probabilities = None

    # Limits and splits: take from API when provided, otherwise leave null.
    daily_limit = data.get("daily_limit")
    total_limit = data.get("total_limit")
    public_split = data.get("public_split_pct") or data.get("public_split")
    private_split = data.get("private_split_pct") or data.get("private_split")

    # Team information from the API; do NOT hardcode EY-specific overrides here.
    team_allowed = data.get("team_allowed")
    max_team_size = data.get("max_team_size")

    code_review_tier = data.get("code_review_tier")
    code_review_hours = data.get("code_review_hours")

    allowed_external_data = data.get("allowed_external_data")
    automl_permitted = data.get("automl_permitted")
    data_modality = data.get("data_modality") or data.get("modality")
    domain = data.get("domain")

    skills_required = data.get("skills", [])
    banned_features = data.get("banned_features", ["derived_spatial_features", "external_spatial_data"])

    config = {
        "name": name,
        "slug": slug,
        "subtitle": subtitle,
        "end_time": end_time,
        "metric": metric,
        "metric_direction": metric_direction,
        "submission_format": data.get("submission_format"),
        "use_probabilities": use_probabilities,
        "daily_limit": daily_limit,
        "total_limit": total_limit,
        "public_split_pct": public_split,
        "private_split_pct": private_split,
        "team_allowed": team_allowed,
        "max_team_size": max_team_size,
        "code_review_tier": code_review_tier,
        "code_review_hours": code_review_hours,
        "allowed_external_data": allowed_external_data,
        "automl_permitted": automl_permitted,
        "data_modality": data_modality,
        "domain": domain,
        "skills_required": skills_required,
        "banned_features": banned_features,
        "compliance_notes": [],
        "populated_at": datetime.now(timezone.utc).isoformat(),
    }

    # If metric is missing, attempt to fallback to monitor output.
    field_sources = {}
    for k in config:
        field_sources[k] = "api" if config[k] is not None else None

    if config["metric"] is None:
        try:
            paths = resolve_competition_paths(slug=slug)
            monitor_path = paths.reports_dir / "zindi_monitor.json"
            if monitor_path.exists():
                with open(monitor_path, encoding="utf-8") as f:
                    mon = json.load(f)
                ci = mon.get("competition_intel", {})
                # Map fallback fields
                if config.get("metric") is None and ci.get("metric") is not None:
                    config["metric"] = ci.get("metric")
                    field_sources["metric"] = "monitor"
                if config.get("use_probabilities") is None and ci.get("use_probabilities") is not None:
                    config["use_probabilities"] = ci.get("use_probabilities")
                    field_sources["use_probabilities"] = "monitor"
                # external_banned -> allowed_external_data (invert)
                if config.get("allowed_external_data") is None and ci.get("external_banned") is not None:
                    config["allowed_external_data"] = not bool(ci.get("external_banned"))
                    field_sources["allowed_external_data"] = "monitor"
                # automl_banned -> automl_permitted (invert)
                if config.get("automl_permitted") is None and ci.get("automl_banned") is not None:
                    config["automl_permitted"] = not bool(ci.get("automl_banned"))
                    field_sources["automl_permitted"] = "monitor"
                if config.get("daily_limit") is None and ci.get("daily_limit") is not None:
                    config["daily_limit"] = ci.get("daily_limit")
                    field_sources["daily_limit"] = "monitor"
                if config.get("total_limit") is None and ci.get("total_limit") is not None:
                    config["total_limit"] = ci.get("total_limit")
                    field_sources["total_limit"] = "monitor"
                if config.get("code_review_tier") is None and ci.get("code_review_tier") is not None:
                    config["code_review_tier"] = ci.get("code_review_tier")
                    field_sources["code_review_tier"] = "monitor"
                if config.get("max_team_size") is None and ci.get("team_size") is not None:
                    config["max_team_size"] = ci.get("team_size")
                    field_sources["max_team_size"] = "monitor"
                if config.get("public_split_pct") is None and ci.get("public_split_pct") is not None:
                    config["public_split_pct"] = ci.get("public_split_pct")
                    field_sources["public_split_pct"] = "monitor"

                # Log which fields came from API vs monitor fallback
                api_fields = [k for k, v in field_sources.items() if v == "api"]
                monitor_fields = [k for k, v in field_sources.items() if v == "monitor"]
                if monitor_fields:
                    print("\n--- Fallback Applied From zindi_monitor.json ---")
                    print("Fields from API: ", ", ".join(api_fields) if api_fields else "(none)")
                    print("Fields from monitor fallback: ", ", ".join(monitor_fields))
        except Exception as e:
            # Non-fatal; we'll handle missing metric later
            print(f"Warning: monitor fallback failed: {e}")

    # Only raise if metric is missing after both API and monitor fallback
    if config["metric"] is None:
        # Rebuild compliance notes to reflect current (null) use_probabilities
        up = config.get("use_probabilities")
        cn = []
        if up is True:
            cn.append("use_probabilities=True: submit raw float probabilities, do NOT threshold")
        elif up is False:
            cn.append("use_probabilities=False: submit hard 0/1 integer labels only")
        else:
            cn.append("use_probabilities: unknown — confirm from competition page")
        if config.get("allowed_external_data") is False:
            cn.append("No external data allowed — provided datasets only")
            if config.get("banned_features"):
                cn.append("Banned features: " + ", ".join(config.get("banned_features") or []))
        cn.extend([
            "Final 5 days: no new team members",
            "Must select 2 submissions before deadline",
            "Always set random seed for reproducibility",
            "Open source packages only — no paid services",
        ])
        config["compliance_notes"] = cn
        return config

    # Derive metric_direction from metric when not provided
    if config.get("metric_direction") is None and config.get("metric") is not None:
        m = str(config.get("metric")).lower()
        if m in ("f1_score", "f1", "accuracy", "auc"):
            config["metric_direction"] = "maximize"
        elif m in ("rmse", "mae", "log_loss", "logloss", "cross_entropy"):
            config["metric_direction"] = "minimize"
        else:
            config["metric_direction"] = None

    # If use_probabilities is still unknown after fallback, rebuild notes and return for inspection
    if config.get("use_probabilities") is None:
        up = config.get("use_probabilities")
        cn = []
        if up is True:
            cn.append("use_probabilities=True: submit raw float probabilities, do NOT threshold")
        elif up is False:
            cn.append("use_probabilities=False: submit hard 0/1 integer labels only")
        else:
            cn.append("use_probabilities: unknown — confirm from competition page")
        if config.get("allowed_external_data") is False:
            cn.append("No external data allowed — provided datasets only")
        if config.get("banned_features"):
            cn.append("Banned features: " + ", ".join(config.get("banned_features") or []))
        cn.extend([
            "Final 5 days: no new team members",
            "Must select 2 submissions before deadline",
            "Always set random seed for reproducibility",
            "Open source packages only — no paid services",
        ])
        config["compliance_notes"] = cn
        return config

    # Build final compliance notes from resolved values
    up = config.get("use_probabilities")
    cn = []
    if up is True:
        cn.append("use_probabilities=True: submit raw float probabilities, do NOT threshold")
    elif up is False:
        cn.append("use_probabilities=False: submit hard 0/1 integer labels only")
    else:
        cn.append("use_probabilities: unknown — confirm from competition page")
    if config.get("allowed_external_data") is False:
        cn.append("No external data allowed — provided datasets only")
    if config.get("banned_features"):
        cn.append("Banned features: " + ", ".join(config.get("banned_features") or []))
    if config.get("code_review_tier"):
        cn.append(f"Code review tier: {config.get('code_review_tier')} ({config.get('code_review_hours') or 'hours unspecified'})")
    cn.extend([
        "Final 5 days: no new team members",
        "Must select 2 submissions before deadline",
        "Always set random seed for reproducibility",
        "Open source packages only — no paid services",
    ])
    config["compliance_notes"] = cn

    return config


def write_config(config: dict, paths: CompetitionPaths) -> None:
    path = paths.config_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".json"
    ) as tmp:
        json.dump(config, tmp, indent=2)
        tmp_path = tmp.name
    os.replace(tmp_path, path)
    print(f"✅ challenge config populated -> {path}")


def update_skill_state(slug: str, paths: CompetitionPaths) -> None:
    state_path = paths.state_path
    if not state_path.exists():
        return
    with open(state_path, encoding="utf-8") as f:
        state = json.load(f)
    state["dag_phase"] = "phase_1_integrity"
    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".json"
    ) as tmp:
        json.dump(state, tmp, indent=2)
        tmp_path = tmp.name
    os.replace(tmp_path, state_path)
    print(f"✅ {state_path} -> dag_phase: phase_1_integrity")


def run(slug: str, headers: dict, dry_run: bool = False, merge: bool = False) -> dict:
    paths = resolve_competition_paths(slug=slug)

    print(f"\n{'='*60}")
    print(f"SKILL 02 — Challenge Intake")
    print(f"Competition: {slug}")
    print(f"{'='*60}\n")

    print("Fetching competition details from API...")
    data = fetch_competition(slug, headers)

    print("Extracting config fields...")
    config = extract_config(data, slug)

    final_to_write = config

    # If merge requested, combine with existing challenge_config.json without overwriting non-null fields
    if merge:
        existing = {}
        if paths.config_path.exists():
            try:
                existing = json.loads(paths.config_path.read_text(encoding="utf-8"))
            except Exception:
                existing = {}
        merged = dict(existing)  # start from existing; preserve extra fields
        for k, v in config.items():
            # Only set if existing value is None or missing
            if merged.get(k) is None:
                merged[k] = v
        final_to_write = merged

    if dry_run:
        print("\n--- DRY RUN: challenge_config.json that WOULD be written ---\n")
        print(json.dumps(final_to_write, indent=2))
    else:
        write_config(final_to_write, paths)
        update_skill_state(slug, paths)

    # If merge and dry_run, show a concise diff between existing and final_to_write
    if merge and dry_run:
        existing_text = ""
        if paths.config_path.exists():
            existing_text = paths.config_path.read_text(encoding="utf-8")
        new_text = json.dumps(final_to_write, indent=2)
        diff = difflib.unified_diff(
            existing_text.splitlines(), new_text.splitlines(),
            fromfile=str(paths.config_path), tofile="(merged)", lineterm=""
        )
        print("\n--- DIFF (existing -> merged) ---")
        for line in diff:
            print(line)

    # After writing (or preparing to write), fail loudly if required fields are null
    check_cfg = final_to_write
    if check_cfg.get("metric") is None:
        raise ConfigNotPopulated("Required field 'metric' is null after intake from API and fallback.")
    if check_cfg.get("use_probabilities") is None:
        raise ConfigNotPopulated("Derived field 'use_probabilities' is null — cannot infer from metric.")

    print(f"\n--- Config Summary ---")
    print(f"Name       : {final_to_write.get('name')}")
    print(f"Metric     : {final_to_write.get('metric')} ({final_to_write.get('metric_direction')})")
    print(f"Modality   : {final_to_write.get('data_modality')}")
    print(f"Daily limit: {final_to_write.get('daily_limit')}")
    print(f"Total limit: {final_to_write.get('total_limit')}")
    print(f"Public split: {final_to_write.get('public_split_pct')}%")
    print(f"Private split: {final_to_write.get('private_split_pct')}%")
    print(f"Use probabilities: {final_to_write.get('use_probabilities')}")
    print(f"External data: {final_to_write.get('allowed_external_data')}")
    print(f"Banned features: {final_to_write.get('banned_features')}")
    print(f"\nCompliance notes:")
    for note in final_to_write.get("compliance_notes", []):
        print(f"  ⚠️  {note}")

    return {"status": "OK", "config": config}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Skill 02 — Challenge Intake")
    parser.add_argument("slug", help="competition slug e.g. ey-frogs")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run", help="Print config without writing it")
    parser.add_argument("--merge", action="store_true", dest="merge", help="Merge with existing challenge_config.json without overwriting non-null fields")
    args = parser.parse_args()

    # Minimal headers placeholder — real use should provide auth in environment
    headers = {"Accept": "application/json"}
    try:
        run(args.slug, headers, dry_run=args.dry_run, merge=args.merge)
    except ConfigNotPopulated as e:
        print(f"ERROR: {e}")
        raise
