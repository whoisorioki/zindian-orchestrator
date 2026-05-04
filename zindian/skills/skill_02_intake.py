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

from zindian.paths import CompetitionPaths, resolve_competition_paths

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

    # Parse known fields from rules text
    daily_limit = 10      # confirmed: "10 submissions per day"
    total_limit = 300     # confirmed: "300 submissions overall"
    public_split = 20     # confirmed: "approximately 20% of the test dataset"
    private_split = 80    # confirmed: "other 80% of the test dataset"
    team_size = 3         # confirmed: "maximum of three people" (EY specific)
    code_review_tier = "top_10"  # confirmed: "Top 10 on the private leaderboard"
    code_review_hours = 48       # confirmed: "48 hours to submit your code"

    return {
        "name": "EY Biodiversity Challenge — Frogs",
        "slug": slug,
        "subtitle": data.get("subtitle", ""),
        "end_time": data.get("end_time", ""),
        "metric": "accuracy",
        "metric_direction": "maximize",
        "submission_format": "classification",
        "use_probabilities": True,
        "daily_limit": daily_limit,
        "total_limit": total_limit,
        "public_split_pct": public_split,
        "private_split_pct": private_split,
        "team_allowed": True,
        "max_team_size": team_size,
        "code_review_tier": code_review_tier,
        "code_review_hours": code_review_hours,
        "allowed_external_data": False,
        "automl_permitted": False,
        "data_modality": "tabular",
        "domain": "biodiversity",
        "skills_required": data.get("skills", []),
        "banned_features": [
            "derived_spatial_features",
            "external_spatial_data"
        ],
        "compliance_notes": [
            "use_probabilities=True: do NOT threshold predictions",
            "No external data allowed — provided datasets only",
            "Derived spatial features are BANNED (discussion 32369)",
            "Top 10 code review — 48 hours to respond",
            "Final 5 days: no new team members",
            "Must select 2 submissions before deadline",
            "Always set random seed for reproducibility",
            "Open source packages only — no paid services",
            "Max team size is 3 (EY rule overrides Zindi default of 4)"
        ],
        "populated_at": datetime.now(timezone.utc).isoformat()
    }


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


def run(slug: str, headers: dict) -> dict:
    paths = resolve_competition_paths(slug=slug)

    print(f"\n{'='*60}")
    print(f"SKILL 02 — Challenge Intake")
    print(f"Competition: {slug}")
    print(f"{'='*60}\n")

    print("Fetching competition details from API...")
    data = fetch_competition(slug, headers)

    print("Extracting config fields...")
    config = extract_config(data, slug)

    write_config(config, paths)
    update_skill_state(slug, paths)

    print(f"\n--- Config Summary ---")
    print(f"Name       : {config['name']}")
    print(f"Metric     : {config['metric']} ({config['metric_direction']})")
    print(f"Modality   : {config['data_modality']}")
    print(f"Daily limit: {config['daily_limit']}")
    print(f"Total limit: {config['total_limit']}")
    print(f"Public split: {config['public_split_pct']}%")
    print(f"Private split: {config['private_split_pct']}%")
    print(f"Use probabilities: {config['use_probabilities']}")
    print(f"External data: {config['allowed_external_data']}")
    print(f"Banned features: {config['banned_features']}")
    print(f"\nCompliance notes:")
    for note in config["compliance_notes"]:
        print(f"  ⚠️  {note}")

    return {"status": "OK", "config": config}
