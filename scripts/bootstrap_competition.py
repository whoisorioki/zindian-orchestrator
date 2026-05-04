#!/usr/bin/env python3
"""Bootstrap a competition folder under `competitions/<slug>/`.

Creates the directory tree, writes templates for `challenge_config.json` and
`SKILL_STATE.json` if missing, and optionally moves known root artifacts
into the competition scope. Designed to be safe: supports dry-run and
confirmation.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from datetime import datetime


ROOT = Path.cwd()
TEMPLATES = ROOT / "templates"


def write_json_atomic(path: Path, data: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def load_template(name: str) -> dict:
    p = TEMPLATES / name
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def find_candidates(root: Path) -> dict:
    candidates = {}
    # common dataset filenames
    names = [
        "Training_Data.csv",
        "Test.csv",
        "SampleSubmission.csv",
        "TerraClimate_output.tiff",
    ]
    for n in names:
        # check root and root/data/raw
        paths = [root / n, root / "data" / "raw" / n]
        for p in paths:
            if p.exists():
                candidates[n] = p
                break
    return candidates


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("slug", help="competition slug (e.g. ey-frogs)")
    parser.add_argument("--move-files", action="store_true", help="Move detected root files into competition data/raw/")
    parser.add_argument("--yes", action="store_true", help="Assume yes for confirmations (non-interactive)")
    args = parser.parse_args(argv)

    slug = args.slug
    comp_dir = ROOT / "competitions" / slug
    raw = comp_dir / "data" / "raw"
    processed = comp_dir / "data" / "processed"
    notebooks = comp_dir / "notebooks"
    reports = comp_dir / "reports"

    # create directories
    for d in (raw, processed, notebooks, reports):
        d.mkdir(parents=True, exist_ok=True)

    # write challenge_config.json if missing
    challenge_path = comp_dir / "challenge_config.json"
    if not challenge_path.exists():
        tpl = load_template("challenge_config_template.json") or {}
        tpl.setdefault("slug", slug)
        write_json_atomic(challenge_path, tpl)
        print(f"Wrote: {challenge_path}")
    else:
        print(f"Found existing: {challenge_path}")

    # write SKILL_STATE.json if missing
    state_path = comp_dir / "SKILL_STATE.json"
    if not state_path.exists():
        tpl = load_template("SKILL_STATE_template.json") or {}
        tpl["competition_slug"] = slug
        tpl.setdefault("dag_phase", "phase_1_integrity_locked")
        tpl.setdefault("last_updated", datetime.utcnow().isoformat() + "Z")
        write_json_atomic(state_path, tpl)
        print(f"Wrote: {state_path}")
    else:
        print(f"Found existing: {state_path}")

    # discovery of candidate files to move
    candidates = find_candidates(ROOT)
    if not candidates:
        print("No known root artifacts found to move.")
    else:
        print("Detected candidate files to move into competition scope:")
        for name, path in candidates.items():
            print(f" - {name}: {path}")

        if args.move_files:
            if not args.yes:
                resp = input("Apply moves? Type YES to proceed: ")
                if resp.strip().upper() != "YES":
                    print("Aborting moves.")
                    return 0
            # perform moves
            for name, path in candidates.items():
                dest = raw / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                print(f"Moving {path} -> {dest}")
                shutil.move(str(path), str(dest))
            print("Moves complete.")
        else:
            print("Run with --move-files to move detected artifacts into the competition folder.")

    print(f"Bootstrap complete for slug: {slug}")
    print(f"Competition folder: {comp_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
