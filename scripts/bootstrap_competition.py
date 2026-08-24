#!/usr/bin/env python3
"""Bootstrap a competition folder under `competitions/<slug>/`.

Creates the directory tree, writes templates for `challenge_config.json` and
`SKILL_STATE.json` if missing, and optionally moves known root artifacts
into the competition scope. Designed to be safe: supports dry-run and
confirmation.
"""

import argparse
import json
from pathlib import Path
import shutil
from datetime import datetime, timezone


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


ROOT = _repo_root()
TEMPLATES = ROOT / "templates"


def sanitize_slug(slug: str) -> str:
    """Normalize the slug: lowercase, replace underscores/spaces with hyphens, collapse consecutive hyphens, keep alphanumeric and hyphens."""
    import re

    s = slug.strip().lower().replace("_", "-").replace(" ", "-")
    s = re.sub(r"[^a-z0-9\-]", "", s)
    s = re.sub(r"-+", "-", s)
    return s


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
    parser.add_argument(
        "--name",
        help="Human-readable name of the competition",
    )
    parser.add_argument(
        "--move-files",
        action="store_true",
        help="Move detected root files into competition data/raw/",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Assume yes for confirmations (non-interactive)",
    )
    args = parser.parse_args(argv)

    slug = sanitize_slug(args.slug)
    name = args.name or slug.replace("-", " ").title()
    comp_dir = ROOT / "competitions" / slug
    raw = comp_dir / "data" / "raw"
    processed = comp_dir / "data" / "processed"
    notebooks = comp_dir / "notebooks"
    reports = comp_dir / "reports"
    submissions = comp_dir / "submissions"

    # create directories
    for d in (raw, processed, notebooks, reports, submissions):
        d.mkdir(parents=True, exist_ok=True)

    # write challenge_config.json if missing
    challenge_path = comp_dir / "challenge_config.json"
    if not challenge_path.exists():
        tpl = load_template("challenge_config_template.json") or {}
        tpl["slug"] = slug
        tpl["name"] = name
        write_json_atomic(challenge_path, tpl)
        print(f"Wrote: {challenge_path}")
    else:
        print(f"Found existing: {challenge_path}")

    # write SKILL_STATE.json if missing
    state_path = comp_dir / "SKILL_STATE.json"
    if not state_path.exists():
        tpl = load_template("SKILL_STATE_template.json") or {}
        # ── Pre-Phase-1 seeds: everything needed before skill_01 runs ──
        tpl["competition"] = name
        tpl["competition_slug"] = slug
        tpl["dag_phase"] = "phase_1_integrity_locked"
        tpl["last_updated"] = datetime.now(timezone.utc).isoformat()
        # Resolve current git branch
        try:
            import subprocess

            branch = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                cwd=str(ROOT),
            )
            if branch.returncode == 0:
                tpl["current_git_branch"] = branch.stdout.strip()
        except Exception:
            pass
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
            print(
                "Run with --move-files to move detected artifacts into the competition folder."
            )

    # Update .env with the competition slug
    _update_env_slug(slug, name)

    print(f"Bootstrap complete for slug: {slug}")
    print(f"Competition folder: {comp_dir}")
    return 0


def _update_env_slug(slug: str, name: str) -> None:
    """Update ZINDIAN_COMPETITION_SLUG and ZINDIAN_COMPETITION_NAME in .env."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        print("  ⚠️  No .env file found — skipping env update")
        return

    lines = env_path.read_text(encoding="utf-8").splitlines()
    new_lines = []

    slug_updated = False
    name_updated = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("ZINDIAN_COMPETITION_SLUG=") or stripped.startswith(
            "COMPETITION_SLUG="
        ):
            new_lines.append(f"ZINDIAN_COMPETITION_SLUG={slug}")
            slug_updated = True
        elif stripped.startswith("ZINDIAN_COMPETITION="):
            new_lines.append(f"ZINDIAN_COMPETITION={slug}")
            slug_updated = True
        elif stripped.startswith("ZINDIAN_COMPETITION_NAME=") or stripped.startswith(
            "COMPETITION_NAME="
        ):
            new_lines.append(f"ZINDIAN_COMPETITION_NAME={name}")
            name_updated = True
        else:
            new_lines.append(line)

    if not slug_updated:
        new_lines.append(f"ZINDIAN_COMPETITION_SLUG={slug}")
    if not name_updated:
        new_lines.append(f"ZINDIAN_COMPETITION_NAME={name}")

    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    print(f"  ✅ .env ZINDIAN_COMPETITION_SLUG set to: {slug}")
    print(f"  ✅ .env ZINDIAN_COMPETITION_NAME set to: {name}")


if __name__ == "__main__":
    raise SystemExit(main())
