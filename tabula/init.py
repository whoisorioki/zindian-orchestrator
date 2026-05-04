from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path.cwd()
TEMPLATES = ROOT / "templates"


def write_json_atomic(path: Path, data: dict, *, dry_run: bool = False) -> None:
    if dry_run:
        print(f"[dry-run] write {path}")
        return

    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def load_template(name: str) -> dict:
    p = TEMPLATES / name
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def ensure_dirs(paths: Iterable[Path], *, dry_run: bool = False) -> None:
    for d in paths:
        if dry_run:
            print(f"[dry-run] mkdir -p {d}")
        else:
            d.mkdir(parents=True, exist_ok=True)


def find_candidates(root: Path) -> dict[str, Path]:
    """Find known root artifacts that should live in competitions/<slug>/data/raw."""
    candidates: dict[str, Path] = {}
    names = [
        "Training_Data.csv",
        "Test.csv",
        "SampleSubmission.csv",
        "TerraClimate_output.tiff",
    ]
    for name in names:
        probe_paths = [
            root / name,
            root / "data" / "raw" / name,
            root / "data" / "processed" / name,
        ]
        for p in probe_paths:
            if p.exists():
                candidates[name] = p
                break
    return candidates


def init_competition(
    slug: str,
    *,
    move_files: bool = False,
    assume_yes: bool = False,
    dry_run: bool = False,
) -> int:
    comp_dir = ROOT / "competitions" / slug
    raw = comp_dir / "data" / "raw"
    processed = comp_dir / "data" / "processed"
    notebooks = comp_dir / "notebooks"
    reports = comp_dir / "reports"
    submissions = comp_dir / "submissions"

    ensure_dirs((raw, processed, notebooks, reports, submissions), dry_run=dry_run)

    challenge_path = comp_dir / "challenge_config.json"
    if challenge_path.exists():
        print(f"Found existing: {challenge_path}")
    else:
        challenge = load_template("challenge_config_template.json")
        challenge.setdefault("slug", slug)
        write_json_atomic(challenge_path, challenge, dry_run=dry_run)

    state_path = comp_dir / "SKILL_STATE.json"
    if state_path.exists():
        print(f"Found existing: {state_path}")
    else:
        state = load_template("SKILL_STATE_template.json")
        # Keep both for backward compatibility with existing code/docs.
        state.setdefault("competition", slug)
        state.setdefault("competition_slug", slug)
        state.setdefault("dag_phase", "phase_0_foundation")
        state.setdefault("last_updated", datetime.now(timezone.utc).isoformat())
        write_json_atomic(state_path, state, dry_run=dry_run)

    candidates = find_candidates(ROOT)
    if not candidates:
        print("No known root artifacts found to move.")
    else:
        print("Detected candidate files to move into competition scope:")
        for name, path in candidates.items():
            print(f" - {name}: {path}")

        if move_files:
            if not assume_yes and not dry_run:
                response = input("Apply moves? Type YES to proceed: ").strip().upper()
                if response != "YES":
                    print("Aborting moves.")
                    return 0

            for name, src in candidates.items():
                dest = raw / name
                if dry_run:
                    print(f"[dry-run] move {src} -> {dest}")
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dest))
                print(f"Moved {src} -> {dest}")
        else:
            print("Run with --move-files to migrate detected artifacts.")

    mode = "DRY-RUN" if dry_run else "APPLY"
    print(f"Bootstrap complete ({mode}) for slug: {slug}")
    print(f"Competition folder: {comp_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tabula")
    sub = parser.add_subparsers(dest="command", required=True)

    init_parser = sub.add_parser("init", help="Bootstrap competition workspace")
    init_parser.add_argument("slug", help="competition slug (e.g. ey-frogs)")
    init_parser.add_argument(
        "--move-files",
        action="store_true",
        help="Move detected root artifacts into competitions/<slug>/data/raw",
    )
    init_parser.add_argument(
        "--yes",
        action="store_true",
        help="Assume YES for move confirmation",
    )
    init_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned actions without modifying files",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "init":
        return init_competition(
            args.slug,
            move_files=args.move_files,
            assume_yes=args.yes,
            dry_run=args.dry_run,
        )

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
