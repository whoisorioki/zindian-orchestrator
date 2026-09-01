from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class CompetitionPaths:
    root: Path
    competition_dir: Optional[Path]
    state_path: Path
    config_path: Path
    reports_dir: Path
    submissions_dir: Path
    data_raw_dir: Path
    data_processed_dir: Path
    notebooks_dir: Path


def resolve_competition_paths(
    slug: str | None = None,
    competition_dir: Path | str | None = None,
    *,
    require_competition: bool = False,
) -> CompetitionPaths:
    """Resolve canonical project paths for the active competition.

    Resolution order:
    1) Explicit competition_dir argument
    2) Explicit slug argument (checked against competitions/<slug> and directly as Path)
    3) Current working directory (if inside competitions/<slug>/ or contains SKILL_STATE.json)
    4) ZINDIAN_COMPETITION, ZINDIAN_COMPETITION_DIR, or COMPETITION_SLUG env var
    5) .env file ZINDIAN_COMPETITION or COMPETITION_SLUG
    6) Auto-detect when exactly one competitions/*/SKILL_STATE.json exists
    7) Legacy root fallback
    """
    root = Path(__file__).resolve().parent.parent
    comp_root = root / "competitions"
    cwd = Path.cwd().resolve()

    comp_dir: Optional[Path] = None

    # 0) Direct competition_dir argument
    if competition_dir is not None:
        c_path = Path(competition_dir).resolve()
        if c_path.exists():
            comp_dir = c_path
        else:
            raise FileNotFoundError(f"Specified competition directory not found: {c_path}")

    selected_slug = slug

    # 1) Direct Path or Slug check
    if comp_dir is None and selected_slug:
        candidate_slug = comp_root / selected_slug
        candidate_direct = Path(selected_slug).resolve()
        if candidate_slug.exists():
            comp_dir = candidate_slug
        elif candidate_direct.exists() and (candidate_direct / "SKILL_STATE.json").exists():
            comp_dir = candidate_direct
        elif not require_competition:
            comp_dir = candidate_slug
        else:
            available = [p.name for p in comp_root.glob("*") if p.is_dir()] if comp_root.exists() else []
            raise FileNotFoundError(
                f"Competition '{selected_slug}' not found at {candidate_slug}. "
                f"Available: {available}"
            )

    # 2) Current Working Directory Check
    if comp_dir is None:
        if comp_root.exists() and cwd.is_relative_to(comp_root) and cwd != comp_root:
            relative = cwd.relative_to(comp_root)
            selected_slug = relative.parts[0]
            comp_dir = comp_root / selected_slug
        elif (cwd / "SKILL_STATE.json").exists() or (cwd / "challenge_config.json").exists():
            comp_dir = cwd

    # 3) Environment Variable Check
    if comp_dir is None:
        env_dir = os.environ.get("ZINDIAN_COMPETITION_DIR")
        if env_dir and Path(env_dir).exists():
            comp_dir = Path(env_dir).resolve()
        else:
            env_slug = (
                os.environ.get("ZINDIAN_COMPETITION")
                or os.environ.get("COMPETITION_SLUG")
                or os.environ.get("ZINDIAN_COMPETITION_SLUG")
            )
            if env_slug:
                candidate = comp_root / env_slug
                if candidate.exists():
                    comp_dir = candidate

    # 4) .env File Check
    if comp_dir is None:
        dotenv_path = root / ".env"
        if dotenv_path.exists():
            try:
                with dotenv_path.open(encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip("'").strip('"')
                            if k in (
                                "ZINDIAN_COMPETITION",
                                "COMPETITION_SLUG",
                                "ZINDIAN_COMPETITION_SLUG",
                            ):
                                candidate = comp_root / v
                                if candidate.exists():
                                    comp_dir = candidate
                                    break
            except Exception:
                pass

    # 5) Auto-detect Fallback
    if comp_dir is None and comp_root.exists():
        matches = list(comp_root.glob("*/SKILL_STATE.json"))
        if len(matches) == 1:
            comp_dir = matches[0].parent
        elif len(matches) > 1:
            if require_competition:
                slugs = [m.parent.name for m in matches]
                raise ValueError(
                    f"Ambiguous competition context: multiple competitions found ({slugs}). "
                    f"Please specify a slug explicitly, set ZINDIAN_COMPETITION, or run from within a competition directory."
                )

    # 6) Fallback error or legacy root fallback
    if comp_dir is None:
        if require_competition:
            raise FileNotFoundError(
                "No active competition context resolved. Please set ZINDIAN_COMPETITION, run inside a competition subdirectory, or define ZINDIAN_COMPETITION in .env."
            )
        return CompetitionPaths(
            root=root,
            competition_dir=None,
            state_path=root / "SKILL_STATE.json",
            config_path=root / "challenge_config.json",
            reports_dir=root / "reports",
            submissions_dir=root / "submissions",
            data_raw_dir=root / "data" / "raw",
            data_processed_dir=root / "data" / "processed",
            notebooks_dir=root / "notebooks",
        )

    return CompetitionPaths(
        root=root,
        competition_dir=comp_dir,
        state_path=comp_dir / "SKILL_STATE.json",
        config_path=comp_dir / "challenge_config.json",
        reports_dir=comp_dir / "reports",
        submissions_dir=comp_dir / "submissions",
        data_raw_dir=comp_dir / "data" / "raw",
        data_processed_dir=comp_dir / "data" / "processed",
        notebooks_dir=comp_dir / "notebooks",
    )
