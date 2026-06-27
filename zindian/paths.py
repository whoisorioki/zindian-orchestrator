from __future__ import annotations

import os
import json
from dataclasses import dataclass
from datetime import datetime
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
    *,
    require_competition: bool = False,
) -> CompetitionPaths:
    """Resolve canonical project paths for the active competition.

    Resolution order:
    1) Explicit slug argument
    2) Current working directory (if inside competitions/<slug>/)
    3) ZINDIAN_COMPETITION or COMPETITION_SLUG env var
    4) .env file ZINDIAN_COMPETITION or COMPETITION_SLUG
    5) Auto-detect when exactly one competitions/*/SKILL_STATE.json exists
    6) Legacy root fallback
    """
    root = Path(__file__).resolve().parent.parent
    comp_root = root / "competitions"
    cwd = Path.cwd().resolve()
    
    selected_slug = slug

    # 1) Current Working Directory Check
    if not selected_slug:
        if comp_root.exists() and cwd.is_relative_to(comp_root) and cwd != comp_root:
            relative = cwd.relative_to(comp_root)
            selected_slug = relative.parts[0]

    # 2) Environment Variable Check
    if not selected_slug:
        selected_slug = (
            os.environ.get("ZINDIAN_COMPETITION")
            or os.environ.get("COMPETITION_SLUG")
            or os.environ.get("ZINDIAN_COMPETITION_SLUG")
        )

    # 3) .env File Check
    if not selected_slug:
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
                            if k in ("ZINDIAN_COMPETITION", "COMPETITION_SLUG", "ZINDIAN_COMPETITION_SLUG"):
                               selected_slug = v
                               break
            except Exception:
                pass

    comp_dir: Optional[Path] = None

    if selected_slug:
        candidate = comp_root / selected_slug
        if candidate.exists():
            comp_dir = candidate
        else:
            raise FileNotFoundError(
                f"Competition '{selected_slug}' not found at {candidate}. "
                f"Available: {[p.name for p in comp_root.glob('*') if p.is_dir()]}"
            )

    # 4) Auto-detect Fallback
    if comp_dir is None and comp_root.exists():
        matches = list(comp_root.glob("*/SKILL_STATE.json"))
        if len(matches) == 1:
            comp_dir = matches[0].parent
        elif len(matches) > 1:
            def _state_sort_key(path: Path) -> tuple[int, float]:
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    ts = data.get("last_updated")
                    if isinstance(ts, str) and ts:
                        norm = ts.replace("Z", "+00:00")
                        return (1, datetime.fromisoformat(norm).timestamp())
                except Exception:
                    pass
                return (0, path.stat().st_mtime)

            best = max(matches, key=_state_sort_key)
            comp_dir = best.parent

    # 5) Fallback error or legacy root fallback
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
