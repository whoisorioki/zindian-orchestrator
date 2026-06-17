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
    2) COMPETITION_SLUG environment variable
    3) Auto-detect when exactly one competitions/*/SKILL_STATE.json exists
    4) Legacy root fallback (unless require_competition=True)
    """
    # Use this file's location to find repo root, not cwd()
    root = Path(__file__).resolve().parent.parent
    # Accept both COMPETITION_SLUG (canonical) and ZINDIAN_COMPETITION_SLUG (alias).
    # The alias is widely used in run commands and diagnostic scripts throughout
    # this repository. Both resolve identically — COMPETITION_SLUG takes precedence.
    selected_slug = (
        slug
        or os.environ.get("COMPETITION_SLUG")
        or os.environ.get("ZINDIAN_COMPETITION_SLUG")
    )
    comp_dir: Optional[Path] = None

    if selected_slug:
        candidate = root / "competitions" / selected_slug
        if candidate.exists():
            comp_dir = candidate
        else:
            raise FileNotFoundError(
                f"Competition '{selected_slug}' not found at {candidate}. "
                f"Available: {[p.name for p in (root / 'competitions').glob('*') if p.is_dir()]}"
            )

    if comp_dir is None:
        matches = list((root / "competitions").glob("*/SKILL_STATE.json"))
        if len(matches) == 1:
            comp_dir = matches[0].parent
        elif len(matches) > 1:
            # Prefer the most recently updated state file when auto-selecting.
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

    if comp_dir is None:
        if require_competition:
            raise FileNotFoundError(
                "No active competition found. Run bootstrap or set COMPETITION_SLUG."
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
