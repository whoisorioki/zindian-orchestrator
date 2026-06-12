"""Skill 13 — Oracle Fusion.

Delegates execution to zindian.oracle_fusion_core.
"""

from __future__ import annotations
from typing import Any
import zindian.oracle_fusion_core as core
from zindian.paths import resolve_competition_paths
from zindian.config import ChallengeConfig


def run(
    config: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Perform out-of-fold blending and multi-lateral correlation pruning."""
    # Temporarily monkeypatch the core module imports with whatever is on this module
    # (to respect any test monkeypatches on the skill module)
    orig_resolve = core.resolve_competition_paths
    orig_config = core.ChallengeConfig
    try:
        core.resolve_competition_paths = resolve_competition_paths
        setattr(core, "ChallengeConfig", ChallengeConfig)
        return core.run(config, state, dry_run)
    finally:
        core.resolve_competition_paths = orig_resolve
        setattr(core, "ChallengeConfig", orig_config)
