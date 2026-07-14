#!/usr/bin/env python3
"""Dynamic state synchronization utility."""

import json
import subprocess
from datetime import datetime, timezone
from typing import Any, Iterable, cast

from zindian.config import ChallengeConfig
from zindian.paths import resolve_competition_paths
from zindian.state import SkillStateStore
from zindian.zindi_client import ZindiClient


def get_git_branch() -> str:
    """Get current git branch."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def sync_submission_board(client: ZindiClient, state: dict[str, Any]) -> dict[str, Any]:
    """Sync submission board data to state."""
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        subs = list(cast(Iterable[Any], client._user.submission_board()))

    selected = [s for s in subs if s.get("chosen")]
    state["selected_submissions"] = [s.get("filename") for s in selected]
    state["submissions_used_total"] = len(subs)

    # Count submissions used today based on Zindi server UTC timestamps
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    used_today = 0
    for s in subs:
        created_at = s.get("created_at")
        if isinstance(created_at, str) and created_at.startswith(today_str):
            used_today += 1
    state["submissions_used_today"] = used_today

    if subs:
        best_score = None
        config = ChallengeConfig.load()

        # Determine metric direction: use target_config metric_direction for target_col if available
        target_col = config.get("target_col")
        target_config = config.get("target_config", {})
        maximize = True

        if (
            target_config
            and isinstance(target_config, dict)
            and "targets" in target_config
        ):
            for t in target_config["targets"]:
                if t.get("name") == target_col:
                    maximize = t.get("metric_direction") == "maximize"
                    break
        else:
            maximize = config.get("metric_direction") == "maximize"

        for s in subs:
            score = s.get("public_score")
            if score is not None:
                if best_score is None:
                    best_score = score
                elif maximize and score > best_score:
                    best_score = score
                elif not maximize and score < best_score:
                    best_score = score

        if best_score is not None:
            state["anchor_lb_score"] = best_score

    return state


def sync_leaderboard(client: ZindiClient, state: dict[str, Any]) -> dict[str, Any]:
    """Sync leaderboard rank to state."""
    try:
        rank = client._user.my_rank
        state["anchor_rank"] = rank
    except Exception:
        pass
    return state


def sync_all() -> dict[str, Any]:
    """Sync all dynamic state."""
    paths = resolve_competition_paths()
    config = ChallengeConfig.load()
    store = SkillStateStore(paths.state_path)
    state = store.read()

    # Git branch
    state["current_git_branch"] = get_git_branch()

    # Zindi data
    try:
        client = ZindiClient()
        client.select_competition(config.slug)
        state = sync_submission_board(client, state)
        state = sync_leaderboard(client, state)
        state["remaining_submissions"] = client.remaining_submissions
    except Exception as e:
        print(f"  [WARN]  Could not sync Zindi data: {e}")

    # Timestamp
    state["last_updated"] = datetime.now(timezone.utc).isoformat()

    store.write(state)
    return state


run = sync_all


if __name__ == "__main__":
    result = sync_all()
    print(
        json.dumps(
            {
                "current_git_branch": result.get("current_git_branch"),
                "selected_submissions": result.get("selected_submissions"),
                "anchor_lb_score": result.get("anchor_lb_score"),
                "anchor_rank": result.get("anchor_rank"),
                "remaining_submissions": result.get("remaining_submissions"),
                "last_updated": result.get("last_updated"),
            },
            indent=2,
        )
    )
