from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .schemas import skill_state_skeleton, validate_skill_state


def _iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(data, indent=2, sort_keys=False) + "\n"

    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as tmp:
        tmp.write(serialized)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


@dataclass
class SkillStateStore:
    path: Path

    def read(self) -> Dict[str, Any]:
        if not self.path.exists():
            state = skill_state_skeleton()
            _atomic_write_json(self.path, state)
            return state
        obj = json.loads(self.path.read_text(encoding="utf-8"))
        return validate_skill_state(obj)

    def write(self, new_state: Dict[str, Any], *, touch_timestamp: bool = True) -> Dict[str, Any]:
        state = dict(new_state)
        if touch_timestamp:
            state["last_updated"] = _iso_now()
        validate_skill_state(state)
        _atomic_write_json(self.path, state)
        return state

    def update(self, **patch: Any) -> Dict[str, Any]:
        state = self.read()
        state.update(patch)
        return self.write(state)
    
    def increment(self, key: str, delta: int = 1) -> int:
        """Increment a numeric field and return new value."""
        state = self.read()
        if key not in state:
            state[key] = 0
        state[key] = state[key] + delta
        self.write(state)
        return state[key]
    
    def append_selected(self, submission_id: int) -> None:
        """Append submission to selected_submissions list."""
        state = self.read()
        if submission_id not in state["selected_submissions"]:
            state["selected_submissions"].append(submission_id)
        self.write(state)


def resolve_active_cv_strategy_id(state_obj: dict, config_obj: dict) -> str:
    """
    Resolve the active CV strategy identifier according to the Source of Truth rules.

    Priority:
      1. If SKILL_STATE contains an active `cv_strategy_override.active` == True,
         return an 'override:<override_strategy>' identifier.
      2. Else, read `challenge_config.json` cv_strategy block and return
         'config:<type>' identifier.
      3. Fallback to 'unknown'.

    This function returns a short string suitable for tagging OOF artifacts
    and SKILL_STATE entries.
    """
    try:
        override = state_obj.get("cv_strategy_override", {}) or {}
        if override.get("active", False):
            return f"override:{override.get('override_strategy') or 'unknown'}"
    except Exception:
        pass

    try:
        cv = (config_obj or {}).get("cv_strategy") if isinstance(config_obj, dict) else None
        if isinstance(cv, dict):
            return f"config:{cv.get('type', 'unknown')}"
    except Exception:
        pass

    return "unknown"


def is_anchor_challenge_active(state_obj: dict) -> bool:
    """Safe accessor for anchor_challenge.active in SKILL_STATE.

    Returns True only if `anchor_challenge` is present and has `active`==True.
    This protects automation from KeyError when the block is absent.
    """
    try:
        return bool((state_obj or {}).get("anchor_challenge", {}) .get("active", False))
    except Exception:
        return False


def get_anchor_challenge_config(state_obj: dict) -> dict:
    """Return the `anchor_challenge` config block or empty dict if absent."""
    try:
        return dict((state_obj or {}).get("anchor_challenge") or {})
    except Exception:
        return {}

