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

