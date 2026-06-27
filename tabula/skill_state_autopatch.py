"""
Auto-patch for SKILL_STATE.json operations.
Import this at the top of any script to automatically handle score externalization.

Usage:
    import tabula.skill_state_autopatch  # noqa
    # Now all SKILL_STATE.json operations are automatically optimized
"""
import json
import builtins
from pathlib import Path
from typing import Any

_original_open = builtins.open
_original_json_load = json.load
_original_json_dump = json.dump
_original_json_dumps = json.dumps


class _SkillStateFile:
    def __init__(self, file, mode, args, kwargs, path):
        self._f = _original_open(file, mode, *args, **kwargs)
        self._is_skill_state = True
        self._skill_state_path = path

    def __getattr__(self, name):
        return getattr(self._f, name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._f.__exit__(exc_type, exc_val, exc_tb)


def _patched_open(file, mode='r', *args, **kwargs):
    """Intercept SKILL_STATE.json opens."""
    f = _original_open(file, mode, *args, **kwargs)

    if isinstance(file, (str, Path)) and "SKILL_STATE.json" in str(file):
        wrapped = _SkillStateFile(file, mode, args, kwargs, Path(file))
        return wrapped

    return f


# NOTE: SKILL_STATE externalization/hydration is now handled by
# zindian.state._atomic_write_json / SkillStateStore.read().
# This autopatch no longer patches json.load/json.dump to avoid
# conflicting with the core writer and causing state corruption.


# Keep file-open wrapper only; JSON hydration/externalization now lives in zindian.state.
builtins.open = _patched_open
