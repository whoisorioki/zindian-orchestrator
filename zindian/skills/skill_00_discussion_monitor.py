"""Compatibility shim for the legacy Skill 00 module name.

Delegates to zindian.zindi_monitor_core.
"""

from __future__ import annotations
from typing import Any
from zindian.zindi_monitor_core import run as core_run


def run(config: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """Expose run for legacy callers, delegating to the core monitor."""
    return core_run(config, state)
