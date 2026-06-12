"""Skill 00 — Zindi Monitor.

Delegates execution to zindian.zindi_monitor_core.
"""

from __future__ import annotations
from typing import Any
from zindian.zindi_monitor_core import run as core_run


def run(config: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """Scrape and check Zindi competition pages, compliance, and metrics."""
    return core_run(config, state)
