"""Resource monitoring integration for cost_monitor."""

import subprocess
from typing import Optional, Tuple


def get_current_utilization() -> Tuple[Optional[float], Optional[float]]:
    """Get current CPU and memory utilization.

    Returns:
        Tuple of (cpu_percent, memory_percent)
    """
    cpu = _get_cpu_usage()
    memory = _get_memory_usage()
    return cpu, memory


def _get_cpu_usage() -> Optional[float]:
    """Get current CPU utilization percentage."""
    try:
        result = subprocess.run(
            ["top", "-bn1"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.split("\n"):
            if "Cpu(s)" in line:
                parts = line.split(",")
                for part in parts:
                    if "id" in part:
                        idle = float(part.split()[0])
                        return round((100 - idle) / 100, 4)
    except Exception:
        pass
    return None


def _get_memory_usage() -> Optional[float]:
    """Get current memory utilization percentage."""
    try:
        result = subprocess.run(
            ["free", "-m"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        lines = result.stdout.split("\n")
        if len(lines) > 1:
            mem_line = lines[1].split()
            total = float(mem_line[1])
            used = float(mem_line[2])
            return round(used / total, 4)
    except Exception:
        pass
    return None
