"""
Defensive import helper — strips the CWD and repo root from sys.path
during import to prevent unintended relative module imports when running
from the repository directory.
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import Any


def safe_import(name: str) -> Any:
    """
    Import module 'name' safely by stripping current working directory
    and script root directory ('', '.', CWD) from sys.path temporarily.
    """
    cwd = os.getcwd()
    bad_paths = {"", ".", cwd, os.path.abspath(cwd)}
    old_path = list(sys.path)
    sys.path = [
        p
        for p in sys.path
        if p not in bad_paths and os.path.abspath(p) not in bad_paths
    ]
    try:
        return importlib.import_module(name)
    finally:
        sys.path = old_path
