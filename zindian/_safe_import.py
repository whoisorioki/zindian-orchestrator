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

    Evicts any module from sys.modules if it was previously imported
    from an untrusted CWD directory.
    """
    cwd = os.getcwd()
    bad_paths = {"", ".", cwd, os.path.abspath(cwd)}

    # If already cached in sys.modules, check if it came from a bad path
    if name in sys.modules:
        mod = sys.modules[name]
        mod_file = getattr(mod, "__file__", None)
        if mod_file:
            mod_dir = os.path.dirname(os.path.abspath(mod_file))
            if mod_dir in bad_paths or os.path.abspath(mod_file) in bad_paths:
                del sys.modules[name]

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
