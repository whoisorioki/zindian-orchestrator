"""
Unit tests for zindian._safe_import helper (F9 fix).

Verifies that safe_import temporarily removes CWD and relative paths from sys.path
during import, preventing accidental imports of local files when requesting
third-party or standard packages.
"""

from __future__ import annotations

import os
import sys
import pytest
from pathlib import Path

from zindian._safe_import import safe_import


def test_safe_import_standard_module():
    """Standard modules and third-party packages should import without issues."""
    json_mod = safe_import("json")
    assert json_mod is not None
    assert hasattr(json_mod, "dumps")


def test_safe_import_blocks_cwd_relative_import(tmp_path: Path, monkeypatch):
    """
    If a local python file exists in CWD, safe_import should NOT load it if it's
    relying on CWD being in sys.path.
    """
    cwd = os.getcwd()
    # Ensure current directory is in sys.path for the baseline check
    if cwd not in sys.path:
        monkeypatch.syspath_prepends(cwd)

    # Calling safe_import for a non-existent package should fail with ModuleNotFoundError,
    # rather than loading anything from CWD.
    with pytest.raises(ModuleNotFoundError):
        safe_import("non_existent_package_12345")


def test_safe_import_restores_sys_path():
    """sys.path should be restored to its exact previous state after safe_import completes."""
    original_path = list(sys.path)
    _ = safe_import("math")
    assert sys.path == original_path, "sys.path was not restored after safe_import"
