#!/usr/bin/env python3
"""
One-time migration: Update all SKILL_STATE operations in codebase
Run from zindian-orchestrator root: python scripts/migrate_skill_state.py
"""

import re
from pathlib import Path


def migrate_file(file_path):
    """Update SKILL_STATE operations in a Python file."""
    content = file_path.read_text()
    original = content

    # Pattern 1: json.load with SKILL_STATE
    pattern1 = r"(\s+)with open\(([^)]*SKILL_STATE[^)]*)\) as ([^:]+):\s*\n\s+(\w+)\s*=\s*json\.load\(\3\)"
    replacement1 = r"\1from tabula.skill_state import read_state\n\1\4 = read_state(\2)"
    content = re.sub(pattern1, replacement1, content)

    # Pattern 2: json.dump with SKILL_STATE
    pattern2 = r'(\s+)with open\(([^)]*SKILL_STATE[^)]*),\s*["\']w["\']\) as ([^:]+):\s*\n\s+json\.dump\(([^,]+),\s*\3'
    replacement2 = r"\1from tabula.skill_state import write_state\n\1write_state(\2, \4"
    content = re.sub(pattern2, replacement2, content)

    if content != original:
        print(f"✓ Updated: {file_path}")
        return content
    return None


if __name__ == "__main__":
    root = Path(__file__).parent.parent
    updated = 0

    for py_file in root.rglob("*.py"):
        if "skill_state" in py_file.name or ".venv" in str(py_file):
            continue

        new_content = migrate_file(py_file)
        if new_content:
            # Uncomment to apply changes:
            # py_file.write_text(new_content)
            updated += 1

    print(f"\n{updated} files need updating")
    print("Uncomment line 35 to apply changes")
