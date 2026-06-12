from __future__ import annotations

import re
from pathlib import Path


def test_no_direct_cross_skill_imports_outside_compatibility_shims() -> None:
    """Fail if a skill directly imports another skill module.

    Compatibility shims are allowed to re-export legacy module names:
      - zindian/skills/skill_00_discussion_monitor.py
      - zindian/skills/skill_13_ensemble.py
    """
    repo_root = Path(__file__).resolve().parents[1]
    allowed_files = {
        "zindian/skills/skill_00_discussion_monitor.py",
        "zindian/skills/skill_13_ensemble.py",
    }
    pattern = re.compile(
        r"^(?:from|import)\s+(?:\.|zindian\.)?skills?\.(skill_[0-9]{2}_[a-z0-9_]+)",
        re.MULTILINE,
    )

    offenders: list[str] = []
    for path in repo_root.rglob("*.py"):
        rel = path.relative_to(repo_root).as_posix()
        if rel in allowed_files or rel.startswith("tests/"):
            continue
        if any(part in {".venv", "venv", "env", "build"} for part in path.parts):
            continue
        if "site-packages" in path.parts or "dist-packages" in path.parts:
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue

        if pattern.search(text):
            offenders.append(rel)

    assert not offenders, (
        "Direct skill-to-skill imports found outside compatibility shims:\n"
        + "\n".join(sorted(offenders))
    )
