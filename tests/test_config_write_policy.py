import re
from pathlib import Path


def test_no_config_writes_outside_allowed_files():
    repo_root = Path(__file__).resolve().parents[1]
    allowed_prefixes = (
        "zindian/skills/skill_02_intake.py",
        "zindian/skills/skill_05_cv.py",
        "scripts/",
    )

    offenders = []
    for p in repo_root.rglob("*.py"):
        rel = p.relative_to(repo_root).as_posix()
        if any(rel == a or rel.startswith(a) for a in allowed_prefixes):
            continue
        parts = p.relative_to(repo_root).parts
        if parts[0] in (".venv", "venv", "env", "build"):
            continue
        if "site-packages" in parts or "dist-packages" in parts:
            continue
        if rel.startswith("tests/"):
            continue

        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue

        # More precise patterns that indicate writes specifically to the challenge config
        patterns = [
            r"\b(config_path|cfg_path|paths\.config_path)\.write_text\(",
            r"open\([^\)]*challenge_config\.json[^\)]*,\s*['\"]?w",
            r"write_text\([^\)]*challenge_config\.json",
        ]
        if any(re.search(pat, text) for pat in patterns):
            offenders.append(rel)

    assert not offenders, f"Found unexpected config writes in: {offenders}"
