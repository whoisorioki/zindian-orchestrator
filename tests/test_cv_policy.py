import re
from pathlib import Path


def test_no_direct_cv_splitters_instantiated():
    """Fail if StratifiedKFold/GroupKFold/KFold are instantiated outside allowed files.

    Allowed locations:
      - zindian/cv.py (the central factory)
      - zindian/skills/skill_05_cv.py (CV architect)
      - scripts/ (utility scripts)
    """
    repo_root = Path(__file__).resolve().parents[1]
    patterns = [r"\bStratifiedKFold\b", r"\bGroupKFold\b", r"\bKFold\b"]
    allowed_prefixes = (
        "zindian/cv.py",
        "zindian/skills/skill_05_cv.py",
        "zindian/skills/_lightgbm_shared.py",
        "scripts/",
        "competitions/",
        "zindian/skills/skill_21_pseudo_label.py",
        "zindian/skills/skill_08_anchor.py",
        "zindian/three_lens.py",
    )

    offenders = []
    for p in repo_root.rglob("*.py"):
        rel = p.relative_to(repo_root).as_posix()
        # skip allowed files entirely
        if any(rel == a or rel.startswith(a) for a in allowed_prefixes):
            continue
        # skip tests and virtualenv/site-packages
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
        for pat in patterns:
            if re.search(pat, text):
                offenders.append((rel, pat))
                break

    if offenders:
        msgs = [f"{f} contains disallowed CV splitter ({pat})" for f, pat in offenders]
        raise AssertionError(
            "Disallowed CV splitter instantiations found:\n" + "\n".join(msgs)
        )
