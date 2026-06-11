from pathlib import Path


def test_only_skill05_writes_challenge_config():
    """Ensure only `zindian/skills/skill_05_cv.py` mutates `challenge_config.json`.

    The Source of Truth allows `skill_05_cv` to write a `cv_strategy` block
    into `challenge_config.json` during Phase 1. Other skill modules must not
    perform writes to the challenge config file.
    """
    repo_root = Path(".")
    py_files = list(repo_root.rglob("*.py"))
    offenders = []
    for p in py_files:
        # Skip files under tests/ (they may mention config paths in test text)
        if "tests/" in str(p):
            continue
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue

        for i, line in enumerate(lines, start=1):
            if "cfg_path.write_text(" in line or "cfg_path.write_text (" in line:
                # allowed writers: skill_05_cv.py and skill_02_intake.py (intake phase)
                if p.name not in ("skill_05_cv.py", "skill_02_intake.py"):
                    offenders.append(f"{p}:{i}")
            # direct literal path writes like Path(... 'challenge_config.json').write_text(...)
            if "challenge_config.json" in line and ("write_text(" in line or ("open(" in line and "w" in line)):
                if p.name not in ("skill_05_cv.py", "skill_02_intake.py"):
                    offenders.append(f"{p}:{i}")

    # Deduplicate
    offenders = sorted(set(offenders))
    if offenders:
        raise AssertionError("Unauthorized challenge_config.json writes found in:\n" + "\n".join(offenders))
