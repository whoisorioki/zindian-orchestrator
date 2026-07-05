"""Test A5 compliance: No hardcoded competition-specific strings."""

import re
from pathlib import Path


def test_no_hardcoded_targets_in_skill_07():
    """skill_07 must not hardcode target names."""
    skill_path = Path("zindian/skills/skill_07_features.py")
    content = skill_path.read_text(encoding="utf-8")

    # Forbidden patterns - target names that should come from config
    forbidden = [
        r'"total_goals"',
        r'"Target"',
        r"'total_goals'",
        r"'Target'",
    ]

    for pattern in forbidden:
        matches = re.findall(pattern, content)
        assert len(matches) == 0, (
            f"Found hardcoded target name {pattern} in skill_07. "
            f"Use config['target_config']['targets'] instead."
        )


def test_no_hardcoded_metrics_in_skills():
    """No skill should hardcode metric names."""
    skill_dir = Path("zindian/skills")
    forbidden = [
        r'"anchor_oof_f1"',
        r'"anchor_oof_auc"',
        r'"anchor_oof_rmse"',
        r"'anchor_oof_f1'",
        r"'anchor_oof_auc'",
        r"'anchor_oof_rmse'",
    ]

    for skill_file in skill_dir.glob("skill_*.py"):
        content = skill_file.read_text(encoding="utf-8")
        for pattern in forbidden:
            matches = re.findall(pattern, content)
            assert len(matches) == 0, (
                f"Found hardcoded metric {pattern} in {skill_file.name}. "
                f"Use f'anchor_oof_{{metric_key}}' pattern instead."
            )
