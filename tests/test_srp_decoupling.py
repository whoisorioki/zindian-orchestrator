import ast
import re
from pathlib import Path


def test_no_cross_skill_imports():
    skills_dir = Path(__file__).resolve().parents[1] / "zindian" / "skills"
    violations = []
    for skill_file in skills_dir.glob("skill_*.py"):
        tree = ast.parse(
            skill_file.read_text(encoding="utf-8"), filename=str(skill_file)
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    if is_cross_skill(name.name):
                        violations.append(f"{skill_file.name}: imports {name.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                level = node.level
                full_name = f"{'.' * level}{module}" if level > 0 else module
                if is_cross_skill(full_name):
                    violations.append(f"{skill_file.name}: imports {full_name}")
    assert not violations, f"Cross-skill import violations found: {violations}"


def is_cross_skill(module_name: str) -> bool:
    name = module_name.lstrip(".")
    if name.startswith("zindian.skills.skill_"):
        return True
    if name.startswith("skills.skill_"):
        return True
    if name.startswith("skill_") and re.match(r"^skill_\d{2}", name):
        return True
    return False
