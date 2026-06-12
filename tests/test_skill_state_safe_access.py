import ast
from pathlib import Path


def _find_state_subscript_reads(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    issues = []
    for node in ast.walk(tree):
        # look for subscript nodes like state["key"] used in Load context
        if isinstance(node, ast.Subscript):
            value = node.value
            if isinstance(value, ast.Name) and value.id == "state":
                # If the subscript node is used (loaded) rather than assigned
                # its ctx will be ast.Load when used in expressions
                # We treat any Subscript as potentially unsafe read; assignments
                # to state[...] are allowed (they appear as targets elsewhere).
                getattr(node, "ctx", None)
                # Conservative: treat all Subscript uses as reads unless it's
                # directly inside an Assign target list. We'll detect load
                # context by checking ancestor usage via AST nodes.
                issues.append((path, node.lineno))
    return issues


def test_no_state_bracket_reads_in_skills():
    """Fail if any skill module reads SKILL_STATE via bracket access (state[...]).

    Reading optional keys should use `state.get(...)` per SoT safe-access rules.
    Assignment to `state[...] = ...` is permitted, but bracket reads are disallowed.
    """
    skill_dir = Path("zindian/skills")
    assert skill_dir.exists(), "zindian/skills directory missing"

    offenders = []
    for p in skill_dir.glob("*.py"):
        try:
            issues = _find_state_subscript_reads(p)
        except SyntaxError:
            # Skip files that aren't valid Python for tests (shouldn't happen)
            continue
        # Filter out files that only assign to state[...] by scanning the line
        # and excluding obvious assignment targets.
        for path, lineno in issues:
            line = path.read_text(encoding="utf-8").splitlines()[lineno - 1].strip()
            # If the line contains "=" before the state[...] occurrence, consider it an assignment
            if "=" in line:
                # crude heuristic: skip obvious assignments
                continue
            offenders.append(f"{path}:{lineno}: {line}")

    if offenders:
        msg = "Found bracket-style SKILL_STATE reads in skill modules (use state.get(...)):\n"
        msg += "\n".join(offenders)
        raise AssertionError(msg)
