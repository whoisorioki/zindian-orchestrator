import sys

sys.path.insert(0, ".")

from zindian.state import SkillStateStore
from zindian.paths import resolve_competition_paths

paths = resolve_competition_paths()
store = SkillStateStore(paths.state_path)
state = store.read()

# Force rewrite through patched json.dump
store.update(**state)

print("Optimized SKILL_STATE.json rewritten via autopatch")
print("human_gate_4_approved:", state.get("human_gate_4_approved"))
print("anchor_baseline_approved:", state.get("anchor_baseline_approved"))
