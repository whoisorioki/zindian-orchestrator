import sys
sys.path.insert(0, ".")
from zindian.state import SkillStateStore
from zindian.paths import resolve_competition_paths

paths = resolve_competition_paths()
store = SkillStateStore(paths.state_path)
state = store.read()
print("human_gate_4_approved raw value:", repr(state.get("human_gate_4_approved")))
print("human_gate_2_anchor-baseline_label_approved raw value:", repr(state.get("human_gate_2_anchor-baseline_label_approved")))
print("anchor_git_branch raw value:", repr(state.get("anchor_git_branch")))
print("anchor_baseline_approved raw value:", repr(state.get("anchor_baseline_approved")))
print("anchor_oof_score raw value:", repr(state.get("anchor_oof_score")))