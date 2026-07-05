import sys

sys.path.insert(0, ".")
from zindian.state import SkillStateStore
from zindian.paths import resolve_competition_paths

paths = resolve_competition_paths()
store = SkillStateStore(paths.state_path)
state = store.read()
oof = state.get("branch_anchor-baseline_label_oof")
if isinstance(oof, dict):
    oof["cv_strategy_id"] = "config:stratified"
    store.write(state)
    print(
        "Updated branch_anchor-baseline_label_oof.cv_strategy_id -> config:stratified"
    )
else:
    print("OOF record not found")
