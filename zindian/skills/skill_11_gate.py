"""
Skill 11 — Branch Gate
Promotes the best passing variant to new anchor.
Blocks if no variant passed the gate this round.
"""
from __future__ import annotations
import subprocess
from datetime import datetime, timezone
from zindian.paths import resolve_competition_paths
from zindian.state import SkillStateStore

def run() -> dict:
    print("\n" + "="*60)
    print("SKILL 11 — Branch Gate")
    print("="*60 + "\n")

    paths = resolve_competition_paths()
    store = SkillStateStore(paths.state_path)
    state = store.read()

    best_variant = state.get("best_variant_this_round")
    best_auc     = float(state.get("best_variant_oof_auc") or 0.0)
    anchor_auc   = float(state.get("anchor_oof_auc") or 0.0)
    variants_passed = int(state.get("variants_passed") or 0)

    print(f"Variants passed this round : {variants_passed}")
    print(f"Best variant               : {best_variant}")
    print(f"Best OOF AUC               : {best_auc:.5f}")
    print(f"Current anchor AUC         : {anchor_auc:.5f}")
    print(f"Delta                      : {best_auc - anchor_auc:+.5f}")

    if variants_passed == 0:
        print("\n❌ GATE BLOCKED — no variant passed this round")
        print("   Return to Skill 07 with different feature hypotheses")
        store.update(dag_phase="phase_3_gate_blocked",
                     last_updated=datetime.now(timezone.utc).isoformat())
        return {"status": "BLOCKED", "reason": "no variants passed"}

    # Promote best variant to new anchor
    round_num  = int(state.get("feature_round") or 1)
    new_branch = f"anchor-v{round_num + 1}"

    print(f"\n✅ GATE PASSED — promoting {best_variant} to {new_branch}")

    try:
        subprocess.run(["git", "checkout", "-b", new_branch],
                      check=True, capture_output=True)
        print(f"✅ Git branch created: {new_branch}")
    except subprocess.CalledProcessError:
        subprocess.run(["git", "checkout", new_branch],
                      check=True, capture_output=True)
        print(f"✅ Switched to: {new_branch}")

    store.update(
        anchor_oof_auc      = best_auc,
        anchor_oof_f1       = state.get("best_variant_oof_f1"),
        anchor_git_branch   = new_branch,
        feature_round       = round_num + 1,
        variants_tested     = 0,
        variants_passed     = 0,
        best_variant_this_round = None,
        best_variant_oof_auc    = None,
        dag_phase               = "phase_3_anchor_promoted",
        last_updated            = datetime.now(timezone.utc).isoformat(),
    )
    print(f"✅ SKILL_STATE.json — new anchor AUC: {best_auc:.5f}")
    print(f"✅ Feature round advanced to: {round_num + 1}")

    return {
        "status":      "PASS",
        "new_anchor":  new_branch,
        "anchor_auc":  best_auc,
        "promoted":    best_variant,
    }

if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2))
