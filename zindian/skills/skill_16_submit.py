"""
Skill 16 — Submission Governance
Validates submission file, human gates, submits, updates state.
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from zindian.paths import resolve_competition_paths
from zindian.config import ChallengeConfig
from zindian.state import SkillStateStore

def validate(sub_path: Path, sample_path: Path) -> list[str]:
    errors = []
    sub    = pd.read_csv(sub_path)
    sample = pd.read_csv(sample_path)
    if list(sub.columns) != list(sample.columns):
        errors.append(f"Column mismatch: {list(sub.columns)} vs {list(sample.columns)}")
    if len(sub) != len(sample):
        errors.append(f"Row count: got {len(sub)}, expected {len(sample)}")
    if set(sub["ID"].astype(str)) != set(sample["ID"].astype(str)):
        errors.append("ID mismatch vs SampleSubmission")
    if list(sub["ID"].astype(str)) != list(sample["ID"].astype(str)):
        errors.append("ID order mismatch")
    if sub.isnull().any().any():
        errors.append(f"Nulls in: {sub.columns[sub.isnull().any()].tolist()}")
    return errors

def run(submission_file: str) -> dict:
    print("\n" + "="*60)
    print("SKILL 16 — Submission Governance")
    print("="*60 + "\n")

    paths  = resolve_competition_paths()
    config = ChallengeConfig.load()
    store  = SkillStateStore(paths.state_path)
    state  = store.read()

    sub_path    = Path(submission_file)
    sample_path = paths.data_raw_dir / "SampleSubmission.csv"

    if not sub_path.exists():
        raise FileNotFoundError(f"Submission file not found: {sub_path}")

    # ── Validate ──────────────────────────────────────────────
    print(f"Validating: {sub_path.name}")
    errors = validate(sub_path, sample_path)
    if errors:
        print(f"\n❌ VALIDATION FAILED:")
        for e in errors:
            print(f"   {e}")
        return {"status": "BLOCKED", "errors": errors}
    print("✅ Validation passed (5/5 checks)")

    # ── Budget check ──────────────────────────────────────────
    remaining = int(state.get("remaining_submissions") or 0)
    used_today = int(state.get("submissions_used_today") or 0)
    print(f"\nBudget: {remaining} remaining | {used_today} used today")
    if remaining <= 2:
        print("❌ BUDGET GUARD — fewer than 2 submissions remaining. Aborting.")
        return {"status": "BLOCKED", "reason": "budget"}

    # ── Human gate ────────────────────────────────────────────
    best_auc = state.get("best_variant_oof_auc") or state.get("anchor_oof_auc")
    best_f1  = state.get("best_variant_oof_f1")  or state.get("anchor_oof_f1")
    print(f"""
{'='*60}
=== HUMAN GATE: Skill 16 — Submit ===
{'='*60}
File             : {sub_path.name}
OOF AUC          : {best_auc}
OOF F1           : {best_f1}
Remaining today  : {remaining}
Validation       : ✅ PASSED

Type YES to submit or NO to abort.
{'='*60}""")

    response = input("Submit? [YES/NO]: ").strip().upper()
    if response != "YES":
        print("🛑 Submission aborted by user.")
        return {"status": "ABORTED"}

    # ── Submit ────────────────────────────────────────────────
    from zindian.zindi_client import ZindiClient
    client  = ZindiClient()
    client.select_competition(config.slug)
    branch  = state.get("current_git_branch", "unknown")
    comment = (f"branch:{branch}"
               f"|oof_auc:{best_auc:.4f}"
               f"|oof_f1:{best_f1:.4f}"
               f"|features:{state.get('best_variant_features','?')}"
               f"|calib:none")

    print(f"\nSubmitting with comment: {comment}")
    result = client.submit(filepath=str(sub_path), comment=comment)

    # ── Update state ──────────────────────────────────────────
    store.update(
        submissions_used_today  = used_today + 1,
        submissions_used_total  = int(state.get("submissions_used_total") or 0) + 1,
        last_updated            = datetime.now(timezone.utc).isoformat(),
    )

    # ── Log to submission_log.md ──────────────────────────────
    log_path = paths.reports_dir / "submission_log.md"
    now      = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    entry    = (f"\n## Submission [{now}]\n"
                f"**File**: {sub_path.name}\n"
                f"**Comment**: {comment}\n"
                f"**Result**: {json.dumps(result)}\n")
    with open(log_path, "a") as f:
        f.write(entry)

    print(f"\n✅ Submitted. Result: {result}")
    print(f"✅ Logged → {log_path}")
    return {"status": "SUBMITTED", "result": result, "comment": comment}

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m zindian.skills.skill_16_submit <submission_file>")
        sys.exit(1)
    print(json.dumps(run(sys.argv[1]), indent=2, default=str))
