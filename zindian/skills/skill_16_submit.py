"""
Skill 16 — Submission Governance
Validates submission file, human gates, submits, updates state.
Post-submission: pulls rank and top 20 leaderboard automatically.
"""
from __future__ import annotations
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast
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
        errors.append("ID set mismatch vs SampleSubmission")
    if list(sub["ID"].astype(str)) != list(sample["ID"].astype(str)):
        errors.append("ID order mismatch vs SampleSubmission")
    if sub.isnull().any().any():
        errors.append(f"Nulls in: {sub.columns[sub.isnull().any()].tolist()}")
    return errors


def determine_submission_metrics(submission_file: Path, state: dict[str, Any]) -> tuple[float, str]:
    file_name = submission_file.name.lower()
    stem = submission_file.stem.lower()

    candidate_keys: list[str] = []

    if "ensemble" in file_name:
        candidate_keys.extend([
            "last_ensemble_oof_f1",
            "best_ensemble_oof_f1",
            "last_variant_oof_f1",
            "best_variant_oof_f1",
            "anchor_oof_f1",
        ])

    if "anchor" in file_name:
        candidate_keys.extend([
            "anchor_oof_f1",
            "best_variant_oof_f1",
            "last_variant_oof_f1",
        ])

    variant_match = re.search(r"(variant-[\w\d_]+)", stem)
    if variant_match:
        variant_tag = variant_match.group(1)
        candidate_keys.extend([
            f"{variant_tag}_oof_f1",
            f"last_{variant_tag}_oof_f1",
            f"best_{variant_tag}_oof_f1",
        ])

    candidate_keys.extend([
        "last_variant_oof_f1",
        "best_variant_oof_f1",
        "last_ensemble_oof_f1",
        "best_ensemble_oof_f1",
        "anchor_oof_f1",
    ])

    for key in candidate_keys:
        value = state.get(key)
        if value is not None:
            try:
                return float(value), key
            except (TypeError, ValueError):
                continue

    fallback_value = (
        state.get("last_variant_oof_f1")
        or state.get("best_variant_oof_f1")
        or state.get("anchor_oof_f1")
        or state.get("anchor_oof_rmse")
        or 0.0
    )
    return float(fallback_value), "best_variant_oof_f1"


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
    remaining  = int(state.get("remaining_submissions") or 10)
    used_today = int(state.get("submissions_used_today") or 0)
    print(f"\nBudget: {remaining} remaining | {used_today} used today")
    if remaining <= 2:
        print("❌ BUDGET GUARD — fewer than 2 submissions remaining. Aborting.")
        return {"status": "BLOCKED", "reason": "budget"}

    # ── Human gate ────────────────────────────────────────────
    best_auc = state.get("best_variant_oof_auc") or state.get("last_ensemble_oof_auc") or state.get("anchor_oof_auc")
    best_f1, metric_source = determine_submission_metrics(sub_path, state)
    branch   = state.get("current_git_branch", "unknown")

    print(f"""
{'='*60}
=== HUMAN GATE: Skill 16 — Submit ===
{'='*60}
File             : {sub_path.name}
Branch           : {branch}
OOF F1           : {best_f1}
Reference ROC-AUC: {best_auc}
Metric source    : {metric_source}
Remaining today  : {remaining}
Validation       : ✅ PASSED

Type YES to submit or NO to abort.
{'='*60}""")

    response = input("Submit? [YES/NO]: ").strip().upper()
    if response != "YES":
        print("🛑 Submission aborted by user.")
        return {"status": "ABORTED"}

    # ── Connect + select competition ──────────────────────────
    from zindian.zindi_client import ZindiClient
    client  = ZindiClient()
    client.select_competition(config.slug)

    feature_count = state.get("last_ensemble_features") or state.get("best_variant_features") or state.get("terraclimate_n_bands") or "?"
    comment = (f"branch:{branch}"
               f"|oof_f1:{best_f1:.4f}"
               f"|features:{feature_count}"
               f"|calib:none")

    print(f"\nSubmitting with comment: {comment}")
    result = client.submit(filepath=str(sub_path), comment=comment)

    # ── Update state ──────────────────────────────────────────
    store.update(
        submissions_used_today = used_today + 1,
        submissions_used_total = int(state.get("submissions_used_total") or 0) + 1,
        last_updated           = datetime.now(timezone.utc).isoformat(),
    )

    # ── Log to submission_log.md ──────────────────────────────
    log_path = paths.reports_dir / "submission_log.md"
    now      = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    entry    = (f"\n## Submission [{now}]\n"
                f"**File**: {sub_path.name}\n"
                f"**Branch**: {branch}\n"
                f"**Comment**: {comment}\n"
                f"**Result**: {json.dumps(result)}\n")
    with open(log_path, "a") as f:
        f.write(entry)

    print(f"\n✅ Submitted. Result: {result}")
    print(f"✅ Logged → {log_path}")

    # ── Post-submission: rank + leaderboard ──────────────────
    print(f"\n{'='*60}")
    print("POST-SUBMISSION RESULTS")
    print(f"{'='*60}")
    try:
        my_rank         = client._user.my_rank
        remaining_after = client.remaining_submissions
        print(f"Current rank     : {my_rank}")
        print(f"Remaining today  : {remaining_after}")
        print("\n--- Top 20 Leaderboard ---")
        client.leaderboard(per_page=20)
        store.update(
            anchor_rank  = my_rank,
            last_updated = datetime.now(timezone.utc).isoformat(),
        )
    except Exception as e:
        print(f"⚠️  Could not fetch leaderboard: {e}")

    return {"status": "SUBMITTED", "result": result, "comment": comment}


def show_submission_board() -> None:
    import io, sys
    from zindian.zindi_client import ZindiClient
    from zindian.config import ChallengeConfig
    config = ChallengeConfig.load()
    client = ZindiClient()
    client.select_competition(config.slug)
    _buf = io.StringIO()
    _old = sys.stdout
    sys.stdout = _buf
    subs = cast(list[dict[str, Any]], list(client._user.submission_board()))
    sys.stdout = _old
    clean = []
    for s in subs:
        clean.append({
            "id": s.get("id"),
            "date": str(s.get("created_at", ""))[:10],
            "file": s.get("filename"),
            "lb_f1": s.get("public_score"),
            "status": s.get("status"),
            "chosen": s.get("chosen"),
            "comment": s.get("comment"),
        })
    col_id   = 12
    col_date = 12
    col_f1   = 13
    col_ch   = 6
    col_file = 40
    sep = "-" * 150
    hdr = f"{'ID':{col_id}} {'Date':{col_date}} {'LB F1':>{col_f1}} {'Ch':>{col_ch}}  {'File':{col_file}} Comment"
    print(hdr)
    print(sep)
    for s in clean:
        chosen = "YES" if s["chosen"] else "   "
        f1     = f"{s['lb_f1']:.9f}" if s["lb_f1"] else "0.000000000"
        row = f"{s['id']:{col_id}} {s['date']:{col_date}} {f1:>{col_f1}} {chosen:>{col_ch}}  {s['file']:{col_file}} {s['comment']}"
        print(row)
    print(sep)


if __name__ == "__main__":
    import sys
    if "--submission-board" in sys.argv:
        show_submission_board()
    elif len(sys.argv) < 2:
        print("Usage:")
        print("  python -m zindian.skills.skill_16_submit <file>")
        print("  python -m zindian.skills.skill_16_submit --submission-board")
        sys.exit(1)
    else:
        print(json.dumps(run(sys.argv[1]), indent=2, default=str))
