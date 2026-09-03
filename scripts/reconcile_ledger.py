"""Repair the submissions ledger and gate baseline after the reconcile incident.

Non-destructive by design: performs targeted UPDATEs only, never DELETE.
Every repaired value carries an explicit provenance note.

Provenance of my_rank / experiment_id values: the pre-reconcile submissions
ledger (recorded 2026-09-02 by skill_16 at submit time, captured verbatim in
the 2026-09-03 audit review). Rows 1-6 predate the orchestrator (manual
submissions) and rows 7-8 predate rank capture - their my_rank is genuinely
unknown and is deliberately left NULL rather than invented.

OOF baseline provenance: experiments.exp6.notes -
"oof_f1=0.817195; oof_auc=0.813342; cv_strategy_id=config:BufferedSpatialCV"
(the anchor-baseline confirmation of record). anchor_oof_f1 is stored at full
original precision (0.8171949630916197 - the pre-remediation anchor_oof_score
value, which equaled oof_f1).
"""

import json
from pathlib import Path

import duckdb

DB_PATH = Path(
    "competitions/climate-risk-health-prediction-challenge/reports/experiments.db"
)
MANIFEST_PATH = Path(
    "competitions/climate-risk-health-prediction-challenge/reports/"
    "submissions_manifest.json"
)
FINAL_SELECTIONS_PATH = Path(
    "competitions/climate-risk-health-prediction-challenge/reports/audits/"
    "final_selections.json"
)
STATE_PATH = Path(
    "competitions/climate-risk-health-prediction-challenge/SKILL_STATE.json"
)

# zindi_id -> (my_rank or None, experiment_id or None, branch_name or None-to-keep)
PROVENANCE = {
    # Pre-orchestrator manual submissions (Aug 31) - no ledger record existed.
    "zPnBJEdS": (None, None, None),
    "qouVDWN6": (None, None, None),
    "mX95uW5j": (None, None, None),
    "fpThxre4": (None, None, None),
    "FY8jAvnu": (None, None, "manual_pre_orchestrator"),
    "VWmbCkBN": (None, None, "manual_pre_orchestrator"),
    # Orchestrator-era rows - values from the pre-reconcile ledger.
    "TWM7P1xa": (None, 5, None),
    "9oXDE1j3": (None, 7, None),
    "YUFL12kq": (174, 8, None),
    "PzruUqvQ": (174, 9, None),
    "QVAuV4td": (174, 10, None),
    "CDjHj9Ct": (175, 11, None),
    "QF4PBpst": (175, 12, None),
    "gDcgMAQ1": (175, 13, None),
    "c2uw9GG8": (175, 14, None),
    "EGv7pdTK": (175, 15, None),
    "BEhrZLp2": (175, 16, None),
    "9GSXD4Ze": (175, 17, None),
}

# Recorded OOF baseline (experiments.exp6): f1 at full precision, auc 6dp.
ANCHOR_OOF_F1 = 0.8171949630916197
ANCHOR_OOF_AUC = 0.813342


def derive_gate5_ids(manifest: list[dict]) -> set[str]:
    """Derive Gate 5 zindi_ids from the audited selections file (not hardcoded)."""
    with FINAL_SELECTIONS_PATH.open(encoding="utf-8") as f:
        sel = json.load(f)
    locked_files = {
        str(item.get("filename", "")) for item in sel.get("selections", [])
    }
    return {
        str(item["zindi_id"])
        for item in manifest
        if str(item.get("filename", "")) in locked_files
    }


def reconcile_database() -> None:
    with MANIFEST_PATH.open(encoding="utf-8") as f:
        manifest = json.load(f)
    gate5_ids = derive_gate5_ids(manifest)
    print(f"[OK] Gate 5 locked zindi_ids (from audits/final_selections.json): {sorted(gate5_ids)}")

    con = duckdb.connect(str(DB_PATH))
    changes = 0
    for zindi_id, (my_rank, experiment_id, branch_name) in PROVENANCE.items():
        row = con.execute(
            "SELECT submission_id, my_rank, experiment_id, branch_name, "
            "selected_for_final, selection_rationale FROM submissions "
            "WHERE zindi_id = ?",
            [zindi_id],
        ).fetchone()
        if row is None:
            print(f"[WARN] no ledger row for {zindi_id} - skipping")
            continue
        sub_id, cur_rank, cur_exp, cur_branch, cur_sel, cur_rationale = row
        should_select = zindi_id in gate5_ids
        updates: list[str] = []
        params: list[object] = []
        # None provenance means genuinely unknown -> the field must be NULL,
        # never a fabricated default.
        if cur_rank != my_rank:
            updates.append("my_rank = ?")
            params.append(my_rank)
        if cur_exp != experiment_id:
            updates.append("experiment_id = ?")
            params.append(experiment_id)
        if branch_name is not None and cur_branch != branch_name:
            updates.append("branch_name = ?")
            params.append(branch_name)
        if bool(cur_sel) != should_select:
            updates.append("selected_for_final = ?")
            params.append(should_select)
            updates.append("selection_rationale = ?")
            params.append(
                "Human Gate 5 locked selection" if should_select else None
            )
        if updates:
            params.append(sub_id)
            con.execute(
                f"UPDATE submissions SET {', '.join(updates)} WHERE submission_id = ?",
                params,
            )
            changes += 1
            print(f"[FIX] {zindi_id}: {', '.join(u.split(' =')[0] for u in updates)}")
    con.commit()

    # Gate baseline decontamination: anchor_oof_auc previously held a
    # leaderboard value (0.827310924 == qouVDWN6 lb_auc). Restore the
    # recorded OOF values from experiments.exp6.
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    score = 0.6 * ANCHOR_OOF_F1 + 0.4 * ANCHOR_OOF_AUC
    if abs(state.get("anchor_oof_auc", -1) - ANCHOR_OOF_AUC) > 1e-9:
        state["anchor_oof_f1"] = ANCHOR_OOF_F1
        state["anchor_oof_auc"] = ANCHOR_OOF_AUC
        state["anchor_oof_score"] = score
        STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(f"[FIX] SKILL_STATE baseline -> OOF composite {score:.16f} "
              f"(f1={ANCHOR_OOF_F1}, auc={ANCHOR_OOF_AUC}; provenance: experiments.exp6)")
    con.close()
    print(f"[OK] Reconcile complete. Rows updated: {changes}")


if __name__ == "__main__":
    reconcile_database()
