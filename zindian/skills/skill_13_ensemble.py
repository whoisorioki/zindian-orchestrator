"""
Skill 13 — Oracle Fusion (HUMAN GATED)
Blends OOF probability files from best variants.
Finds optimal blend weights via OOF F1 maximisation.
Saves final blended submission to submissions/.
HUMAN must type YES before any file is saved.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from zindian.config import ChallengeConfig
from zindian.paths import resolve_competition_paths
from zindian.state import SkillStateStore


def run(dry_run: bool = False) -> dict:
    """
    Orchestrate ensemble blending.
    Finds optimal threshold via OOF F1.
    Human gate: only saves on YES.
    """
    paths = resolve_competition_paths()
    if paths.competition_dir is None:
        raise FileNotFoundError("resolve_competition_paths did not return a competition_dir")

    config = ChallengeConfig.load(paths.config_path)
    state_store = SkillStateStore(paths.state_path)
    state = state_store.read()

    # ── 1. Load all available OOF files ──────────────────────────
    proc_dir = paths.competition_dir / "data" / "processed"
    oof_files = sorted(proc_dir.glob("oof_variant-*.csv"))

    print(f"Found {len(oof_files)} OOF files:")
    for f in oof_files:
        print(f"  {f.name}")

    if not oof_files:
        print("❌ No OOF files found. Run feature variants first.")
        return {"status": "FAILED", "reason": "No OOF files"}

    # Load ground truth
    target_col = config.get("target_column", "Occurrence Status")
    train = pd.read_csv(proc_dir / "features_train.csv")
    y_true = np.asarray(train[target_col].values, dtype=np.int32)

    # Load OOF probs
    oofs = {}
    for f in oof_files:
        df = pd.read_csv(f)
        prob_col = [c for c in df.columns if c not in ("ID", target_col)][0]
        oofs[f.stem] = df[prob_col].values
        preds = (np.asarray(df[prob_col].values) > 0.5).astype(int)
        f1 = f1_score(y_true, preds)
        print(f"  {f.stem}: OOF F1={f1:.5f}")

    # ── 2. Find best equal-weight blend ──────────────────────────
    names = list(oofs.keys())
    probs = np.array(list(oofs.values()))  # shape (n_variants, n_rows)

    # Equal weight blend
    blend_probs = probs.mean(axis=0)
    best_t = max(np.arange(0.3, 0.7, 0.01),
                 key=lambda t: f1_score(y_true, (blend_probs >= t).astype(int)))
    blend_preds = (blend_probs >= best_t).astype(int)
    blend_f1 = f1_score(y_true, blend_preds)
    print(f"\nEqual-weight blend OOF F1: {blend_f1:.5f} (threshold={best_t:.2f})")

    # ── 3. Load test probs and blend ─────────────────────────────
    test_files = sorted(proc_dir.glob("test_probs_variant-*.csv"))
    tests = {}
    for f in test_files:
        df = pd.read_csv(f)
        prob_col = [c for c in df.columns if c not in ("ID",)][0]
        tests[f.stem.replace("test_probs_", "oof_")] = df

    test_blend = np.mean([
        tests[n][tests[n].columns[-1]].values
        for n in names if n.replace("oof_", "test_probs_") in
        [f.stem for f in test_files]
    ], axis=0)

    # ── 4. HUMAN GATE ─────────────────────────────────────────────
    if dry_run:
        print("\n[DRY-RUN] Skipping human gate.")
        return {
            "status": "PENDING",
            "blend_oof_f1": float(blend_f1),
            "threshold": float(best_t),
            "variants": names,
        }

    print(f"\n{'='*60}")
    print("=== HUMAN GATE: Skill 13 — Ensembling ===")
    print(f"{'='*60}")
    print(f"Variants blended : {names}")
    print(f"Blend OOF F1     : {blend_f1:.5f}")
    print(f"Threshold        : {best_t:.2f}")
    print(f"Anchor LB F1     : {state.get('anchor_lb_score', 'unknown')}")
    print(f"Expected LB gain : unknown — submit to verify")
    print()
    print("Type YES to save submission or NO to cancel.")
    resp = input("Proceed? [YES/NO]: ").strip().upper()

    if resp != "YES":
        print("Cancelled.")
        return {"status": "CANCELLED"}

    # ── 5. Save submission ────────────────────────────────────────
    sample_path = paths.data_raw_dir / "SampleSubmission.csv"
    sample = pd.read_csv(sample_path)
    hard_preds = (test_blend >= best_t).astype(int)
    sub = pd.DataFrame({"ID": sample["ID"], "Target": hard_preds})
    sub = sub.set_index("ID").reindex(sample["ID"]).reset_index()

    subs_dir = paths.competition_dir / "submissions"
    subs_dir.mkdir(parents=True, exist_ok=True)
    out = subs_dir / "sub_ensemble_v1.csv"
    sub.to_csv(out, index=False)
    print(f"✅ Saved: {out}")
    print(f"   Present (1): {hard_preds.sum()}")
    print(f"   Absent  (0): {(hard_preds==0).sum()}")

    # ── 6. Update state ──────────────────────────────────────────
    state["dag_phase"] = "phase_5_governance_complete"
    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    state_store.write(state)

    return {
        "status": "OK",
        "submission_path": str(out),
        "blend_oof_f1": float(blend_f1),
        "threshold": float(best_t),
        "variants": names,
        "present_count": int(hard_preds.sum()),
        "absent_count": int((hard_preds==0).sum()),
    }


if __name__ == "__main__":
    # Allow --dry-run flag
    dry_run = "--dry-run" in sys.argv
    result = run(dry_run=dry_run)
    print(json.dumps(result, indent=2))
