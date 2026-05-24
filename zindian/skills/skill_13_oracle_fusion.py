"""
Skill 13 — Oracle Fusion (HUMAN GATED)
Blends OOF probability files from the top N variants by OOF F1.
Finds optimal blend threshold via OOF F1 search.
Saves blended submission to submissions/.
HUMAN must type YES before any file is saved.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from zindian.config import ChallengeConfig
from zindian.paths import resolve_competition_paths
from zindian.state import SkillStateStore

TOP_N = 3  # Number of variants to blend


def _load_oof_files(proc_dir: Path, reports_dir: Path, y_true: np.ndarray) -> list[dict]:
    """Load all OOF files and score them. Returns list sorted by OOF F1 desc."""
    oof_files = sorted({
        *proc_dir.glob("oof_variant-*.csv"),
        *reports_dir.glob("oof_probs_blend_iter*.csv"),
        *reports_dir.glob("oof_probs_pseudo_iter*.csv"),
    })
    if not oof_files:
        return []

    def canonical_name(path: Path) -> str:
        stem = path.stem
        if stem.startswith("oof_probs_pseudo_iter"):
            return stem.replace("oof_probs_", "oof_variant-", 1)
        return stem

    deduped: dict[str, Path] = {}
    for path in oof_files:
        key = canonical_name(path)
        current = deduped.get(key)
        if current is None or current.suffix == ".csv" and current.parent != proc_dir and path.parent == proc_dir:
            deduped[key] = path

    scored = []
    for f in deduped.values():
        df = pd.read_csv(f)
        prob_col = [c for c in df.columns if c != "ID"][0]
        probs = np.asarray(df[prob_col].values, dtype=np.float64)

        best_f1, best_t = 0.0, 0.5
        for t in np.arange(0.3, 0.7, 0.01):
            f1 = float(f1_score(y_true, (probs >= float(t)).astype(int)))
            if f1 > best_f1:
                best_f1 = float(f1)
                best_t = float(t)

        auc = roc_auc_score(y_true, probs)
        scored.append({
            "name":      f.stem,
            "probs":     probs,
            "oof_f1":    round(best_f1, 5),
            "oof_auc":   round(auc, 5),
            "threshold": round(best_t, 2),
        })

    return sorted(scored, key=lambda x: -x["oof_f1"])


def _candidate_test_names(oof_name: str) -> list[str]:
    if "pseudo_iter" in oof_name:
        suffix = oof_name.split("pseudo_iter", 1)[1]
        return [f"test_probs_pseudo_iter{suffix}"]
    if oof_name.startswith("oof_probs_"):
        return [oof_name.replace("oof_probs_", "test_probs_", 1)]
    if oof_name.startswith("oof_"):
        return [oof_name.replace("oof_", "test_probs_", 1)]
    return [f"test_probs_{oof_name}"]


def _check_test_files(proc_dir: Path, reports_dir: Path, names: list[str]) -> dict[str, Path] | None:
    """
    Map OOF variant names to their test prob files.
    Returns None if any are missing.
    """
    mapping = {}
    missing = []
    for name in names:
        test_path = None
        for test_name in _candidate_test_names(name):
            for base_dir in (proc_dir, reports_dir):
                candidate = base_dir / f"{test_name}.csv"
                if candidate.exists():
                    test_path = candidate
                    break
            if test_path is not None:
                break
        if test_path is None:
            missing.append(_candidate_test_names(name)[0])
        else:
            mapping[name] = test_path

    if missing:
        print(f"\n❌ Missing test prob files: {missing}")
        print("   Rerun those variants with --force-save before blending.")
        return None

    return mapping


def run(dry_run: bool = False) -> dict:
    print("\n" + "=" * 60)
    print("SKILL 13 — Oracle Fusion")
    print("=" * 60 + "\n")

    paths      = resolve_competition_paths(require_competition=True)
    config     = ChallengeConfig.load()
    store      = SkillStateStore(paths.state_path)
    state      = store.read()

    if paths.competition_dir is None:
        raise FileNotFoundError("Competition directory could not be resolved")

    proc_dir   = paths.competition_dir / "data" / "processed"
    reports_dir = paths.reports_dir
    subs_dir   = paths.competition_dir / "submissions"
    raw_dir    = paths.data_raw_dir

    target_col = config.get("target_column", "Occurrence Status")

    # ── 1. Load ground truth ─────────────────────────────────────
    train_path = proc_dir / "features_train.csv"
    if not train_path.exists():
        print(f"❌ Train file not found: {train_path}")
        return {"status": "FAILED", "reason": "Train file missing"}

    train  = pd.read_csv(train_path)
    y_true = train[target_col].values.astype(np.int32)
    print(f"Train rows    : {len(y_true)}")
    print(f"Positive rate : {y_true.mean():.3f}")

    # ── 2. Score all OOF files ───────────────────────────────────
    all_variants = _load_oof_files(proc_dir, reports_dir, y_true)
    if not all_variants:
        print("❌ No OOF files found. Run feature variants first.")
        return {"status": "FAILED", "reason": "No OOF files"}

    print(f"\nAll variants ranked by OOF F1 (found {len(all_variants)}):")
    print(f"  {'Variant':<30} {'OOF AUC':>10} {'OOF F1':>10} {'Threshold':>10}")
    print("  " + "-" * 62)
    for v in all_variants:
        print(f"  {v['name']:<30} {v['oof_auc']:>10} {v['oof_f1']:>10} {v['threshold']:>10}")

    # ── 3. Select top N ──────────────────────────────────────────
    selected = all_variants[:TOP_N]
    names    = [v["name"] for v in selected]
    print(f"\nSelected top {TOP_N} for blend: {names}")

    # ── 4. Check test prob files exist ───────────────────────────
    test_map = _check_test_files(proc_dir, reports_dir, names)
    if test_map is None:
        return {"status": "FAILED", "reason": "Missing test prob files"}

    # ── 5. Blend OOF probs and find optimal threshold ────────────
    oof_matrix  = np.array([v["probs"] for v in selected])
    blend_probs = oof_matrix.mean(axis=0)

    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.3, 0.7, 0.01):
        f1 = f1_score(y_true, (blend_probs >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t

    blend_auc = roc_auc_score(y_true, blend_probs)
    anchor_f1 = float(state.get("anchor_oof_f1") or 0.0)
    anchor_lb = float(state.get("anchor_lb_score") or 0.0)

    print(f"\nBlend OOF AUC  : {blend_auc:.5f}")
    print(f"Blend OOF F1   : {best_f1:.5f}  (threshold={best_t:.2f})")
    print(f"Anchor OOF F1  : {anchor_f1:.5f}")
    print(f"Anchor LB F1   : {anchor_lb}")
    print(f"Delta OOF F1   : {best_f1 - anchor_f1:+.5f}")

    # ── 6. Blend test probs ──────────────────────────────────────
    test_matrices = []
    for name, test_path in test_map.items():
        df = pd.read_csv(test_path)
        prob_col = [c for c in df.columns if c != "ID"][0]
        test_matrices.append(df[prob_col].values)

    test_blend  = np.array(test_matrices).mean(axis=0)
    hard_preds  = (test_blend >= best_t).astype(int)

    print(f"\nTest predictions: {hard_preds.sum()} present, "
          f"{(hard_preds == 0).sum()} absent")

    # ── 7. Dry-run exit ──────────────────────────────────────────
    if dry_run:
        print("\n[DRY-RUN] No files saved.")
        return {
            "status":       "DRY_RUN",
            "blend_oof_f1": float(best_f1),
            "blend_oof_auc": float(blend_auc),
            "threshold":    float(best_t),
            "variants":     names,
            "present_count": int(hard_preds.sum()),
            "absent_count":  int((hard_preds == 0).sum()),
        }

    # ── 8. Human gate ────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("HUMAN GATE — type YES to save, anything else cancels")
    print(f"{'=' * 60}")
    resp = input("Proceed? [YES/NO]: ").strip().upper()
    if resp != "YES":
        print("Cancelled.")
        return {"status": "CANCELLED"}

    # ── 9. Load sample submission and validate row count ─────────
    sample_path = raw_dir / "SampleSubmission.csv"
    if not sample_path.exists():
        print(f"❌ SampleSubmission.csv not found at {sample_path}")
        return {"status": "FAILED", "reason": "SampleSubmission missing"}

    sample = pd.read_csv(sample_path)
    if len(hard_preds) != len(sample):
        print(f"❌ Row count mismatch: preds={len(hard_preds)}, sample={len(sample)}")
        return {"status": "FAILED", "reason": "Row count mismatch"}

    # ── 10. Save submission ───────────────────────────────────────
    target_out_col = sample.columns[-1]  # Use whatever column name sample uses
    sub = pd.DataFrame({"ID": sample["ID"], target_out_col: hard_preds})
    sub = sub.set_index("ID").reindex(sample["ID"]).reset_index()

    subs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path  = subs_dir / f"sub_ensemble_v1_{timestamp}.csv"
    sub.to_csv(out_path, index=False)
    print(f"\n✅ Saved: {out_path}")

    # ── 11. Update state ─────────────────────────────────────────
    state["last_ensemble_path"]    = str(out_path)
    state["last_ensemble_oof_f1"]  = float(best_f1)
    state["last_ensemble_oof_auc"] = float(blend_auc)
    state["last_ensemble_threshold"] = float(best_t)
    state["last_ensemble_variants"]  = names
    state["last_ensemble_threshold"] = round(float(best_t), 2)
    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    # Record SoT v1.7 gate approval for Skill 13 (ensembling)
    state["human_gate_3_approved"] = True
    store.write(state)

    return {
        "status":          "OK",
        "submission_path": str(out_path),
        "blend_oof_f1":    float(best_f1),
        "blend_oof_auc":   float(blend_auc),
        "threshold":       float(best_t),
        "variants":        names,
        "present_count":   int(hard_preds.sum()),
        "absent_count":    int((hard_preds == 0).sum()),
    }


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    result  = run(dry_run=dry_run)
    print("\n" + json.dumps(result, indent=2))