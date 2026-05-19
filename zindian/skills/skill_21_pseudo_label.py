"""
Skill 21 — Semi-Supervised Pseudo-Labeling
============================================

Semi-supervised learning via high-confidence pseudo-labels on unlabelled test set.
Model: LGB + RF 50/50 blend.
Locked parameters (do not tune):
  - CONF_POS: 0.85 (positive threshold)
  - CONF_NEG: 0.15 (negative threshold)
  - weight: 0.5 (pseudo-label sample weight vs real labels)
  - threshold: 0.46 (fixed prediction threshold)
  - seeds: [42, 123, 7]
  - cv: StratifiedKFold(5, shuffle=True, random_state=42)

Human gate: Predicted positive count must be in [1330, 1360].

Two problems served:
  Problem 1 (Generic Agent): Sample space augmentation via pseudo-labels (distinct from feature engineering)
  Problem 2 (EY Frogs):      Lift OOF F1 by leveraging unlabelled test rows

Architecture:
  - Skill 20 (Scientist): mutates feature space X (columns)
  - Skill 21 (Pseudo):    mutates sample space y (rows/labels) — THIS SKILL
  - Skill 09 (Calib):     rescales probabilities (no new rows/features)

Never merge these three layers. A generic agent must toggle each ON/OFF independently.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb

from zindian.paths import resolve_competition_paths
from zindian.config import ChallengeConfig
from zindian.state import SkillStateStore

# ── LOCKED PARAMETERS ─────────────────────────────────────────────────────────
CONF_POS = 0.85        # High-confidence positive threshold
CONF_NEG = 0.15        # High-confidence negative threshold
SAMPLE_WEIGHT = 0.5    # Pseudo-label sample weight vs real labels
THRESHOLD = 0.46       # Fixed prediction threshold (do not grid search)
SEEDS = [42, 123, 7]
N_SPLITS = 5
MAX_ITERATIONS = 4

LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1,
}

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1,
}


# ── DIAGNOSTIC GATE FUNCTION ──────────────────────────────────────────────────
def check_distribution_gate(
    preds: np.ndarray | Any,
    threshold: float = THRESHOLD,
) -> dict[str, Any]:
    """
    Validate predicted positive count is within [1330, 1360].
    Anchor (tfcawL75) = 1,340 positives.
    
    Returns dict with pass/fail status and diagnostic message.
    """
    GATE_MIN = 1330
    GATE_MAX = 1360
    ANCHOR = 1340

    # Ensure preds is np.ndarray
    preds_array: np.ndarray = np.asarray(preds, dtype=np.float64)
    labels = (preds_array >= threshold).astype(np.int32)
    pos_count: int = int(labels.sum())
    drift: int = pos_count - ANCHOR

    result: dict[str, Any] = {
        "pos_count": pos_count,
        "drift": drift,
        "gate_min": GATE_MIN,
        "gate_max": GATE_MAX,
        "anchor": ANCHOR,
        "passed": GATE_MIN <= pos_count <= GATE_MAX,
        "threshold": float(threshold),
    }

    if not result["passed"]:
        if pos_count > GATE_MAX:
            result["diagnosis"] = (
                f"❌ Over-prediction by {pos_count - GATE_MAX} samples. "
                f"Positive boundary inflated via pseudo-label feedback loop. "
                f"Remediation: raise threshold above {threshold:.2f}."
            )
        else:
            result["diagnosis"] = (
                f"❌ Under-prediction by {GATE_MIN - pos_count} samples. "
                f"Threshold likely over-optimized on OOF distribution. "
                f"Remediation: lower threshold below {threshold:.2f}."
            )
    else:
        result["diagnosis"] = (
            f"✅ Distribution matches anchor baseline (drift={drift:+d} samples). "
            f"Gate PASSED."
        )

    return result


def get_feature_cols(df: pd.DataFrame, drop_cols: set[str]) -> list[str]:
    """Extract feature columns, excluding ID, target, and coordinates."""
    return [c for c in df.columns if c not in drop_cols]


def train_ensemble_and_predict(
    X_train: np.ndarray,
    y_train: np.ndarray | Any,
    X_test: np.ndarray,
    feature_cols: list[str],
    sample_weight: np.ndarray | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Train LGB + RF ensemble (50/50 blend).
    Returns (oof_probs, test_probs, oof_auc).
    """
    # Ensure y_train is np.ndarray
    y_train_array: np.ndarray = np.asarray(y_train, dtype=np.int32)
    
    oof: np.ndarray = np.zeros(len(X_train), dtype=np.float64)
    preds: np.ndarray = np.zeros(len(X_test), dtype=np.float64)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_train_array)):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y_train_array[tr_idx], y_train_array[va_idx]
        w_tr = sample_weight[tr_idx] if sample_weight is not None else None

        # Scale for class imbalance
        pos_rate: float = float(y_tr.mean())
        scale_pos: float = (1.0 - pos_rate) / pos_rate if pos_rate > 0 else 1.0

        # Train LGB
        dtrain = lgb.Dataset(X_tr, y_tr, weight=w_tr, feature_name=feature_cols)
        dval = lgb.Dataset(X_va, y_va, reference=dtrain)

        model_lgb = lgb.train(
            {**LGB_PARAMS, "scale_pos_weight": scale_pos, "seed": seed},
            dtrain,
            num_boost_round=500,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )

        lgb_val: np.ndarray = np.array(model_lgb.predict(X_va), dtype=np.float64)
        lgb_test: np.ndarray = np.array(model_lgb.predict(X_test), dtype=np.float64)

        # Train RF
        model_rf = RandomForestClassifier(**RF_PARAMS)
        model_rf.fit(X_tr, y_tr, sample_weight=w_tr)

        rf_val: np.ndarray = np.array(model_rf.predict_proba(X_va)[:, 1], dtype=np.float64)
        rf_test: np.ndarray = np.array(model_rf.predict_proba(X_test)[:, 1], dtype=np.float64)

        # Blend 50/50
        oof[va_idx] = 0.5 * lgb_val + 0.5 * rf_val
        preds += (0.5 * lgb_test + 0.5 * rf_test) / N_SPLITS

    auc: float = float(roc_auc_score(y_train, oof))
    return oof, preds, auc


def best_f1_threshold(
    y_true: np.ndarray | Any, probs: np.ndarray | Any
) -> tuple[float, float]:
    """Find best F1 score and corresponding threshold."""
    # Ensure inputs are np.ndarray
    y_true_array: np.ndarray = np.asarray(y_true, dtype=np.int32)
    probs_array: np.ndarray = np.asarray(probs, dtype=np.float64)
    
    best_f1: float = 0.0
    best_t: float = 0.5
    for t_val in np.arange(0.3, 0.7, 0.01):
        f1: float = float(f1_score(y_true_array, (probs_array >= t_val).astype(int)))
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t_val)
    return round(best_f1, 5), round(best_t, 2)


def run(dry_run: bool = False) -> dict:
    """
    Run pseudo-labeling loop.

    Args:
        dry_run: If True, run but do not save/update state.

    Returns:
        Dict with status, best_oof_f1, best_iteration, positive_count.
    """
    print("\n" + "=" * 70)
    print("SKILL 21 — Semi-Supervised Pseudo-Labeling (LGB + RF Blend)")
    print("=" * 70 + "\n")

    # ── Setup ─────────────────────────────────────────────────────────────
    paths = resolve_competition_paths()
    config = ChallengeConfig.load()
    store = SkillStateStore(paths.state_path)
    state = store.read()

    drop_cols = {"ID", "Occurrence Status", "Latitude", "Longitude", "swe_min"}

    # Load data
    train_file = paths.data_processed_dir / "features_train.csv"
    test_file = paths.data_processed_dir / "features_test.csv"
    sample_sub_file = paths.data_raw_dir / "SampleSubmission.csv"

    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    sample_sub = pd.read_csv(sample_sub_file)

    feature_cols = get_feature_cols(train, drop_cols)
    feature_cols = [c for c in feature_cols if c in test.columns]

    X_labelled = train[feature_cols].values.astype(np.float32)
    y_labelled = train["Occurrence Status"].values.astype(np.int32)
    X_test = test[feature_cols].values.astype(np.float32)
    test_ids = test["ID"].values

    print(f"Labelled rows  : {len(X_labelled)}")
    print(f"Test rows      : {len(X_test)}")
    print(f"Features       : {len(feature_cols)}")
    print(f"Confidence thresholds: pos>={CONF_POS} | neg<={CONF_NEG}")
    print(f"Sample weight  : {SAMPLE_WEIGHT}")
    print(f"Prediction threshold: {THRESHOLD}")
    print(f"Seeds          : {SEEDS}")
    print()

    results = []
    test_probs_prev = None

    # ── Pseudo-label iterations ───────────────────────────────────────────
    for iteration in range(MAX_ITERATIONS + 1):
        print(f"{'=' * 70}")
        print(f"Iteration {iteration}")
        print(f"{'=' * 70}")

        if iteration == 0:
            # Baseline — labelled data only
            X_all = X_labelled
            y_all: np.ndarray = np.asarray(y_labelled, dtype=np.int32)
            w_all: np.ndarray | None = None
            print("Using labelled data only (baseline)")
        else:
            # Add high-confidence pseudo-labels
            if test_probs_prev is None:
                raise RuntimeError("No test probs from previous iteration")

            pseudo_mask_pos = test_probs_prev >= CONF_POS
            pseudo_mask_neg = test_probs_prev <= CONF_NEG
            pseudo_mask = pseudo_mask_pos | pseudo_mask_neg

            pseudo_X = X_test[pseudo_mask]
            pseudo_y: np.ndarray = (test_probs_prev[pseudo_mask] >= 0.5).astype(np.int32)
            pseudo_w: np.ndarray = np.full(len(pseudo_X), SAMPLE_WEIGHT, dtype=np.float64)
            real_w: np.ndarray = np.ones(len(X_labelled), dtype=np.float64)

            X_all = np.vstack([X_labelled, pseudo_X])
            y_all = np.concatenate([np.asarray(y_labelled, dtype=np.int32), pseudo_y]).astype(np.int32)
            w_all = np.concatenate([real_w, pseudo_w]).astype(np.float64)

            n_pos = pseudo_mask_pos.sum()
            n_neg = pseudo_mask_neg.sum()
            print(f"Pseudo-labels added: {n_pos} positive, {n_neg} negative "
                  f"({len(pseudo_X)} total)")
            print(f"Total training rows: {len(X_all)}")

        # ── Multi-seed ensemble ───────────────────────────────────────────
        oof_list: list[np.ndarray] = []
        test_list: list[np.ndarray] = []
        auc_list: list[float] = []

        for seed_val in SEEDS:
            print(f"  Training seed {seed_val}...")
            oof, test_pred, auc = train_ensemble_and_predict(
                X_all, y_all, X_test, feature_cols, w_all, seed=seed_val
            )
            oof_list.append(oof)
            test_list.append(test_pred)
            auc_list.append(auc)
            print(f"    ✓ Seed {seed_val} AUC: {auc:.5f}")

        # Average across seeds
        oof_probs: np.ndarray = np.mean(np.array(oof_list), axis=0).astype(np.float64)
        test_probs: np.ndarray = np.mean(np.array(test_list), axis=0).astype(np.float64)
        mean_auc: float = float(np.mean(auc_list))

        # Score on labelled portion
        oof_labelled: np.ndarray = oof_probs[: len(X_labelled)]
        oof_f1, _ = best_f1_threshold(y_labelled, oof_labelled)

        print(f"OOF AUC (mean {len(SEEDS)} seeds): {mean_auc:.5f}")
        print(f"OOF F1  (labelled portion):       {oof_f1:.5f}")

        results.append(
            {
                "iteration": iteration,
                "oof_auc": float(mean_auc),
                "oof_f1": float(oof_f1),
                "threshold": THRESHOLD,
                "train_rows": len(X_all),
            }
        )

        # ── Make hard predictions ─────────────────────────────────────────
        hard_preds: np.ndarray = (test_probs >= THRESHOLD).astype(np.int32)
        pos_count: int = int(hard_preds.sum())

        # Save submission
        target_col = sample_sub.columns[-1]
        sub = pd.DataFrame({"ID": test_ids, target_col: hard_preds})
        sub = sub.set_index("ID").reindex(sample_sub["ID"]).reset_index()
        out_path = paths.submissions_dir / f"variant-pseudo_iter{iteration}.csv"
        sub.to_csv(out_path, index=False)

        print(f"Submission saved: {out_path.name}")
        print(f"  Positive count : {pos_count} / {len(test_ids)}")
        print(f"  Negative count : {len(test_ids) - pos_count} / {len(test_ids)}")

        test_probs_prev = test_probs

        # Early stop if no improvement
        if iteration >= 2:
            delta = results[-1]["oof_f1"] - results[-2]["oof_f1"]
            print(f"Delta OOF F1 vs prev: {delta:+.5f}")
            if delta <= 0:
                print("No improvement — stopping early.")
                break

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("PSEUDO-LABEL ITERATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Iter':>5} {'OOF AUC':>10} {'OOF F1':>10} {'Threshold':>10} {'Train rows':>12}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['iteration']:>5} {r['oof_auc']:>10.5f} {r['oof_f1']:>10.5f} "
            f"{r['threshold']:>10.2f} {r['train_rows']:>12}"
        )

    best_result = max(results, key=lambda x: x["oof_f1"])
    best_iteration = int(best_result["iteration"])
    best_oof_f1 = float(best_result["oof_f1"])

    best_sub_path = paths.submissions_dir / f"variant-pseudo_iter{best_iteration}.csv"
    best_sub = pd.read_csv(best_sub_path)
    target_col = best_sub.columns[-1]
    best_pos_count = int((best_sub[target_col] == 1).sum())

    print(f"\nBest iteration: {best_iteration} — OOF F1 {best_oof_f1:.5f}")
    print(f"Positive count: {best_pos_count}")

    # ── DIAGNOSTIC GATE CHECK ─────────────────────────────────────────────
    # Load best submission to get predictions for gate check
    best_sub_data = pd.read_csv(best_sub_path)
    gate_report = check_distribution_gate(
        preds=np.asarray(best_sub_data.iloc[:, -1].values, dtype=np.float64),
        threshold=THRESHOLD,
    )

    # ── HUMAN GATE ────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("=== HUMAN GATE: Skill 21 — Distribution Validation ===")
    print(f"{'=' * 70}")
    print(f"Best iteration     : {best_iteration}")
    print(f"Best OOF F1        : {best_oof_f1:.5f}")
    print(f"Predicted positive : {gate_report['pos_count']}")
    print(f"Anchor baseline    : {gate_report['anchor']}")
    print(f"Drift from anchor  : {gate_report['drift']:+d}")
    print(f"Valid range        : [{gate_report['gate_min']}, {gate_report['gate_max']}]")
    print(f"Status             : {'✅ PASS' if gate_report['passed'] else '❌ FAIL'}")
    print(f"Diagnosis          : {gate_report['diagnosis']}")
    print(f"{'=' * 70}")

    if not gate_report["passed"]:
        print(f"\n❌ GATE BLOCKED")
        if not dry_run:
            print("Aborting. Do not submit.")
        return {
            "status": "BLOCKED",
            "reason": "positive_count_out_of_range",
            "positive_count": gate_report['pos_count'],
            "best_iteration": best_iteration,
            "best_oof_f1": best_oof_f1,
            "diagnosis": gate_report['diagnosis'],
        }

    if dry_run:
        print("\n[DRY RUN] Gate passed. Would proceed to submission.")
        return {
            "status": "GATE_PASSED_DRY_RUN",
            "best_iteration": best_iteration,
            "best_oof_f1": best_oof_f1,
            "positive_count": gate_report['pos_count'],
            "gate_report": gate_report,
        }

    # ── Human confirmation ────────────────────────────────────────────────
    response = (
        input(
            f"\n✅ Gate PASSED. Submit {best_sub_path.name}? [YES/NO]: "
        )
        .strip()
        .upper()
    )
    if response != "YES":
        print("🛑 Submission aborted by user.")
        return {"status": "ABORTED"}

    # ── Update state ──────────────────────────────────────────────────────
    store.update(
        pseudo_label_oof_f1=best_oof_f1,
        pseudo_label_iteration=best_iteration,
        pseudo_label_submission_file=best_sub_path.name,
        pseudo_label_positive_count=best_pos_count,
        last_updated=datetime.now(timezone.utc).isoformat(),
    )

    print(f"\n✅ SKILL_STATE.json updated with pseudo-label results")
    print(f"Ready for Skill 16 submission → {best_sub_path.name}")

    return {
        "status": "READY_TO_SUBMIT",
        "best_iteration": best_iteration,
        "best_oof_f1": best_oof_f1,
        "positive_count": best_pos_count,
        "submission_file": best_sub_path.name,
    }


def main() -> None:
    """CLI entry point."""
    dry_run = "--dry-run" in sys.argv

    if dry_run:
        print("[DRY RUN MODE]")

    result = run(dry_run=dry_run)
    print(f"\nResult: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()
