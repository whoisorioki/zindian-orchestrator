"""
Skill 08 — Anchor Baseline
Train LightGBM baseline on raw features (Latitude, Longitude).
Lock first confirmed anchor submission and create git branch.
Must run after Skill 07 feature engineering completes.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from zindian.config import ChallengeConfig
from zindian.paths import resolve_competition_paths
from zindian.state import SkillStateStore
from zindian.ledger import Ledger


def load_data(paths) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Load training and test data, return (train, test, target_col)."""
    train = pd.read_csv(paths.data_raw_dir / "Training_Data.csv")
    test  = pd.read_csv(paths.data_raw_dir / "Test.csv")

    # Training target column (actual label column in Training_Data.csv)
    target_col = "Occurrence Status"

    # Submission target column (column name Zindi expects in submission file)
    sample = pd.read_csv(paths.data_raw_dir / "SampleSubmission.csv")
    submission_col = [c for c in sample.columns if c.upper() != "ID"][0]

    return train, test, target_col, submission_col


def compute_oof_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: ChallengeConfig,
    target_col: str,
    n_splits: int = 5,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Train LightGBM with KFold cross-validation.
    Returns (oof_preds, test_preds, oof_logloss, oof_auc).
    """
    from sklearn.metrics import log_loss, roc_auc_score

    np.random.seed(random_seed)

    feature_cols = ["Latitude", "Longitude"]

    X      = train[feature_cols].values.astype(np.float32)
    y      = train[target_col].values.astype(np.int32)
    X_test = test[feature_cols].values.astype(np.float32)

    scaler = StandardScaler()
    X      = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)

    oof_preds  = np.zeros(len(train), dtype=np.float64)
    test_preds = np.zeros(len(test),  dtype=np.float64)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        params = {
            "objective":     "binary",
            "metric":        "binary_logloss",
            "learning_rate": 0.05,
            "num_leaves":    31,
            "verbose":       -1,
            "seed":          random_seed + fold_idx,
        }

        train_set = lgb.Dataset(X_tr, label=y_tr)
        val_set   = lgb.Dataset(X_val, label=y_val, reference=train_set)

        model = lgb.train(
            params,
            train_set,
            num_boost_round=500,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(period=-1)],
        )

        val_pred           = model.predict(X_val)
        oof_preds[val_idx] = val_pred
        test_preds        += model.predict(X_test) / n_splits

        fold_ll  = log_loss(y_val, val_pred)
        fold_auc = roc_auc_score(y_val, val_pred)
        print(f"  Fold {fold_idx + 1}/{n_splits}: logloss={fold_ll:.6f}  auc={fold_auc:.6f}")

    oof_logloss = float(log_loss(y, oof_preds))
    oof_auc     = float(roc_auc_score(y, oof_preds))
    print(f"\nOOF Log Loss : {oof_logloss:.6f}")
    print(f"OOF AUC      : {oof_auc:.6f}")

    return oof_preds, test_preds, oof_logloss, oof_auc


def save_submission(
    test_ids:    np.ndarray,
    predictions: np.ndarray,
    target_col:  str,           # ← passed in explicitly, not a free variable
    output_path: Path,
) -> None:
    """Save predictions in Zindi submission format (probabilities)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sub_df = pd.DataFrame({
        "ID":      np.asarray(test_ids),
        target_col: np.asarray(predictions, dtype=np.float64),
    })
    sub_df.to_csv(output_path, index=False)
    print(f"✅ Submission saved → {output_path}")


def create_git_branch(branch_name: str = "anchor-baseline") -> None:
    """Create and switch to the anchor git branch."""
    import subprocess

    try:
        subprocess.run(
            ["git", "checkout", "-b", branch_name],
            check=True, capture_output=True,
        )
        print(f"✅ Git branch created: {branch_name}")
    except subprocess.CalledProcessError:
        try:
            subprocess.run(
                ["git", "checkout", branch_name],
                check=True, capture_output=True,
            )
            print(f"✓  Switched to existing branch: {branch_name}")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Git branch operation failed: {e.stderr.decode()}")


def run(*, n_splits: int = 5, random_seed: int = 42) -> dict:
    """
    Skill 08 — Anchor Baseline.

    Returns dict with status, oof_logloss, oof_auc, submission path, git branch.
    """
    print(f"\n{'='*60}")
    print(f"SKILL 08 — Anchor Baseline (LightGBM · Lat/Lon only)")
    print(f"{'='*60}\n")

    paths  = resolve_competition_paths()
    config = ChallengeConfig.load()

    print(f"Competition      : {config.slug}")
    print(f"Metric           : {config.metric}")
    print(f"Use probabilities: {config.use_probabilities}")

    # ── Load data ──────────────────────────────────────────────
    train, test, target_col, submission_col = load_data(paths)
    print(f"\nData loaded:")
    print(f"  Train  : {train.shape}")
    print(f"  Test   : {test.shape}")
    print(f"  Target : {target_col}")

    # ── Train ──────────────────────────────────────────────────
    print(f"\nTraining LightGBM anchor baseline…")
    oof_preds, test_preds, oof_logloss, oof_auc = compute_oof_predictions(
        train, test, config, target_col,
        n_splits=n_splits, random_seed=random_seed,
    )

    # ── Save OOF predictions ───────────────────────────────────
    oof_path = paths.data_raw_dir / "oof_anchor.csv"
    pd.DataFrame({
        "ID":        train["ID"],
        "Predicted": oof_preds,
        target_col:  train[target_col],
    }).to_csv(oof_path, index=False)
    print(f"✅ OOF predictions saved → {oof_path}")

    # ── Save submission ────────────────────────────────────────
    sub_path = paths.submissions_dir / "sub_001_anchor.csv"
    save_submission(test["ID"].values, test_preds, submission_col, sub_path)

    # ── Log to DuckDB ledger ───────────────────────────────────
    ledger = Ledger()
    exp_id = ledger.log_experiment(
        branch_name="anchor-baseline",
        oof_rmse=oof_logloss,           # stored as oof_rmse per schema convention
        feature_count=2,
        calibration_method="none",
        gate_result="PASS",
        gate_reason="Initial anchor baseline — Lat/Lon only, compliant",
        dag_phase="phase_2_anchor_confirmed",
    )
    ledger.close()
    print(f"✅ Experiment logged → DuckDB exp_id={exp_id}")

    # ── Update SKILL_STATE.json ────────────────────────────────
    state_store = SkillStateStore(paths.state_path)
    state_store.update(
        anchor_oof_rmse=oof_logloss,
        anchor_oof_auc=oof_auc,
        anchor_git_branch="anchor-baseline",
        dag_phase="phase_2_anchor_confirmed",
        last_updated=datetime.now(timezone.utc).isoformat(),
    )
    print(f"✅ SKILL_STATE.json updated: logloss={oof_logloss:.6f}  auc={oof_auc:.6f}")

    # ── Create git branch ──────────────────────────────────────
    create_git_branch("anchor-baseline")

    return {
        "status":          "OK",
        "oof_logloss":     oof_logloss,
        "oof_auc":         oof_auc,
        "submission_path": str(sub_path),
        "git_branch":      "anchor-baseline",
        "n_features":      2,
        "message":         "Anchor baseline trained and locked",
    }


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))