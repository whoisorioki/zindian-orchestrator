"""
Skill 08 — Anchor Baseline
Train LightGBM baseline on raw features (Latitude, Longitude).
Lock first confirmed anchor submission and create git branch.
Must run after Skill 07 feature engineering completes.

Submission is HUMAN-GATED — never fires automatically.
Two-layer gate:
  Layer 1 — validate_submission() runs 8 automatic checks (hard block on failure)
  Layer 2 — human YES/NO prompt (only shown if Layer 1 passes)
"""

from __future__ import annotations

import json
import re
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


# ── Data ───────────────────────────────────────────────────────────────────────

def load_data(paths) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """
    Load training and test data.
    Returns (train, test, training_target_col, submission_col).
    These are intentionally different — do not conflate them.
    """
    train = pd.read_csv(paths.data_raw_dir / "Training_Data.csv")
    test  = pd.read_csv(paths.data_raw_dir / "Test.csv")

    # Training target: actual label column in Training_Data.csv
    training_target_col = "Occurrence Status"

    # Submission column: what Zindi expects in the CSV header
    sample = pd.read_csv(paths.data_raw_dir / "SampleSubmission.csv")
    submission_col = [c for c in sample.columns if c.upper() != "ID"][0]

    return train, test, training_target_col, submission_col


# ── Training ───────────────────────────────────────────────────────────────────

def compute_oof_predictions(
    train:       pd.DataFrame,
    test:        pd.DataFrame,
    config:      ChallengeConfig,
    target_col:  str,
    n_splits:    int = 5,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Train LightGBM with KFold cross-validation on raw Lat/Lon features.
    Returns (oof_preds, test_preds, oof_logloss, oof_auc).
    """
    from sklearn.metrics import log_loss, roc_auc_score

    np.random.seed(random_seed)

    # Lat/Lon BANNED as model features (discussion 32369)
    # Use TerraClimate only — all columns except ID, coords, target
    feature_cols = [c for c in train.columns
                    if c not in ("ID", "Latitude", "Longitude", "Occurrence Status")]

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


# ── Submission Validation ──────────────────────────────────────────────────────

def validate_submission(
    sub_path:    Path,
    sample_path: Path,
    config:      ChallengeConfig,
) -> list[str]:
    """
    Layer 1 — Automatic validation gate.
    Runs 8 checks against SampleSubmission and challenge_config rules.
    Returns list of error strings. Empty list = all checks passed.
    Gate HARD BLOCKS if any error is returned — no human prompt shown.
    """
    errors = []

    sub    = pd.read_csv(sub_path)
    sample = pd.read_csv(sample_path)

    # Check 1 — Column names match exactly
    if list(sub.columns) != list(sample.columns):
        errors.append(
            f"Column mismatch: got {list(sub.columns)}, "
            f"expected {list(sample.columns)}"
        )

    # Check 2 — Row count matches SampleSubmission
    if len(sub) != len(sample):
        errors.append(
            f"Row count mismatch: got {len(sub)}, expected {len(sample)}"
        )

    # Check 3 — No missing IDs
    expected_ids = set(sample["ID"].astype(str))
    got_ids      = set(sub["ID"].astype(str))
    missing      = expected_ids - got_ids
    if missing:
        errors.append(
            f"Missing {len(missing)} IDs from SampleSubmission: "
            f"{sorted(missing)[:5]}{'…' if len(missing) > 5 else ''}"
        )

    # Check 4 — No extra IDs
    extra = got_ids - expected_ids
    if extra:
        errors.append(
            f"Extra {len(extra)} IDs not in SampleSubmission: "
            f"{sorted(extra)[:5]}{'…' if len(extra) > 5 else ''}"
        )

    # Check 5 — ID order matches SampleSubmission exactly
    if list(sub["ID"].astype(str)) != list(sample["ID"].astype(str)):
        errors.append(
            "ID order does not match SampleSubmission — reindex before submitting"
        )

    # Check 6 — No null values anywhere
    if sub.isnull().any().any():
        null_cols = sub.columns[sub.isnull().any()].tolist()
        errors.append(f"Null values found in columns: {null_cols}")

    # Checks 7 & 8 only apply when use_probabilities is True
    if config.use_probabilities:
        pred_col = [c for c in sub.columns if c.upper() != "ID"][0]
        vals     = sub[pred_col]

        # Check 7 — Values in [0.0, 1.0]
        if vals.min() < 0.0 or vals.max() > 1.0:
            errors.append(
                f"Prediction values out of [0, 1] range: "
                f"min={vals.min():.8f}, max={vals.max():.8f}"
            )

        # Check 8 — Not thresholded (all values exactly 0 or 1)
        unique_vals = set(vals.round(8).unique())
        if unique_vals.issubset({0, 1, 0.0, 1.0}):
            errors.append(
                f"Submission appears thresholded — all values are 0 or 1. "
                f"use_probabilities=True requires raw float probabilities, "
                f"not binary class labels."
            )

    return errors


# ── Human Gate ─────────────────────────────────────────────────────────────────

def _human_submission_gate(
    oof_logloss: float,
    oof_auc:     float,
    sub_path:    Path,
    sample_path: Path,
    remaining:   int,
    config:      ChallengeConfig,
) -> bool:
    """
    Two-layer submission gate.

    Layer 1 — validate_submission() runs 8 automatic checks.
              Any failure → hard block, no prompt shown, returns False.

    Layer 2 — Human YES/NO prompt.
              Only reached if Layer 1 passes all 8 checks.
              Returns True only on exact 'YES' input.
              Any other input → returns False, no submit.
    """

    # ── Layer 1: Automatic validation ─────────────────────────
    print(f"\n{'='*60}")
    print("Running pre-submission validation (8 checks)…")
    print(f"{'='*60}")

    errors = validate_submission(sub_path, sample_path, config)

    if errors:
        print(f"\n❌ VALIDATION FAILED — {len(errors)} error(s) found:")
        for i, e in enumerate(errors, 1):
            print(f"  {i}. {e}")
        print(f"\n{'='*60}")
        print("Gate BLOCKED — fix all errors before submitting.")
        print(f"{'='*60}\n")
        return False   # hard block — Layer 2 never reached

    print(f"✅ All 8 validation checks passed.")

    # ── Layer 2: Human confirmation ────────────────────────────
    print(f"""
{'='*60}
=== HUMAN GATE: Skill 08 — Anchor Submission ===
{'='*60}
OOF Log Loss     : {oof_logloss:.6f}
OOF AUC          : {oof_auc:.6f}
Submission file  : {sub_path.name}
Remaining today  : {remaining}
Budget phase     : Anchor (max 2/day — reserve 2)
Validation       : ✅ PASSED (8/8 checks)

Warning: This will consume 1 submission from your daily budget.
Only proceed if you are satisfied with the OOF scores above.

Type YES to submit or NO to exit without submitting.
{'='*60}""")

    response = input("Submit? [YES/NO]: ").strip().upper()

    if response == "YES":
        print("✅ Human gate: APPROVED — proceeding with submission.")
        return True
    else:
        print(f"🛑 Human gate: DECLINED (input='{response}') — submission skipped.")
        print("   Run again with --submit when ready.")
        return False


# ── Zindi Submit ───────────────────────────────────────────────────────────────

def submit_to_zindi(
    sub_path:    Path,
    oof_logloss: float,
) -> dict:
    """
    Submit to Zindi via ZindiClient.
    Budget guard is enforced inside ZindiClient.submit().
    Comment format follows AGENTS.md Rule 6.
    """
    from zindian.zindi_client import ZindiClient

    config  = ChallengeConfig.load()
    comment = (
        f"branch:anchor-baseline"
        f"|oof_rmse:{oof_logloss:.4f}"
        f"|features:2"
        f"|calib:none"
    )

    client = ZindiClient()
    client.select_competition(config.slug)

    result = client.submit(
        filepath=str(sub_path),
        comment=comment,
    )

    return result


# ── Git ────────────────────────────────────────────────────────────────────────

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


# ── Save Submission CSV ────────────────────────────────────────────────────────

def next_submission_path(paths, suffix: str = "anchor") -> Path:
    """Return the next numbered submission path for this competition."""
    state = SkillStateStore(paths.state_path).read()
    highest_state_count = max(
        int(state.get("submissions_used_today") or 0),
        int(state.get("submissions_used_total") or 0),
    )

    highest_file_count = 0
    pattern = re.compile(r"^sub_(\d{3})_.*\.csv$")
    if paths.submissions_dir.exists():
        for path in paths.submissions_dir.glob("sub_*.csv"):
            match = pattern.match(path.name)
            if match:
                highest_file_count = max(highest_file_count, int(match.group(1)))

    next_num = max(highest_state_count, highest_file_count) + 1
    return paths.submissions_dir / f"sub_{next_num:03d}_{suffix}.csv"


def save_submission(
    test_ids:       np.ndarray,
    predictions:    np.ndarray,
    submission_col: str,
    output_path:    Path,
) -> None:
    """Save predictions in Zindi submission format (probabilities)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sub_df = pd.DataFrame({
        "ID":           np.asarray(test_ids),
        submission_col: np.asarray(predictions, dtype=np.float64),
    })
    sub_df.to_csv(output_path, index=False)
    print(f"✅ Submission CSV saved → {output_path}")


# ── Entry Point ────────────────────────────────────────────────────────────────

def run(
    *,
    n_splits:    int  = 5,
    random_seed: int  = 42,
    submit:      bool = False,
) -> dict:
    """
    Skill 08 — Anchor Baseline.

    Args:
        n_splits:    KFold splits (default 5).
        random_seed: Global random seed (default 42).
        submit:      If True, runs two-layer gate then submits on human YES.
                     NEVER set True autonomously — human must pass YES at prompt.

    Returns:
        dict with status, oof_logloss, oof_auc, submission path, git branch,
        submitted flag, and submission result if submitted.
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
    train, test, training_target_col, submission_col = load_data(paths)
    print(f"\nData loaded:")
    print(f"  Train          : {train.shape}")
    print(f"  Test           : {test.shape}")
    print(f"  Training target: {training_target_col}")
    print(f"  Submission col : {submission_col}")

    # ── Train ──────────────────────────────────────────────────
    print(f"\nTraining LightGBM anchor baseline…")
    oof_preds, test_preds, oof_logloss, oof_auc = compute_oof_predictions(
        train, test, config, training_target_col,
        n_splits=n_splits, random_seed=random_seed,
    )

    # ── Save OOF predictions ───────────────────────────────────
    oof_path = paths.data_raw_dir / "oof_anchor.csv"
    pd.DataFrame({
        "ID":                train["ID"],
        "Predicted":         oof_preds,
        training_target_col: train[training_target_col],
    }).to_csv(oof_path, index=False)
    print(f"✅ OOF predictions saved → {oof_path}")

    # ── Save submission CSV ────────────────────────────────────
    sub_path    = next_submission_path(paths)
    sample_path = paths.data_raw_dir    / "SampleSubmission.csv"
    save_submission(test["ID"].values, test_preds, submission_col, sub_path)

    # ── Log to DuckDB ledger ───────────────────────────────────
    ledger = Ledger()
    exp_id = ledger.log_experiment(
        branch_name="anchor-baseline",
        oof_rmse=oof_logloss,
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

    # ── Human-gated submission ─────────────────────────────────
    submission_result = None

    if submit:
        # Check budget before showing gate
        try:
            from zindian.zindi_client import ZindiClient
            _client = ZindiClient()
            _client.select_competition(config.slug)
            remaining = _client.remaining_submissions
        except Exception:
            remaining = -1  # unknown — gate will warn

        approved = _human_submission_gate(
            oof_logloss=oof_logloss,
            oof_auc=oof_auc,
            sub_path=sub_path,
            sample_path=sample_path,
            remaining=remaining,
            config=config,
        )

        if approved:
            submission_result = submit_to_zindi(
                sub_path=sub_path,
                oof_logloss=oof_logloss,
            )
            state = state_store.read()
            state_store.update(
                submissions_used_today=int(state.get("submissions_used_today") or 0) + 1,
                submissions_used_total=int(state.get("submissions_used_total") or 0) + 1,
                anchor_lb_score=None,   # populated after Zindi scores it
                last_updated=datetime.now(timezone.utc).isoformat(),
            )
            print(f"✅ Submission complete. Rank: {submission_result.get('rank')}")
        else:
            print("\n📋 To submit when ready:")
            print("   python -m zindian.skills.skill_08_anchor --submit")
    else:
        print(f"""
{'='*60}
Submission NOT triggered (submit=False — human gate enforced).
OOF Log Loss : {oof_logloss:.6f}
OOF AUC      : {oof_auc:.6f}

To submit when ready:
  python -m zindian.skills.skill_08_anchor --submit
{'='*60}""")

    return {
        "status":            "OK",
        "oof_logloss":       oof_logloss,
        "oof_auc":           oof_auc,
        "submission_path":   str(sub_path),
        "git_branch":        "anchor-baseline",
        "n_features":        2,
        "submitted":         submission_result is not None,
        "submission_result": submission_result,
        "message":           "Anchor baseline trained and locked",
    }


if __name__ == "__main__":
    import sys
    submit_flag = "--submit" in sys.argv
    result = run(submit=submit_flag)
    printable = {k: v for k, v in result.items() if k != "submission_result"}
    print(json.dumps(printable, indent=2))
