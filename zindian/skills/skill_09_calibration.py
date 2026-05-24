"""
Skill 09 — Probability Calibration

Simple, config-driven probability calibration (Platt / isotonic / none).
Reads OOF and test probability artifacts, fits a calibrator on OOF and applies
to test probs, writing calibrated test files and updating SKILL_STATE.json.

Usage: python -m zindian.skills.skill_09_calibration --method isotonic
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from zindian.paths import resolve_competition_paths
from zindian.state import SkillStateStore, resolve_active_cv_strategy_id
from zindian.config import ChallengeConfig


def _fit_platt(oof_probs: np.ndarray, y: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(solver="lbfgs", max_iter=500)
    model.fit(np.asarray(oof_probs).reshape(-1, 1), y)
    return model


def _fit_isotonic(oof_probs: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(np.asarray(oof_probs), y)
    return iso


def run(method: str = "none", dry_run: bool = False) -> Dict[str, object]:
    print("\n" + "=" * 60)
    print("SKILL 09 — Probability Calibration")
    print("=" * 60 + "\n")

    paths = resolve_competition_paths(require_competition=True)
    config = ChallengeConfig.load()
    store = SkillStateStore(paths.state_path)
    state = store.read()

    proc_dir = paths.data_processed_dir
    reports_dir = paths.reports_dir

    # Find OOF aggregate file
    oof_files = sorted(proc_dir.glob("oof_variant-*.csv"))
    if not oof_files:
        print("No OOF files found for calibration")
        return {"status": "NO_OOF"}

    # Use best OOF found by skill_12 metric scan (if present)
    best_oof = oof_files[0]
    oof_df = pd.read_csv(best_oof)
    prob_col = [c for c in oof_df.columns if c != "ID"][0]
    oof_probs = np.asarray(oof_df[prob_col].values, dtype=float)

    # Load train labels to calibrate against
    train = pd.read_csv(proc_dir / "features_train.csv")
    target = config.get("target_column", "Occurrence Status")
    y = np.asarray(train[target].values, dtype=int)

    if method == "none":
        print("No calibration requested — copying original test probs to calibrated files")
        mapping = {f: (proc_dir / f.name) for f in proc_dir.glob("test_probs_*.csv")}
    else:
        calibrator_platt = None
        calibrator_iso = None
        if method == "platt":
            calibrator_platt = _fit_platt(oof_probs, y)
        elif method == "isotonic":
            calibrator_iso = _fit_isotonic(oof_probs, y)
        else:
            raise ValueError(f"Unknown method: {method}")

        mapping = {}
        for test_path in proc_dir.glob("test_probs_*.csv"):
            df = pd.read_csv(test_path)
            pcol = [c for c in df.columns if c != "ID"][0]
            probs = np.asarray(df[pcol].values, dtype=float)
            if calibrator_platt is not None:
                calibrated = calibrator_platt.predict_proba(probs.reshape(-1, 1))[:, 1]
            elif calibrator_iso is not None:
                calibrated = calibrator_iso.transform(probs)
            else:
                calibrated = probs
            out = proc_dir / f"calib_{test_path.name}"
            if not dry_run:
                pd.DataFrame({"ID": df["ID"], pcol: calibrated}).to_csv(out, index=False)
            mapping[test_path.name] = str(out)

    if not dry_run:
        state_patch = {
            "calibration_method": method,
            "calibration_written_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            cfg_data = ChallengeConfig.load()._data
        except Exception:
            cfg_data = {}
        try:
            cv_id = resolve_active_cv_strategy_id(store.read(), cfg_data)
            state_patch["last_oof_cv_strategy_id"] = cv_id
            state_patch["calibration_oof_cv_strategy_id"] = cv_id
        except Exception:
            pass
        store.update(**state_patch)

    return {"status": "OK", "method": method, "mapping": mapping}


if __name__ == "__main__":
    import argparse, json
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="none")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    print(json.dumps(run(method=args.method, dry_run=args.dry_run), indent=2))
