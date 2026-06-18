"""
Skill 09 — Probability Calibration

Simple, config-driven probability calibration (Platt / isotonic / none).
Reads OOF and test probability artifacts, fits a calibrator on OOF and applies
to test probs, writing calibrated test files and updating SKILL_STATE.json.

Usage: python -m zindian.skills.skill_09_calibration --method isotonic
"""

from __future__ import annotations
import tabula.skill_state_autopatch  # noqa

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from zindian.cv import get_cv_splits
from zindian.config import ChallengeConfig
from zindian.paths import resolve_competition_paths
from zindian.state import (
    SkillStateStore,
    resolve_active_cv_strategy_id,
    write_oof_record,
)


def _fit_platt(oof_probs: np.ndarray, y: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(solver="lbfgs", max_iter=500)
    model.fit(np.asarray(oof_probs).reshape(-1, 1), y)
    return model


def _fit_isotonic(oof_probs: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(np.asarray(oof_probs), y)
    return iso


def _resolve_target_col(config: ChallengeConfig) -> str:
    target_col = config.get("target_col") or config.get("target_column")
    if not target_col:
        raise RuntimeError("target_col not initialized in challenge_config.json")
    return str(target_col)


def _resolve_cv_strategy(
    config: ChallengeConfig, state: dict[str, Any]
) -> dict[str, Any]:
    raw_config = getattr(config, "_data", {}) or {}
    strategy = dict(
        raw_config.get("cv_strategy") or config.get("cv_strategy", {}) or {}
    )
    override = state.get("cv_strategy_override", {}) or {}
    if bool(override.get("active", False)):
        override_strategy = override.get("override_strategy")
        if isinstance(override_strategy, dict):
            strategy.update(override_strategy)
        elif override_strategy is not None:
            strategy["type"] = override_strategy
    return strategy


def _resolve_candidate_branch(state: dict[str, Any], retraining_active: bool) -> str:
    for key in ("best_variant_this_round", "best_variant_branch", "anchor_git_branch"):
        value = state.get(key)
        if isinstance(value, str) and value:
            return value

    branch_names: list[str] = []
    for key, value in state.items():
        if not (
            isinstance(key, str) and key.startswith("branch_") and key.endswith("_oof")
        ):
            continue
        if not isinstance(value, dict):
            continue
        branch_name = str(
            value.get("branch_name") or key.removeprefix("branch_").removesuffix("_oof")
        )
        if retraining_active and not branch_name.endswith("_augmented"):
            continue
        if not retraining_active and branch_name.endswith("_augmented"):
            continue
        branch_names.append(branch_name)

    branch_names = list(dict.fromkeys(branch_names))
    if len(branch_names) == 1:
        return branch_names[0]

    raise RuntimeError("No promoted branch candidate found in SKILL_STATE.json")


def _resolve_oof_record(
    state: dict[str, Any], branch_name: str, retraining_active: bool
) -> dict[str, Any]:
    candidates = []
    if retraining_active and not branch_name.endswith("_augmented"):
        candidates.append(f"branch_{branch_name}_augmented_oof")
        candidates.append(f"branch_{branch_name}_oof_augmented")
    candidates.append(f"branch_{branch_name}_oof")
    if branch_name.endswith("_augmented"):
        candidates.append(
            f"branch_{branch_name.removesuffix('_augmented')}_oof_augmented"
        )

    for key in candidates:
        record = state.get(key)
        if isinstance(record, dict) and record.get("scores") is not None:
            return record

    raise RuntimeError(f"No OOF record found in SKILL_STATE for branch '{branch_name}'")


def _candidate_test_names(branch_name: str, retraining_active: bool) -> list[str]:
    names = []
    if retraining_active and not branch_name.endswith("_augmented"):
        names.append(f"test_probs_{branch_name}_augmented")
    names.append(f"test_probs_{branch_name}")
    if branch_name.endswith("_augmented"):
        names.append(f"test_probs_{branch_name.removesuffix('_augmented')}_augmented")
    return list(dict.fromkeys(names))


def _resolve_test_prob_path(
    proc_dir: Path, reports_dir: Path, branch_name: str, retraining_active: bool
) -> Path:
    for test_name in _candidate_test_names(branch_name, retraining_active):
        for base_dir in (proc_dir, reports_dir):
            candidate = base_dir / f"{test_name}.csv"
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        f"No test probability file found for branch '{branch_name}'"
    )


def _get_groups(train: pd.DataFrame, config: ChallengeConfig) -> np.ndarray | None:
    cv_strategy = config.get("cv_strategy", {}) or {}
    group_col = cv_strategy.get("group_column") or (
        config.get("spatial_signal", {}) or {}
    ).get("group_col")
    if group_col and group_col in train.columns:
        return np.asarray(train[group_col].values)
    return None


def _fit_calibrator_foldwise(
    method: str,
    oof_probs: np.ndarray,
    y: np.ndarray,
    cv_strategy: dict[str, Any],
    groups: np.ndarray | None,
) -> tuple[np.ndarray, Any]:
    calibrated_oof = np.asarray(oof_probs, dtype=np.float64).copy()
    splitter = get_cv_splits(
        np.zeros((len(y), 1)), np.asarray(y), groups=groups, cv_strategy=cv_strategy
    )

    for train_idx, val_idx in splitter:
        x_train = np.asarray(oof_probs)[train_idx]
        y_train = np.asarray(y)[train_idx]
        x_val = np.asarray(oof_probs)[val_idx]

        if np.unique(y_train).size < 2:
            calibrated_oof[val_idx] = x_val
            continue

        if method == "platt":
            calibrator = _fit_platt(x_train, y_train)
            calibrated_oof[val_idx] = calibrator.predict_proba(x_val.reshape(-1, 1))[
                :, 1
            ]
        elif method == "isotonic":
            calibrator = _fit_isotonic(x_train, y_train)
            calibrated_oof[val_idx] = calibrator.transform(x_val)
        else:
            calibrated_oof[val_idx] = x_val

    if np.unique(y).size < 2:
        return calibrated_oof, None

    if method == "platt":
        return calibrated_oof, _fit_platt(oof_probs, y)
    if method == "isotonic":
        return calibrated_oof, _fit_isotonic(oof_probs, y)
    return calibrated_oof, None


def run(method: str = "none", dry_run: bool = False) -> Dict[str, object]:
    print("\n" + "=" * 60)
    print("SKILL 09 — Probability Calibration")
    print("=" * 60 + "\n")

    paths = resolve_competition_paths(require_competition=True)
    config = ChallengeConfig.load()
    store = SkillStateStore(paths.state_path)
    state = store.read()

    raw_config = getattr(config, "_data", {}) or {}
    task_type = str(
        raw_config.get("task_type", config.get("task_type", "classification"))
    )
    
    # Skip for regression or multi-target competitions
    if task_type == "regression" or task_type == "multi_target":
        return {
            "status": "SKIPPED",
            "reason": "Probability calibration applies to single-target classification tasks only",
        }
    
    # Check if multi-target via target_config
    target_config = config.get("target_config")
    if target_config and len(target_config.get("targets", [])) > 1:
        return {
            "status": "SKIPPED",
            "reason": "Probability calibration not supported for multi-target competitions",
        }

    proc_dir = paths.data_processed_dir
    reports_dir = paths.reports_dir
    train = pd.read_csv(proc_dir / "features_train.csv")

    target = _resolve_target_col(config)
    if target not in train.columns:
        raise RuntimeError(f"target column '{target}' not present in training data")
    y = np.asarray(train[target].values, dtype=int)

    retraining_active = bool(
        state.get("pseudo_label_result", {}).get("retraining_required", False)
    )
    candidate_branch = _resolve_candidate_branch(
        state, retraining_active=retraining_active
    )
    oof_record = _resolve_oof_record(
        state, candidate_branch, retraining_active=retraining_active
    )
    oof_probs = np.asarray(oof_record["scores"], dtype=np.float64)

    if len(oof_probs) != len(y):
        raise RuntimeError(
            f"OOF length mismatch for branch '{candidate_branch}': oof={len(oof_probs)} train={len(y)}"
        )

    cv_strategy = _resolve_cv_strategy(config, state)
    groups = _get_groups(train, config)

    if method == "none":
        print(
            "No calibration requested — copying original test probs to calibrated files"
        )
        test_path = _resolve_test_prob_path(
            proc_dir, reports_dir, candidate_branch, retraining_active
        )
        mapping = {test_path.name: str(proc_dir / f"calib_{test_path.name}")}
    else:
        if method not in ("platt", "isotonic"):
            raise ValueError(f"Unknown method: {method}")

        calibrated_oof, global_calibrator = _fit_calibrator_foldwise(
            method,
            oof_probs,
            y,
            cv_strategy=cv_strategy,
            groups=groups,
        )

        test_path = _resolve_test_prob_path(
            proc_dir, reports_dir, candidate_branch, retraining_active
        )
        df = pd.read_csv(test_path)
        try:
            cfg = ChallengeConfig.load()
            id_col = cfg.get("id_col") or cfg.get("id_column") or "ID"
        except Exception:
            id_col = "ID"
        pcol = [c for c in df.columns if c != id_col][0]
        probs = np.asarray(df[pcol].values, dtype=np.float64)

        if global_calibrator is not None:
            if method == "platt":
                calibrated = global_calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]
            else:
                calibrated = global_calibrator.transform(probs)
        else:
            calibrated = probs

        out = proc_dir / f"calib_{test_path.name}"
        mapping = {test_path.name: str(out)}
        if not dry_run:
            pd.DataFrame({id_col: df[id_col], pcol: calibrated}).to_csv(
                out, index=False
            )

    if not dry_run:
        state_patch = {
            "calibration_method": method,
            "calibration_written_at": datetime.now(timezone.utc).isoformat(),
            "calibration_candidate_branch": candidate_branch,
            "calibration_candidate_oof_key": (
                f"branch_{candidate_branch}_oof"
                if not retraining_active or candidate_branch.endswith("_augmented")
                else f"branch_{candidate_branch}_oof_augmented"
            ),
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

        # Persist a canonical OOF record for the calibration step (SoT-shaped)
        try:
            if method != "none":
                try:
                    cv_id = resolve_active_cv_strategy_id(store.read(), cfg_data)
                except Exception:
                    cv_id = "unknown"
                seed = int(cfg_data.get("reproducibility", {}).get("seed", 42))
                try:
                    write_oof_record(
                        store,
                        branch_name=f"calibration_{candidate_branch}",
                        scores=np.asarray(calibrated_oof, dtype=np.float64).tolist(),
                        cv_strategy_id=cv_id,
                        seed=seed,
                        model_config={
                            "method": method,
                            "source_branch": candidate_branch,
                            "cv_strategy": cv_strategy,
                        },
                    )
                except Exception:
                    # Do not fail the skill if OOF writing failed
                    pass
        except Exception:
            pass

    return {"status": "OK", "method": method, "mapping": mapping}


if __name__ == "__main__":
    import argparse
    import json

    p = argparse.ArgumentParser()
    p.add_argument("--method", default="none")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    print(json.dumps(run(method=args.method, dry_run=args.dry_run), indent=2))
