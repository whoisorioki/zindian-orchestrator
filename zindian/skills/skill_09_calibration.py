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
from sklearn.preprocessing import LabelEncoder
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

    # Check if multi-target via target_config
    target_config = config.get("target_config")
    if target_config and len(target_config.get("targets", [])) > 1:
        return _run_multi_target(
            method, dry_run, paths, config, store, state, target_config
        )

    # Skip for regression
    if task_type == "regression":
        return {
            "status": "SKIPPED",
            "reason": "Probability calibration applies to classification tasks only",
        }

    proc_dir = paths.data_processed_dir
    reports_dir = paths.reports_dir
    train = pd.read_csv(proc_dir / "features_train.csv")

    target = _resolve_target_col(config)
    if target not in train.columns:
        raise RuntimeError(f"target column '{target}' not present in training data")
    y_raw = train[target]
    if y_raw.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
    else:
        y = np.asarray(y_raw.values, dtype=int)

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


def _run_multi_target(
    method: str,
    dry_run: bool,
    paths: Any,
    config: ChallengeConfig,
    store: SkillStateStore,
    state: dict[str, Any],
    target_config: dict[str, Any],
) -> Dict[str, object]:
    print("Multi-target mode detected")

    targets = target_config.get("targets", [])
    proc_dir = paths.data_processed_dir
    reports_dir = paths.reports_dir
    train_features = pd.read_csv(proc_dir / "features_train.csv")
    train_raw = pd.read_csv(paths.data_raw_dir / "Train.csv")

    retraining_active = bool(
        state.get("pseudo_label_result", {}).get("retraining_required", False)
    )

    # For multi-target, find branch from actual OOF keys
    candidate_branch = None
    for target_spec in targets:
        target_name = target_spec.get("name")
        # Look for OOF keys for this target
        for key in state.keys():
            if (
                key.startswith("branch_")
                and f"_{target_name}_" in key
                and key.endswith("_oof")
            ):
                # Extract branch name
                parts = key.removeprefix("branch_").removesuffix("_oof")
                if retraining_active and parts.endswith("_augmented"):
                    candidate_branch = parts.removesuffix(f"_{target_name}_augmented")
                else:
                    candidate_branch = parts.removesuffix(f"_{target_name}")
                break
        if candidate_branch:
            break

    if not candidate_branch:
        candidate_branch = _resolve_candidate_branch(
            state, retraining_active=retraining_active
        )

    print(f"Using branch: {candidate_branch}")

    cv_strategy = _resolve_cv_strategy(config, state)
    groups = _get_groups(train_features, config)

    skipped_targets = []
    mapping = {}
    _calibrated_any = False

    for target_spec in targets:
        target_name = target_spec.get("name")
        target_task_type = target_spec.get("task_type")

        if target_task_type != "classification":
            print(f"Skipping {target_name} (task_type={target_task_type})")
            skipped_targets.append(target_name)
            continue

        print(f"\nCalibrating classification target: {target_name}")

        if target_name not in train_raw.columns:
            print(f"Warning: {target_name} not in training data, skipping")
            skipped_targets.append(target_name)
            continue

        # Load OOF file with probabilities
        oof_file = paths.data_raw_dir / "oof_anchor_multi.csv"
        if not oof_file.exists():
            print("Warning: OOF file not found, skipping actual calibration")
            skipped_targets.append(target_name)
            continue

        oof_df = pd.read_csv(oof_file)

        # Check for probability columns in OOF
        prob_cols = [
            c for c in oof_df.columns if c.startswith(f"{target_name}_prob_class_")
        ]
        if not prob_cols:
            print(f"Warning: No probability columns for {target_name} in OOF, skipping")
            skipped_targets.append(target_name)
            continue

        print(f"Found {len(prob_cols)} probability classes for {target_name}")

        # For multi-class, calibrate each class probability separately
        y_raw = train_raw[target_name]
        if y_raw.dtype.kind in ("U", "S", "O"):
            le = LabelEncoder()
            y = le.fit_transform(y_raw.astype(str)).astype(int)
        else:
            y = np.asarray(y_raw.values, dtype=int)

        oof_key = f"branch_{candidate_branch}_{target_name}_oof"
        if retraining_active and not candidate_branch.endswith("_augmented"):
            oof_key = f"branch_{candidate_branch}_{target_name}_augmented_oof"

        oof_record = state.get(oof_key)
        if not isinstance(oof_record, dict):
            print(f"Warning: No OOF record at {oof_key}, skipping")
            skipped_targets.append(target_name)
            continue

        # Extract probability matrix from OOF file
        oof_probs_matrix = oof_df[prob_cols].values

        if len(oof_probs_matrix) != len(y):
            print(f"Warning: OOF length mismatch for {target_name}, skipping")
            skipped_targets.append(target_name)
            continue

        # Find test probability files
        test_names = _candidate_test_names(candidate_branch, retraining_active)
        test_names_target = [
            n.replace(candidate_branch, f"{candidate_branch}_{target_name}")
            for n in test_names
        ]
        test_names_target.extend([f"test_probs_{candidate_branch}_{target_name}"])

        test_path = None
        for test_name in test_names_target:
            for base_dir in (proc_dir, reports_dir):
                candidate = base_dir / f"{test_name}.csv"
                if candidate.exists():
                    test_path = candidate
                    break
            if test_path:
                break

        if not test_path:
            print(f"Note: No test probs for {target_name}, calibrating OOF only")
            # Still calibrate OOF for state tracking
            if method != "none":
                calibrated_oof_matrix = np.zeros_like(oof_probs_matrix)
                for class_idx in range(len(prob_cols)):
                    y_binary = (y == class_idx).astype(int)
                    oof_probs_class = oof_probs_matrix[:, class_idx]
                    calibrated_oof_class, _ = _fit_calibrator_foldwise(
                        method,
                        oof_probs_class,
                        y_binary,
                        cv_strategy=cv_strategy,
                        groups=groups,
                    )
                    calibrated_oof_matrix[:, class_idx] = calibrated_oof_class
                row_sums = calibrated_oof_matrix.sum(axis=1, keepdims=True)
                row_sums = np.where(row_sums == 0, 1, row_sums)
                calibrated_oof_matrix = calibrated_oof_matrix / row_sums
                _calibrated_any = True
                if not dry_run:
                    try:
                        cfg_data = getattr(config, "_data", {}) or {}
                        cv_id = resolve_active_cv_strategy_id(state, cfg_data)
                        seed = int(cfg_data.get("reproducibility", {}).get("seed", 42))
                        calibrated_oof_pred = np.argmax(calibrated_oof_matrix, axis=1)
                        write_oof_record(
                            store,
                            branch_name=f"calibration_{candidate_branch}_{target_name}",
                            scores=calibrated_oof_pred.tolist(),
                            cv_strategy_id=cv_id,
                            seed=seed,
                            model_config={
                                "method": method,
                                "source_branch": candidate_branch,
                                "target": target_name,
                            },
                        )
                    except Exception:
                        pass
            continue

        if method == "none":
            out = proc_dir / f"calib_{test_path.name}"
            mapping[test_path.name] = str(out)
            continue

        # Calibrate each class probability separately
        calibrated_oof_matrix = np.zeros_like(oof_probs_matrix)

        for class_idx in range(len(prob_cols)):
            y_binary = (y == class_idx).astype(int)
            oof_probs_class = oof_probs_matrix[:, class_idx]

            calibrated_oof_class, global_calibrator = _fit_calibrator_foldwise(
                method,
                oof_probs_class,
                y_binary,
                cv_strategy=cv_strategy,
                groups=groups,
            )
            calibrated_oof_matrix[:, class_idx] = calibrated_oof_class

        # Normalize probabilities to sum to 1
        row_sums = calibrated_oof_matrix.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        calibrated_oof_matrix = calibrated_oof_matrix / row_sums

        # Load and calibrate test probabilities
        df_test = pd.read_csv(test_path)
        id_col = config.get("id_col") or config.get("id_column") or "ID"

        # Check if test file has probability columns
        test_prob_cols = [
            c
            for c in df_test.columns
            if c.startswith(f"{target_name}_prob_class_") or c.startswith("prob_class_")
        ]
        if not test_prob_cols:
            print(
                f"Warning: No probability columns in test file for {target_name}, skipping"
            )
            skipped_targets.append(target_name)
            continue

        test_probs_matrix = df_test[test_prob_cols].values
        calibrated_test_matrix = np.zeros_like(test_probs_matrix)

        for class_idx in range(len(prob_cols)):
            y_binary = (y == class_idx).astype(int)
            oof_probs_class = oof_probs_matrix[:, class_idx]

            if np.unique(y_binary).size < 2:
                calibrated_test_matrix[:, class_idx] = test_probs_matrix[:, class_idx]
                continue

            if method == "platt":
                calibrator = _fit_platt(oof_probs_class, y_binary)
                calibrated_test_matrix[:, class_idx] = calibrator.predict_proba(
                    test_probs_matrix[:, class_idx].reshape(-1, 1)
                )[:, 1]
            elif method == "isotonic":
                calibrator = _fit_isotonic(oof_probs_class, y_binary)
                calibrated_test_matrix[:, class_idx] = calibrator.transform(
                    test_probs_matrix[:, class_idx]
                )
            else:
                calibrated_test_matrix[:, class_idx] = test_probs_matrix[:, class_idx]

        # Normalize test probabilities
        row_sums = calibrated_test_matrix.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        calibrated_test_matrix = calibrated_test_matrix / row_sums

        # Save calibrated test probabilities
        out = proc_dir / f"calib_{test_path.name}"
        mapping[test_path.name] = str(out)
        if not dry_run:
            out_df = pd.DataFrame({id_col: df_test[id_col]})
            for i, col in enumerate(test_prob_cols):
                out_df[col] = calibrated_test_matrix[:, i]
            out_df.to_csv(out, index=False)

            try:
                cfg_data = getattr(config, "_data", {}) or {}
                cv_id = resolve_active_cv_strategy_id(state, cfg_data)
                seed = int(cfg_data.get("reproducibility", {}).get("seed", 42))
                # Store calibrated OOF as argmax for compatibility
                calibrated_oof_pred = np.argmax(calibrated_oof_matrix, axis=1)
                write_oof_record(
                    store,
                    branch_name=f"calibration_{candidate_branch}_{target_name}",
                    scores=calibrated_oof_pred.tolist(),
                    cv_strategy_id=cv_id,
                    seed=seed,
                    model_config={
                        "method": method,
                        "source_branch": candidate_branch,
                        "target": target_name,
                    },
                )
            except Exception:
                pass

    if not dry_run:
        state_patch = {
            "calibration_method": method,
            "calibration_written_at": datetime.now(timezone.utc).isoformat(),
            "calibration_candidate_branch": candidate_branch,
            "calibration_skipped_targets": skipped_targets,
        }
        try:
            cfg_data = getattr(config, "_data", {}) or {}
            cv_id = resolve_active_cv_strategy_id(state, cfg_data)
            state_patch["calibration_oof_cv_strategy_id"] = cv_id
        except Exception:
            pass
        store.update(**state_patch)

    return {
        "status": "OK",
        "method": method,
        "mapping": mapping,
        "skipped_targets": skipped_targets,
    }


if __name__ == "__main__":
    import argparse
    import json

    p = argparse.ArgumentParser()
    p.add_argument("--method", default="none")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    print(json.dumps(run(method=args.method, dry_run=args.dry_run), indent=2))
