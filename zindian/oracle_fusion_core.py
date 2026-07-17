"""
Skill 13 — Oracle Fusion

Refactored to SoT contracts:
- Read-only human gate consumption (no skill writes gate approvals)
- Candidate intake from SKILL_STATE verified branch records
- Correlation pruning (Pearson for classification, Spearman for regression)
- Metric-adaptive optimization and probability-format aware outputs
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from zindian.config import ChallengeConfig
from zindian.paths import resolve_competition_paths
from zindian.state import SkillStateStore

TOP_N = 3  # Number of variants to blend


def _resolve_target_col(config: ChallengeConfig, train: pd.DataFrame) -> str:
    target_col = config.get("target_col") or config.get("target_column")
    if target_col and target_col in train.columns:
        return str(target_col)

    # Backward-compatible fallback for older fixtures: infer from known train columns.
    for col in ("target", "label", "y", "occ"):
        if col in train.columns:
            return col

    raise RuntimeError("target_col not initialized in challenge_config.json")


def _classification_threshold_and_f1(
    y_true: np.ndarray, probs: np.ndarray
) -> tuple[float, float]:
    best_f1 = 0.0
    best_t = 0.5
    for t in np.arange(0.3, 0.7, 0.01):
        cur = float(f1_score(y_true, (probs >= float(t)).astype(int)))
        if cur > best_f1:
            best_f1 = cur
            best_t = float(t)
    return best_f1, best_t


def _score_predictions(
    y_true: np.ndarray,
    preds: np.ndarray,
    *,
    task_type: str,
    metric_name: str,
    use_probabilities: bool,
) -> tuple[float, float | None]:
    metric = metric_name.lower()

    if task_type == "classification":
        if metric in ("f1", "f1_score"):
            f1, thr = _classification_threshold_and_f1(y_true, preds)
            return float(f1), float(thr)
        if metric in ("auc", "roc_auc", "roc_auc_score"):
            return float(roc_auc_score(y_true, preds)), None

        # Default classification fallback.
        if use_probabilities:
            return float(roc_auc_score(y_true, preds)), None
        f1, thr = _classification_threshold_and_f1(y_true, preds)
        return float(f1), float(thr)

    # Regression metric handling.
    if metric in ("rmse",):
        return float(np.sqrt(mean_squared_error(y_true, preds))), None
    if metric in ("mse",):
        return float(mean_squared_error(y_true, preds)), None
    if metric in ("mae",):
        return float(mean_absolute_error(y_true, preds)), None
    if metric in ("r2", "r2_score"):
        return float(r2_score(y_true, preds)), None

    # Safe regression default.
    return float(np.sqrt(mean_squared_error(y_true, preds))), None


def _is_better(score_a: float, score_b: float, direction: str) -> bool:
    if direction == "minimize":
        return score_a < score_b
    return score_a > score_b


def _collect_verified_candidates(
    state: dict[str, Any],
    y_true: np.ndarray,
    *,
    task_type: str,
    metric_name: str,
    direction: str,
    use_probabilities: bool,
    retraining_active: bool,
) -> list[dict[str, Any]]:
    """
    Build candidate pool from SKILL_STATE branch records only.

    Selection rules:
    - Key pattern: branch_{name}_oof
    - Include only branches with human_gate_2_{name}_approved == True
    - If retraining is active, include only names ending with _augmented
    """
    candidates: list[dict[str, Any]] = []
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
        is_augmented = branch_name.endswith("_augmented")
        if retraining_active and not is_augmented:
            continue
        if not retraining_active and is_augmented:
            continue

        gate2_key = f"human_gate_2_{branch_name}_approved"
        if not bool(state.get(gate2_key, False)):
            continue

        scores_obj = value.get("scores", [])
        scores_arr = np.asarray(scores_obj, dtype=np.float64)
        if len(scores_arr) != len(y_true):
            continue

        oof_score, threshold = _score_predictions(
            y_true,
            scores_arr,
            task_type=task_type,
            metric_name=metric_name,
            use_probabilities=use_probabilities,
        )
        candidates.append(
            {
                "name": branch_name,
                "probs": scores_arr,
                "score": float(oof_score),
                "threshold": threshold,
                "cv_strategy_id": value.get("cv_strategy_id"),
                "model_config": value.get("model_config", {}),
            }
        )

    return sorted(
        candidates, key=lambda x: x["score"], reverse=(direction != "minimize")
    )


def _correlation(x: np.ndarray, y: np.ndarray, task_type: str) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if len(x_arr) < 2 or len(y_arr) < 2:
        return 0.0

    if task_type == "classification":
        corr = np.corrcoef(x_arr, y_arr)[0, 1]
        return 0.0 if np.isnan(corr) else float(corr)

    # Spearman rank correlation for regression.
    x_rank = pd.Series(x_arr).rank(method="average").to_numpy(dtype=np.float64)
    y_rank = pd.Series(y_arr).rank(method="average").to_numpy(dtype=np.float64)
    corr = np.corrcoef(x_rank, y_rank)[0, 1]
    return 0.0 if np.isnan(corr) else float(corr)


def _prune_collinear(
    candidates: list[dict[str, Any]], *, task_type: str, direction: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Drop lower-scoring candidate for any pair with correlation > 0.95."""
    working = list(candidates)
    dropped: list[dict[str, Any]] = []

    changed = True
    while changed and len(working) > 1:
        changed = False
        for i in range(len(working)):
            for j in range(i + 1, len(working)):
                a = working[i]
                b = working[j]
                corr = _correlation(
                    np.asarray(a["probs"], dtype=np.float64),
                    np.asarray(b["probs"], dtype=np.float64),
                    task_type,
                )
                if corr <= 0.95:
                    continue

                if _is_better(float(a["score"]), float(b["score"]), direction):
                    keep_idx, drop_idx = i, j
                else:
                    keep_idx, drop_idx = j, i

                dropped.append(
                    {
                        "dropped": working[drop_idx]["name"],
                        "kept": working[keep_idx]["name"],
                        "correlation": float(corr),
                    }
                )
                del working[drop_idx]
                changed = True
                break
            if changed:
                break

    return working, dropped


def _candidate_test_names(oof_name: str) -> list[str]:
    # Branch records in SKILL_STATE carry branch_name (e.g., "variant-1").
    # Legacy names are included for backward compatibility.
    names = [f"test_probs_{oof_name}"]
    if oof_name.startswith("oof_probs_"):
        names.append(oof_name.replace("oof_probs_", "test_probs_", 1))
    if oof_name.startswith("oof_"):
        names.append(oof_name.replace("oof_", "test_probs_", 1))
    return names


def _check_test_files(
    proc_dir: Path, reports_dir: Path, names: list[str]
) -> dict[str, Path] | None:
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
        print(f"\n[FAIL] Missing test prob files: {missing}")
        print("   Rerun those variants with --force-save before blending.")
        return None

    return mapping


def run(
    config: dict[str, Any] | ChallengeConfig | None = None,
    state: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> dict:
    # Handle legacy call where dry_run is passed as first positional arg
    if isinstance(config, bool):
        dry_run = config
        config = None

    print("\n" + "=" * 60)
    print("SKILL 13 — Oracle Fusion")
    print("=" * 60 + "\n")

    paths = resolve_competition_paths(require_competition=False)
    in_memory = state is not None

    if not in_memory:
        config_obj = ChallengeConfig.load()
        if paths.state_path is None:
            raise FileNotFoundError("State path could not be resolved")
        store = SkillStateStore(paths.state_path)
        state_obj = store.read()
    else:
        if isinstance(config, ChallengeConfig):
            config_obj = config
        else:
            config_obj = ChallengeConfig(Path("challenge_config.json"), config or {})
        state_obj = state if state is not None else {}

    if paths.competition_dir is None and not in_memory:
        raise FileNotFoundError("Competition directory could not be resolved")

    # For in-memory testing, use local path or default to Path(".")
    comp_dir = paths.competition_dir if paths.competition_dir else Path(".")
    proc_dir = comp_dir / "data" / "processed"
    reports_dir = paths.reports_dir if paths.reports_dir else Path("reports")
    subs_dir = comp_dir / "submissions"
    raw_dir = paths.data_raw_dir if paths.data_raw_dir else comp_dir / "data" / "raw"

    # Multi-target detection (A11)
    target_config = config_obj.get("target_config")
    if target_config and target_config.get("targets"):
        return _run_multi_target_fusion(
            config_obj,
            state_obj,
            paths,
            proc_dir,
            reports_dir,
            subs_dir,
            raw_dir,
            in_memory,
            dry_run,
        )

    # Phase 1: Dynamic intake & strategy check (single-target path)
    branch_name = (
        state_obj.get("anchor_git_branch") or "anchor-baseline"
        if state_obj
        else "anchor-baseline"
    )
    train_path = proc_dir / f"features_train_{branch_name}.csv"
    if not train_path.exists():
        print(f"Train file not found: {train_path}")
        return {"status": "FAILED", "reason": "Train file missing"}

    train = pd.read_csv(train_path)
    result = _run_single_target_fusion(
        config_obj,
        state_obj,
        paths,
        proc_dir,
        reports_dir,
        subs_dir,
        raw_dir,
        train,
        in_memory,
        dry_run,
        None,
    )
    if not dry_run and result.get("status") == "OK":
        sample_path = raw_dir / "SampleSubmission.csv"
        if sample_path.exists():
            sample = pd.read_csv(sample_path)
            id_col = str(sample.columns[0])
            
            input_files = config_obj.get("input_files", {}) or {}
            train_file = input_files.get("train", "Train.csv")
            raw_train = pd.read_csv(raw_dir / train_file)

            store = SkillStateStore(paths.state_path)
            from zindian.state import write_oof_record, resolve_active_cv_strategy_id

            blend_probs = result.get("blend_probs")
            submission_values = result.get("submission_column")

            # Save OOF probs
            if blend_probs is not None:
                oof_df = pd.DataFrame({id_col: raw_train[id_col], "oof_prob": blend_probs})
                oof_path = proc_dir / f"oof_ensemble.csv"
                oof_df.to_csv(oof_path, index=False)
                print(f"  [OK] Ensemble OOF probabilities saved -> {oof_path}")

            # Save Test probs
            if submission_values is not None:
                test_prob_df = pd.DataFrame({id_col: sample[id_col], "test_prob": submission_values})
                test_prob_path = proc_dir / f"test_probs_ensemble.csv"
                test_prob_df.to_csv(test_prob_path, index=False)
                print(f"  [OK] Ensemble Test probabilities saved -> {test_prob_path}")

            # Write OOF record to state
            if blend_probs is not None:
                write_oof_record(
                    store,
                    branch_name="ensemble",
                    scores=blend_probs.tolist(),
                    cv_strategy_id=resolve_active_cv_strategy_id(state_obj, config_obj._data),
                    seed=42,
                    model_config={"target_name": "target", "variant": "ensemble"},
                )

            # Update anchor_git_branch to ensemble
            if not in_memory:
                store.update(
                    anchor_git_branch="ensemble",
                    last_ensemble_path=str(test_prob_path),
                    last_ensemble_oof_metric=result.get("blend_oof_metric"),
                    last_ensemble_metric_name=result.get("metric"),
                    last_ensemble_threshold=result.get("threshold"),
                    last_ensemble_variants=result.get("variants"),
                    last_ensemble_dropped_collinear=result.get("dropped_collinear"),
                    last_updated=datetime.now(timezone.utc).isoformat(),
                )
            else:
                state_obj["anchor_git_branch"] = "ensemble"

    # Remove ndarrays from result to prevent JSON serialization crash
    result.pop("submission_column", None)
    result.pop("blend_probs", None)
    return result


def _run_multi_target_fusion(
    config_obj: ChallengeConfig,
    state_obj: dict[str, Any],
    paths: Any,
    proc_dir: Path,
    reports_dir: Path,
    subs_dir: Path,
    raw_dir: Path,
    in_memory: bool,
    dry_run: bool,
) -> dict:
    """Multi-target fusion with dynamic target isolation (A11)."""
    print("\n[TARGET] MULTI-TARGET FUSION MODE (A11)\n")
    target_config = config_obj.get("target_config", {})
    targets = target_config.get("targets", [])
    print(f"Targets: {[t['name'] for t in targets]}\n")

    branch_name = (
        state_obj.get("anchor_git_branch") or "anchor-baseline"
        if state_obj
        else "anchor-baseline"
    )
    train_path = proc_dir / f"features_train_{branch_name}.csv"
    if not train_path.exists():
        return {"status": "FAILED", "reason": "Train file missing"}
    train = pd.read_csv(train_path)

    # Load raw train for target columns (merge by index since features are encoded)
    raw_train_path = raw_dir / "Train.csv"
    if raw_train_path.exists():
        raw_train = pd.read_csv(raw_train_path)
        # Merge by index since features_train is label-encoded
        _targets_to_merge = []
        for target_spec in targets:
            target_name = target_spec["name"]
            if target_name not in raw_train.columns or target_name in train.columns:
                continue
            target_values = raw_train[target_name].values
            # Encode categorical targets using pd.factorize
            # to match the encoding used by skill_07 and skill_08 multi-target paths.
            if (
                target_spec["task_type"] == "classification"
                and target_values.dtype == object
            ):
                target_values = pd.factorize(target_values)[0]
            _targets_to_merge.append(target_name)
            raw_train[target_name] = target_values
        if _targets_to_merge:
            train = pd.concat(
                [train, raw_train[_targets_to_merge].reset_index(drop=True)],
                axis=1,
            )

    fusion_results = {}
    submission_columns = {}

    for target_spec in targets:
        target_name = target_spec["name"]
        task_type = target_spec["task_type"]
        print(f"\n{'-' * 60}")
        print(f"Fusion: {target_name} ({task_type})")
        print(f"{'-' * 60}")

        # Dynamic config override (A5)
        target_config_override = ChallengeConfig(
            path=config_obj.path,
            _data={
                **config_obj._data,
                "target_col": target_name,
                "task_type": task_type,
                "metric": target_spec.get("metric", config_obj.get("metric")),
                "metric_direction": target_spec.get(
                    "metric_direction", config_obj.get("metric_direction")
                ),
                "use_probabilities": target_spec.get(
                    "use_probabilities", config_obj.get("use_probabilities")
                ),
            },
        )

        # Run single-target fusion logic
        result = _run_single_target_fusion(
            target_config_override,
            state_obj,
            paths,
            proc_dir,
            reports_dir,
            subs_dir,
            raw_dir,
            train,
            in_memory,
            dry_run,
            target_name,
        )
        fusion_results[target_name] = result

        if result["status"] == "OK" and "submission_column" in result:
            submission_columns[target_name] = result["submission_column"]

    # Save target-specific ensembled probabilities
    if not dry_run and submission_columns:
        sample_path = raw_dir / "SampleSubmission.csv"
        if sample_path.exists():
            sample = pd.read_csv(sample_path)
            id_col = str(sample.columns[0])

            # Load raw train data for target alignment
            input_files = config_obj.get("input_files", {}) or {}
            train_file = input_files.get("train", "Train.csv")
            raw_train = pd.read_csv(raw_dir / train_file)

            store = SkillStateStore(paths.state_path)
            from zindian.state import write_oof_record, resolve_active_cv_strategy_id

            for target_name, values in submission_columns.items():
                # Save OOF probs
                target_result = fusion_results[target_name]
                blend_probs = target_result.get("blend_probs")
                if blend_probs is not None:
                    oof_df = pd.DataFrame({id_col: raw_train[id_col], "oof_prob": blend_probs})
                    oof_path = proc_dir / f"oof_ensemble_{target_name}.csv"
                    oof_df.to_csv(oof_path, index=False)
                    print(f"  [OK] Ensemble OOF probabilities saved -> {oof_path}")

                # Save Test probs
                test_prob_df = pd.DataFrame({id_col: sample[id_col], "test_prob": values})
                test_prob_path = proc_dir / f"test_probs_ensemble_{target_name}.csv"
                test_prob_df.to_csv(test_prob_path, index=False)
                print(f"  [OK] Ensemble Test probabilities saved -> {test_prob_path}")

                # Write OOF record to state
                if blend_probs is not None:
                    write_oof_record(
                        store,
                        branch_name=f"ensemble_{target_name}",
                        scores=blend_probs.tolist(),
                        cv_strategy_id=resolve_active_cv_strategy_id(state_obj, config_obj._data),
                        seed=42,
                        model_config={"target_name": target_name, "variant": "ensemble"},
                    )

            # Remove ndarrays from fusion_results to prevent JSON serialization crash
            for r in fusion_results.values():
                if isinstance(r, dict):
                    r.pop("submission_column", None)
                    r.pop("blend_probs", None)

            # Update anchor_git_branch to ensemble so Phase 4 knows to format the ensemble
            if not in_memory:
                store.update(
                    anchor_git_branch="ensemble",
                    last_ensemble_multi_target_results=fusion_results,
                    last_updated=datetime.now(timezone.utc).isoformat(),
                )
            else:
                state_obj["anchor_git_branch"] = "ensemble"
                state_obj["last_ensemble_multi_target_results"] = fusion_results

            return {
                "status": "OK",
                "multi_target": True,
                "results": fusion_results,
            }

    return {
        "status": (
            "OK"
            if all(r.get("status") == "OK" for r in fusion_results.values())
            else "PARTIAL"
        ),
        "multi_target": True,
        "results": fusion_results,
    }


def _run_single_target_fusion(
    config_obj: ChallengeConfig,
    state_obj: dict[str, Any],
    paths: Any,
    proc_dir: Path,
    reports_dir: Path,
    subs_dir: Path,
    raw_dir: Path,
    train: pd.DataFrame,
    in_memory: bool,
    dry_run: bool,
    target_name: str | None = None,
) -> dict:
    """Single-target fusion logic (extracted for reuse)."""
    target_col = _resolve_target_col(config_obj, train)

    task_type = str(config_obj.get("task_type", "classification"))
    metric_name = str(config_obj.get("metric", "f1_score"))
    direction = str(config_obj.get("metric_direction", "maximize"))
    use_probabilities = bool(config_obj.get("use_probabilities", True))

    retraining_active = bool(
        state_obj.get("pseudo_label_result", {}).get("retraining_required", False)
    )

    if task_type == "classification":
        target_values = train[target_col].values
        if target_values.dtype.kind in ("U", "S", "O"):
            # Encode string labels with pd.factorize to match training pipeline
            # pd.factorize returns np.ndarray at runtime; cast for type-checker compatibility
            _codes_raw = np.asarray(pd.factorize(target_values)[0], dtype=np.int32)
            y_true = _codes_raw
        else:
            y_true = np.asarray(target_values, dtype=np.int32)
    else:
        y_true = np.asarray(train[target_col].values, dtype=np.float64)

    print(f"Train rows    : {len(y_true)}")
    if task_type == "classification":
        print(f"Positive rate : {y_true.mean():.3f}")

    # Phase 2: Verified candidate extraction & pruning
    all_variants = _collect_verified_candidates(
        state_obj,
        y_true,
        task_type=task_type,
        metric_name=metric_name,
        direction=direction,
        use_probabilities=use_probabilities,
        retraining_active=retraining_active,
    )
    if not all_variants:
        return {
            "status": "FAILED",
            "reason": "No verified candidates in SKILL_STATE with Human Gate 2 approvals",
        }

    print(f"\nVerified candidates ranked by {metric_name} (found {len(all_variants)}):")
    print(f"  {'Branch':<36} {'Score':>12}")
    print("  " + "-" * 50)
    for v in all_variants:
        print(f"  {v['name']:<36} {v['score']:>12.6f}")

    pruned_variants, dropped_pairs = _prune_collinear(
        all_variants, task_type=task_type, direction=direction
    )
    if not pruned_variants:
        return {
            "status": "FAILED",
            "reason": "All verified candidates dropped by correlation pruning",
        }

    selected = pruned_variants[: int(config_obj.get("fusion_top_n", TOP_N) or TOP_N)]
    names = [v["name"] for v in selected]
    print(f"\nSelected after pruning: {names}")
    if dropped_pairs:
        print("Dropped collinear candidates:")
        for row in dropped_pairs:
            print(
                f"  dropped={row['dropped']} kept={row['kept']} corr={row['correlation']:.5f}"
            )

    # Validate Human Gate 3 as a read-only gate for non-dry execution.
    if not dry_run and not bool(state_obj.get("human_gate_3_approved", False)):
        return {"status": "BLOCKED", "reason": "Human Gate 3 missing"}

    # Check selected test files.
    test_map = _check_test_files(proc_dir, reports_dir, names)
    if test_map is None:
        return {"status": "FAILED", "reason": "Missing test prob files"}

    # Phase 3: Normalized optimization loop
    oof_matrix = np.array([v["probs"] for v in selected], dtype=np.float64)
    blend_probs = oof_matrix.mean(axis=0)

    blend_score, blend_threshold = _score_predictions(
        y_true,
        blend_probs,
        task_type=task_type,
        metric_name=metric_name,
        use_probabilities=use_probabilities,
    )

    if target_name:
        # Multi-target path: resolve target-specific anchor score from the metrics dict
        multi_metrics = state_obj.get("anchor_multi_target_metrics", {})
        target_metrics = multi_metrics.get(target_name, {})
        if task_type == "regression":
            metric_key = "oof_rmse"
        else:
            metric_key = (
                "oof_f1" if metric_name.lower() in ("f1", "f1_score") else "oof_auc"
            )
        anchor_score = float(target_metrics.get(metric_key) or 0.0)
    else:
        if retraining_active:
            anchor_score = float(
                state_obj.get(
                    "anchor_oof_score_augmented",
                    state_obj.get("anchor_oof_score") or 0.0,
                )
                or 0.0
            )
        else:
            anchor_score = float(state_obj.get("anchor_oof_score") or 0.0)

    print(f"\nBlend OOF metric ({metric_name}): {blend_score:.6f}")
    if blend_threshold is not None:
        print(f"Blend threshold            : {blend_threshold:.2f}")
    print(f"Anchor metric reference    : {anchor_score:.6f}")
    if direction == "minimize":
        print(
            f"Delta (blend-anchor)       : {blend_score - anchor_score:+.6f} (lower is better)"
        )
    else:
        print(
            f"Delta (blend-anchor)       : {blend_score - anchor_score:+.6f} (higher is better)"
        )

    # Blend test predictions
    test_matrices = []
    for name, test_path in test_map.items():
        df = pd.read_csv(test_path)
        prob_col = [c for c in df.columns if c != "ID"][0]
        test_matrices.append(np.asarray(df[prob_col].values, dtype=np.float64))

    test_blend = np.array(test_matrices, dtype=np.float64).mean(axis=0)
    if task_type == "classification" and not use_probabilities:
        thr = 0.5 if blend_threshold is None else float(blend_threshold)
        submission_values = (test_blend >= thr).astype(int)
    else:
        submission_values = test_blend

    if task_type == "classification" and not use_probabilities:
        print(
            f"\nTest predictions: {int(submission_values.sum())} present, "
            f"{int((submission_values == 0).sum())} absent"
        )

    # Dry-run exit
    if dry_run:
        print("\n[DRY-RUN] No files saved.")
        return {
            "status": "DRY_RUN",
            "blend_oof_metric": float(blend_score),
            "metric": metric_name,
            "threshold": (None if blend_threshold is None else float(blend_threshold)),
            "variants": names,
            "dropped_collinear": dropped_pairs,
        }

    # Phase 4: Stateless output and gate compliance
    sample_path = raw_dir / "SampleSubmission.csv"
    if not sample_path.exists():
        print(f"SampleSubmission.csv not found at {sample_path}")
        return {"status": "FAILED", "reason": "SampleSubmission missing"}

    sample = pd.read_csv(sample_path)
    if len(submission_values) != len(sample):
        print(
            f"Row count mismatch: preds={len(submission_values)}, sample={len(sample)}"
        )
        return {"status": "FAILED", "reason": "Row count mismatch"}

    # Return ensembled predictions and metrics to the caller
    return {
        "status": "OK",
        "blend_oof_metric": float(blend_score),
        "metric": metric_name,
        "threshold": (None if blend_threshold is None else float(blend_threshold)),
        "variants": names,
        "dropped_collinear": dropped_pairs,
        "submission_column": submission_values,
        "blend_probs": blend_probs,
    }


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    result = run(dry_run=dry_run)
    print("\n" + json.dumps(result, indent=2))
