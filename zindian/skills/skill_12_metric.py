"""Skill 12 — Metric analysis (SoT-aligned).

This implementation follows the Source of Truth contract:
- Reads `SKILL_STATE.json["eda"]["fold_scores"]` (safe `.get()` access)
- Computes unbiased sample variance with `ddof=1`
- Writes `state["metric_analysis"]` with the results for downstream
  consumers (e.g. `skill_11`)

The function is defensive: if `fold_scores` is missing, it writes a
helpful diagnostic into `metric_analysis` rather than raising.
"""

from __future__ import annotations
import tabula.skill_state_autopatch  # noqa

import json
from datetime import datetime, timezone
from typing import Any, Dict, Sequence

import numpy as np

from zindian.paths import resolve_competition_paths
from zindian.state import SkillStateStore


def _get_nb_factor(
    K: int, fold_sizes: Sequence[tuple[int, int]] | None = None
) -> float:
    if K <= 1:
        return 0.0
    if fold_sizes and len(fold_sizes) == K:
        ratios = []
        for n_train, n_val in fold_sizes:
            if n_train > 0:
                ratios.append(float(n_val) / float(n_train))
            else:
                ratios.append(0.0)
        mean_ratio = float(np.mean(ratios))
        gamma_bar = min(mean_ratio, 1.0)
        return (1.0 / K) + gamma_bar
    else:
        return (1.0 / K) + (1.0 / (K - 1))


def run(config: Any = None, state: Dict[str, Any] | None = None) -> Dict[str, Any]:
    print("=" * 60)
    print("SKILL 12 — Metric Analysis")
    print("=" * 60)
    in_memory = state is not None
    if not in_memory:
        paths = resolve_competition_paths(require_competition=True)
        if paths.state_path is None:
            raise FileNotFoundError("State path could not be resolved")

        state_store = SkillStateStore(paths.state_path)
        state = state_store.read()
    else:
        assert state is not None

    active_branch = (
        state.get("best_variant_this_round")
        or state.get("best_variant_branch")
        or state.get("current_active_branch")
        or state.get("anchor_git_branch")
        or "anchor-baseline"
    )

    if config is None:
        try:
            from zindian.config import ChallengeConfig

            config = ChallengeConfig.load()
        except Exception:
            config = {}

    target_config = config.get("target_config", {}) if config else {}
    targets = target_config.get("targets", []) if target_config else []
    # S3 - implemented 2026-08-03
    use_inverse_variance_weighting = bool(
        (config.get("use_inverse_variance_weighting", False) if config else False)
        or (
            target_config.get("use_inverse_variance_weighting", False)
            if isinstance(target_config, dict)
            else False
        )
    )

    fold_scores = None
    recommended_threshold = 0.5
    metric_analysis: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    if targets:
        # Multi-target variance calculation
        target_fold_scores = {}
        target_fold_sizes = {}
        for t in targets:
            t_name = t.get("name")
            if not t_name:
                continue
            oof_key = f"branch_{active_branch}_{t_name}_oof"
            if oof_key in state:
                oof_dict = state[oof_key]
                if isinstance(oof_dict, dict):
                    model_config = oof_dict.get("model_config", {}) or {}
                    t_fold_scores = model_config.get("fold_scores")
                    if t_fold_scores:
                        target_fold_scores[t_name] = t_fold_scores
                        target_fold_sizes[t_name] = model_config.get("fold_sizes")

        # If all targets have fold scores, calculate composite fold scores
        if len(target_fold_scores) == len(targets):
            first_scores = next(iter(target_fold_scores.values()))
            n_splits = len(first_scores) if isinstance(first_scores, list) else 0
            composite_fold_scores = []
            per_target_analysis: Dict[str, Any] = {}
            target_effective_weights: dict[str, float] = {}

            for t in targets:
                t_name = t.get("name")
                if not t_name:
                    continue
                scores_arr = np.asarray(target_fold_scores[t_name], dtype=np.float64)
                K = len(scores_arr)
                variance_sample = float(np.var(scores_arr, ddof=1)) if K > 1 else 0.0
                nb_factor = _get_nb_factor(K, target_fold_sizes.get(t_name))
                variance_nb = float(variance_sample * nb_factor)

                base_weight = float(t.get("weight", 0.5) or 0.0)
                effective_weight = (
                    base_weight / (variance_nb + 1e-8)
                    if use_inverse_variance_weighting
                    else base_weight
                )
                target_effective_weights[str(t_name)] = effective_weight
                # [v2.7] Per-target standard error — NB-based, all targets (L2).
                # matches the top-level convention se_oof = sqrt(Var_NB) with no /K.
                per_target_analysis[str(t_name)] = {
                    "fold_scores": [float(value) for value in scores_arr.tolist()],
                    "fold_score_variance": variance_nb,
                    "fold_score_variance_sample": variance_sample,
                    "weight": base_weight,
                    "effective_weight": effective_weight,
                    "se_oof": float(np.sqrt(variance_nb)) if variance_nb > 0 else 0.0,
                }

            # [v2.7] Composite standard error for the promotion margin (Fix-1 + Fix-3):
            #   - regression targets ONLY (matches effective_target_std / D0-D4 scope)
            #   - sqrt( sum(w_eff * per-target NB variance) ) with NO extra /sqrt(K):
            #     each per-target variance is already NB-corrected, so dividing by the
            #     fold count again is the A.1 double-scaling error.
            # Uses the SAME effective weights as the composite distance computation.
            regression_targets = [
                t for t in targets
                if t.get("task_type") == "regression" and t.get("name")
            ]
            if regression_targets and per_target_analysis:
                composite_se_oof = float(
                    np.sqrt(
                        sum(
                            target_effective_weights.get(str(rt["name"]), 0.0)
                            * per_target_analysis[str(rt["name"])]["fold_score_variance"]
                            for rt in regression_targets
                            if rt["name"] in per_target_analysis
                        )
                    )
                )
                metric_analysis["composite_se_oof"] = composite_se_oof

            for i in range(n_splits):
                weighted_sum = 0.0
                total_weight = 0.0
                for t in targets:
                    t_name = t.get("name")
                    weight = target_effective_weights.get(
                        str(t_name), float(t.get("weight", 0.5) or 0.0)
                    )
                    task_type = t.get("task_type", "classification")
                    raw_score = target_fold_scores[t_name][i]

                    if task_type == "regression":
                        eda_std = float(
                            state.get("eda", {}).get(f"{t_name}_std", 0.0) or 0.0
                        )
                        if eda_std <= 0.0:
                            eda_std = float(
                                state.get("eda", {}).get("target_std", 1.0) or 1.0
                            )
                        score_val = raw_score / eda_std if eda_std > 0 else raw_score
                    else:
                        score_val = 1.0 - raw_score

                    weighted_sum += score_val * weight
                    total_weight += weight

                composite_fold_scores.append(
                    weighted_sum / total_weight if total_weight > 0 else weighted_sum
                )

            if composite_fold_scores:
                fold_scores = composite_fold_scores
                # Store composite fold variance separately
                composite_variance = float(np.var(composite_fold_scores, ddof=1))
                metric_analysis["per_target"] = per_target_analysis
                metric_analysis["composite_fold_score_variance"] = composite_variance
                metric_analysis["use_inverse_variance_weighting"] = (
                    use_inverse_variance_weighting
                )

    fold_sizes = None
    if not fold_scores:
        oof_key = f"branch_{active_branch}_oof"
        if oof_key in state:
            oof_dict = state[oof_key]
            if isinstance(oof_dict, dict):
                model_config = oof_dict.get("model_config", {}) or {}
                fold_scores = model_config.get("fold_scores")
                fold_sizes = model_config.get("fold_sizes")
                recommended_threshold = model_config.get("threshold", 0.5)

    # Fallback to search any branch_.*_oof key if not found
    if not fold_scores:
        for key, val in state.items():
            if (
                key.startswith("branch_")
                and key.endswith("_oof")
                and isinstance(val, dict)
            ):
                model_config = val.get("model_config", {}) or {}
                if "fold_scores" in model_config:
                    fold_scores = model_config["fold_scores"]
                    fold_sizes = model_config.get("fold_sizes")
                    recommended_threshold = model_config.get("threshold", 0.5)
                    break

    # Fallback to eda block for backward compatibility
    if not fold_scores:
        eda = state.get("eda", {}) or {}
        if isinstance(eda, dict):
            fold_scores = eda.get("fold_scores")

    if not fold_scores:
        metric_analysis.update(
            {
                "error": "missing_fold_scores",
                "message": (
                    "SKILL_STATE.json missing fold scores metadata. "
                    "Ensure Skill 07 or Skill 08 writes fold_scores inside model_config before running Skill 12."
                ),
            }
        )
        if not in_memory:
            state_store.update(metric_analysis=metric_analysis)
        else:
            state["metric_analysis"] = metric_analysis
        print("[WARN]  metric_analysis written with diagnostic: missing fold_scores")
        return state if in_memory else metric_analysis

    # Ensure numeric array
    arr = np.asarray(fold_scores, dtype=np.float64)
    # Unbiased sample variance (ddof=1)
    fold_score_variance_sample = float(np.var(arr, ddof=1))

    # S1 - implemented 2026-08-03
    # Nadeau-Bengio Corrected Variance (v2.4 S1): Var_NB = Var_sample(ddof=1) * (1/K + n_val/n_train)
    # For K-fold CV: n_val/n_train = 1/(K-1)
    K = len(arr)
    if K > 1:
        nb_factor = _get_nb_factor(K, fold_sizes)
        fold_score_variance_nb = float(fold_score_variance_sample * nb_factor)
        se_oof = float(np.sqrt(fold_score_variance_nb))
    else:
        fold_score_variance_nb = fold_score_variance_sample
        se_oof = 0.0

    # Primary fold_score_variance reports Nadeau-Bengio corrected variance per v2.4 spec
    fold_score_variance = fold_score_variance_nb

    # Calculate oof_vs_lb_delta if possible
    # Use provided config dict if available; fallback to ChallengeConfig only if needed.
    if config is None:
        try:
            from zindian.config import ChallengeConfig

            config_obj = ChallengeConfig.load()
            metric_key = str(config_obj.get("metric", "f1")).lower()
        except Exception:
            metric_key = "f1"
    else:
        metric_key = str(config.get("metric", "f1")).lower()

    oof_score = None
    if active_branch == "anchor-baseline":
        oof_score = state.get(f"anchor_oof_{metric_key}")
    else:
        oof_score = state.get(f"best_variant_oof_{metric_key}")

    if oof_score is None:
        oof_score = state.get("anchor_oof_score") or state.get("best_variant_oof_score")

    lb_score = state.get("last_lb_score") or state.get("best_lb_score")
    oof_vs_lb_delta = None
    if oof_score is not None and lb_score is not None:
        try:
            oof_vs_lb_delta = float(oof_score) - float(lb_score)
        except (ValueError, TypeError):
            pass

    metric_analysis.update(
        {
            "fold_scores": fold_scores,
            "fold_score_variance_sample": fold_score_variance_sample,
            "fold_score_variance_nb": fold_score_variance_nb,
            "fold_score_variance": fold_score_variance,
            "se_oof": se_oof,
            "recommended_threshold": float(recommended_threshold),
            "oof_vs_lb_delta": oof_vs_lb_delta,
        }
    )

    if not in_memory:
        state_store.update(metric_analysis=metric_analysis)
    else:
        state["metric_analysis"] = metric_analysis
    print(f"  - Active branch: {active_branch}")
    print(f"  - Number of splits K: {K}")
    print(f"  - OOF score: {oof_score}")
    print(f"  - LB score: {lb_score}")
    print(f"  - OOF-vs-LB delta: {oof_vs_lb_delta}")
    print(f"  - Fold score sample variance (ddof=1): {fold_score_variance_sample:.6g}")
    print(f"  - Nadeau-Bengio corrected variance: {fold_score_variance_nb:.6g}")
    print(f"  - Recommended classification threshold: {recommended_threshold:.4f}")
    print(f"[OK] metric_analysis written (variance ddof=1 = {fold_score_variance:.6g})")
    return state if in_memory else metric_analysis


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
