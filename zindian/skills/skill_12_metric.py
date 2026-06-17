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
from typing import Any, Dict

import numpy as np

from zindian.paths import resolve_competition_paths
from zindian.state import SkillStateStore


def run(
    config: Dict[str, Any] | None = None, state: Dict[str, Any] | None = None
) -> Dict[str, Any]:
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

    oof_key = f"branch_{active_branch}_oof"
    fold_scores = None
    recommended_threshold = 0.5

    if oof_key in state:
        oof_dict = state[oof_key]
        if isinstance(oof_dict, dict):
            model_config = oof_dict.get("model_config", {}) or {}
            fold_scores = model_config.get("fold_scores")
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
                    recommended_threshold = model_config.get("threshold", 0.5)
                    break

    # Fallback to eda block for backward compatibility
    if not fold_scores:
        eda = state.get("eda", {}) or {}
        if isinstance(eda, dict):
            fold_scores = eda.get("fold_scores")

    metric_analysis: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

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
        print("⚠️  metric_analysis written with diagnostic: missing fold_scores")
        return state if in_memory else metric_analysis

    # Ensure numeric array
    arr = np.asarray(fold_scores, dtype=np.float64)
    # Unbiased sample variance (ddof=1) per SoT
    fold_score_variance = float(np.var(arr, ddof=1))

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
            "fold_score_variance": fold_score_variance,
            "recommended_threshold": float(recommended_threshold),
            "oof_vs_lb_delta": oof_vs_lb_delta,
        }
    )

    if not in_memory:
        state_store.update(metric_analysis=metric_analysis)
    else:
        state["metric_analysis"] = metric_analysis
    print(f"✅ metric_analysis written (variance ddof=1 = {fold_score_variance:.6g})")
    return state if in_memory else metric_analysis


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
