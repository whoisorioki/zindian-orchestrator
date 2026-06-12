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

    eda = state.get("eda", {})
    fold_scores = eda.get("fold_scores")

    metric_analysis: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    if not fold_scores:
        metric_analysis.update(
            {
                "error": "missing_fold_scores",
                "message": (
                    "SKILL_STATE.json missing 'eda.fold_scores'. "
                    "Ensure Skill 05 or the EDA writes per-fold scores before running Skill 12."
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

    metric_analysis.update(
        {
            "fold_scores": fold_scores,
            "fold_score_variance": fold_score_variance,
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
