from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .schemas import skill_state_skeleton, validate_skill_state


def _iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(data, indent=2, sort_keys=False) + "\n"

    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=str(path.parent), encoding="utf-8"
    ) as tmp:
        tmp.write(serialized)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


@dataclass
class SkillStateStore:
    path: Path

    def read(self) -> Dict[str, Any]:
        if not self.path.exists():
            state = skill_state_skeleton()
            _atomic_write_json(self.path, state)
            return state
        obj = json.loads(self.path.read_text(encoding="utf-8"))
        return validate_skill_state(obj)

    def write(
        self, new_state: Dict[str, Any], *, touch_timestamp: bool = True
    ) -> Dict[str, Any]:
        state = dict(new_state)
        if touch_timestamp:
            state["last_updated"] = _iso_now()
        validate_skill_state(state)
        _atomic_write_json(self.path, state)
        return state

    def update(self, **patch: Any) -> Dict[str, Any]:
        state = self.read()
        state.update(patch)
        return self.write(state)

    def increment(self, key: str, delta: int = 1) -> int:
        """Increment a numeric field and return new value."""
        state = self.read()
        if key not in state:
            state[key] = 0
        state[key] = state[key] + delta
        self.write(state)
        return state[key]

    def append_selected(self, submission_id: int) -> None:
        """Append submission to selected_submissions list."""
        state = self.read()
        sel = state.get("selected_submissions")
        if not isinstance(sel, list):
            sel = []
        if submission_id not in sel:
            sel.append(submission_id)
        state["selected_submissions"] = sel
        self.write(state)


def resolve_active_cv_strategy_id(state_obj: dict, config_obj: dict) -> str:
    """
    Resolve the active CV strategy identifier according to the Source of Truth rules.

    Priority:
      1. If SKILL_STATE contains an active `cv_strategy_override.active` == True,
         return an 'override:<override_strategy>' identifier.
      2. Else, read `challenge_config.json` cv_strategy block and return
         'config:<type>' identifier.
      3. Fallback to 'unknown'.

    This function returns a short string suitable for tagging OOF artifacts
    and SKILL_STATE entries.
    """
    try:
        override = state_obj.get("cv_strategy_override", {}) or {}
        if override.get("active", False):
            return f"override:{override.get('override_strategy') or 'unknown'}"
    except Exception:
        pass

    try:
        cv = (
            (config_obj or {}).get("cv_strategy")
            if isinstance(config_obj, dict)
            else None
        )
        if isinstance(cv, dict):
            return f"config:{cv.get('type', 'unknown')}"
    except Exception:
        pass

    return "unknown"


def compute_secondary_metrics(y_true: Any, y_pred: Any) -> dict[str, Any]:
    """Calculate regression diagnostics (MAE, MAPE, R2) on concatenated arrays."""
    from sklearn.metrics import mean_absolute_error, r2_score
    import numpy as np

    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)

    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    r2 = float(r2_score(y_true_arr, y_pred_arr))

    # Guard against division-by-zero for MAPE
    non_zero = y_true_arr != 0
    if np.sum(non_zero) > 0:
        mape: float | None = float(np.mean(np.abs((y_true_arr[non_zero] - y_pred_arr[non_zero]) / y_true_arr[non_zero])))
    else:
        mape = None  # SOT/user correction: mape is None when all targets are zero

    return {"mae": mae, "mape": mape, "r2": r2}


def write_oof_record(
    store: SkillStateStore,
    *,
    branch_name: str,
    scores: Any,
    cv_strategy_id: str,
    seed: int,
    model_config: dict[str, Any],
    secondary_metrics: dict[str, Any] | None = None,
    touch_timestamp: bool = True,
) -> dict[str, Any]:
    """Persist a SoT-shaped OOF record under `branch_{branch_name}_oof`."""
    if isinstance(scores, (list, tuple)):
        score_list = [float(value) for value in scores]
    else:
        score_list = [float(scores)]

    record = {
        "scores": score_list,
        "cv_strategy_id": str(cv_strategy_id),
        "seed": int(seed),
        "branch_name": str(branch_name),
        "model_config": dict(model_config),
    }
    if secondary_metrics is not None:
        record["secondary_metrics"] = secondary_metrics


    state = store.read()
    key = f"branch_{branch_name}_oof"
    # Enforce SoT retraining rules: when pseudo-label retraining is active,
    # augmented outputs must use the `_augmented` suffix and original keys
    # must not be overwritten. This prevents accidental overwrites of baseline OOFs.
    retraining_active = bool(
        (state.get("pseudo_label_result") or {}).get("retraining_required", False)
    )
    if retraining_active and not str(branch_name).endswith("_augmented"):
        raise RuntimeError(
            "Retraining active: OOF records during retraining must use the '_augmented' suffix for branch_name"
        )
    # Prevent overwriting original non-augmented key when retraining
    original_key = f"branch_{str(branch_name).removesuffix('_augmented')}_oof"
    if (
        retraining_active
        and original_key in state
        and not str(branch_name).endswith("_augmented")
    ):
        raise RuntimeError(
            f"Retraining attempted to overwrite original OOF key: {original_key}. Write to '{original_key}_augmented' instead."
        )

    state[key] = record
    store.write(state, touch_timestamp=touch_timestamp)
    return record


def is_anchor_challenge_active(state_obj: dict) -> bool:
    """Safe accessor for anchor_challenge.active in SKILL_STATE.

    Returns True only if `anchor_challenge` is present and has `active`==True.
    This protects automation from KeyError when the block is absent.
    """
    try:
        return bool((state_obj or {}).get("anchor_challenge", {}).get("active", False))
    except Exception:
        return False


def get_anchor_challenge_config(state_obj: dict) -> dict:
    """Return the `anchor_challenge` config block or empty dict if absent."""
    try:
        return dict((state_obj or {}).get("anchor_challenge") or {})
    except Exception:
        return {}
