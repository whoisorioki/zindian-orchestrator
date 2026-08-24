from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Dict

from .schemas import skill_state_skeleton, validate_skill_state


def _iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    # Externalize large payloads if this is SKILL_STATE.json
    if "SKILL_STATE.json" in str(path):
        scores_dir = path.parent / "scores"
        scores_dir.mkdir(exist_ok=True)

        _EDA_KEYS = frozenset(
            {
                "band_summary_stats",
                "temporal_trends",
                "target_correlation_per_feature",
                "class_separability_index",
            }
        )

        data_copy = {}
        for key, value in data.items():
            # Externalize large OOF score lists (>100 entries)
            if isinstance(value, dict) and "scores" in value:
                scores = value["scores"]
                if isinstance(scores, list) and len(scores) > 100:
                    score_file = scores_dir / f"{key}.json"
                    with open(score_file, "w") as sf:
                        json.dump(scores, sf)

                    new_value = {k: v for k, v in value.items() if k != "scores"}
                    new_value["scores_file"] = f"scores/{key}.json"
                    new_value["count"] = len(scores)
                    data_copy[key] = new_value
                else:
                    data_copy[key] = value
            # Externalize large EDA metric dicts (>10 keys)
            elif key in _EDA_KEYS and isinstance(value, dict) and len(value) > 10:
                eda_file = scores_dir / f"eda_{key}.json"
                with open(eda_file, "w") as sf:
                    json.dump(value, sf)

                new_value = {"eda_file": f"scores/eda_{key}.json", "count": len(value)}
                data_copy[key] = new_value
            # Externalize large cv_split_indices lists (>0 elements)
            elif (
                key == "cv_split_indices" and isinstance(value, list) and len(value) > 0
            ):
                splits_file = scores_dir / "cv_split_indices.json"
                with open(splits_file, "w") as sf:
                    json.dump(value, sf)

                data_copy[key] = {
                    "cv_splits_file": "scores/cv_split_indices.json",
                    "count": len(value),
                }
            # Externalize any other top-level list (>100 elements) to prevent state bloat
            elif isinstance(value, list) and len(value) > 100:
                list_file = scores_dir / f"{key}.json"
                with open(list_file, "w") as sf:
                    json.dump(value, sf)

                data_copy[key] = {
                    "list_file": f"scores/{key}.json",
                    "count": len(value),
                }
            else:
                data_copy[key] = value
        data = data_copy

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

    # Class-level lock: serializes all read-modify-write operations across
    # ALL instances in the same process. Required because run_deep_research
    # spawns a daemon thread that creates its own SkillStateStore instance;
    # an instance-level lock would not protect against that cross-instance race.
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def read(self) -> Dict[str, Any]:
        if not self.path.exists():
            state = skill_state_skeleton()
            _atomic_write_json(self.path, state)
            return state
        obj = json.loads(self.path.read_text(encoding="utf-8"))

        # Hydrate externalized scores, EDA metrics, CV split indices, and large lists
        for key, value in obj.items():
            if not isinstance(value, dict):
                continue
            if "scores_file" in value:
                score_path = self.path.parent / value["scores_file"]
                if score_path.exists():
                    with open(score_path, "r") as sf:
                        value["scores"] = json.load(sf)
            elif "eda_file" in value:
                eda_path = self.path.parent / value["eda_file"]
                if eda_path.exists():
                    with open(eda_path, "r") as sf:
                        loaded = json.load(sf)
                        value.update(loaded)
            elif "cv_splits_file" in value:
                splits_path = self.path.parent / value["cv_splits_file"]
                if splits_path.exists():
                    try:
                        with open(splits_path, "r") as sf:
                            obj[key] = json.load(sf)
                    except Exception as e:
                        print(
                            f"[WARN] Corrupted cv_splits_file ({splits_path}): {e}. Falling back to empty list []."
                        )
                        obj[key] = []
                else:
                    obj[key] = []
            elif "list_file" in value:
                list_path = self.path.parent / value["list_file"]
                if list_path.exists():
                    try:
                        with open(list_path, "r") as sf:
                            obj[key] = json.load(sf)
                    except Exception as e:
                        print(
                            f"[WARN] Corrupted list_file ({list_path}): {e}. Falling back to empty list []."
                        )
                        obj[key] = []
                else:
                    obj[key] = []

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
        with self._lock:
            state = self.read()
            state.update(patch)
            return self.write(state)

    def increment(self, key: str, delta: int = 1) -> int:
        """Increment a numeric field and return new value."""
        with self._lock:
            state = self.read()
            if key not in state:
                state[key] = 0
            state[key] = state[key] + delta
            self.write(state)
            return state[key]

    def append_selected(self, submission_id: int) -> None:
        """Append submission to selected_submissions list."""
        with self._lock:
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


def compute_secondary_metrics(
    y_true: Any,
    y_pred: Any,
    *,
    temporal_present: bool = False,
    mae_naive_baseline: float | None = None,
) -> dict[str, Any]:
    """Calculate regression diagnostics on concatenated arrays."""
    from sklearn.metrics import mean_absolute_error, r2_score
    import numpy as np

    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)

    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    r2 = float(r2_score(y_true_arr, y_pred_arr))

    # Guard against division-by-zero for MAPE
    non_zero = y_true_arr != 0
    if np.sum(non_zero) > 0:
        mape: float | None = float(
            np.mean(
                np.abs(
                    (y_true_arr[non_zero] - y_pred_arr[non_zero]) / y_true_arr[non_zero]
                )
            )
        )
    else:
        mape = None  # SOT/user correction: mape is None when all targets are zero

    # S2 - implemented 2026-08-03
    zero_fraction = float(np.mean(y_true_arr == 0))
    metrics: dict[str, Any] = {
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "zero_fraction": zero_fraction,
    }
    if temporal_present:
        baseline = float(mae_naive_baseline or 0.0)
        metrics["mase"] = mae / baseline if baseline > 0.0 else None

    return metrics


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
    retraining_active = bool(
        (state.get("pseudo_label_result") or {}).get("retraining_required", False)
    )
    if retraining_active:
        if not str(branch_name).endswith("_augmented"):
            raise RuntimeError(
                "Retraining active: OOF records during retraining must use the '_augmented' suffix for branch_name"
            )
        base_branch = str(branch_name).removesuffix("_augmented")
        key = f"branch_{base_branch}_oof_augmented"
        original_key = f"branch_{base_branch}_oof"
        if key == original_key:
            raise RuntimeError(
                f"Retraining loop attempted to overwrite original OOF key '{original_key}'. Write to '{key}' instead."
            )
    else:
        key = f"branch_{branch_name}_oof"

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
