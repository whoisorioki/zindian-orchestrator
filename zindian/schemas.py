from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple


@dataclass(frozen=False)
class ValidationError(Exception):
    message: str
    path: str = ""

    def __str__(self) -> str:
        return f"{self.path + ': ' if self.path else ''}{self.message}"


def _require_keys(d: Dict[str, Any], keys: Iterable[str], *, path: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise ValidationError(f"missing keys {missing}", path=path)


def _is_jsonable_scalar(x: Any) -> bool:
    return x is None or isinstance(x, (str, int, float, bool))


SKILL_STATE_KEYS: Tuple[str, ...] = (
    "competition",
    "md5_target_hash",
    "anchor_oof_f1",
    "anchor_oof_rmse",
    "anchor_lb_score",
    "submissions_used_today",
    "submissions_used_total",
    "remaining_submissions",
    "dag_phase",
    "selected_submissions",
    "last_updated",
)


CHALLENGE_CONFIG_KEYS: Tuple[str, ...] = (
    "name",
    "slug",
    "metric",
    "metric_direction",
    "submission_format",
    "use_probabilities",
    "daily_limit",
    "total_limit",
    "public_split_pct",
    "private_split_pct",
    "team_allowed",
    "code_review_tier",
    "allowed_external_data",
    "automl_permitted",
    "data_modality",
    "domain",
)


def validate_skill_state(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValidationError("SKILL_STATE must be a JSON object")
    _require_keys(obj, SKILL_STATE_KEYS, path="SKILL_STATE")

    # Basic type checks (allow nulls where schema permits).
    for k in SKILL_STATE_KEYS:
        if k not in ("selected_submissions",):
            if not _is_jsonable_scalar(obj[k]):
                raise ValidationError("must be scalar/null", path=f"SKILL_STATE.{k}")

    if not isinstance(obj["submissions_used_today"], int):
        raise ValidationError("must be int", path="SKILL_STATE.submissions_used_today")
    if not isinstance(obj["submissions_used_total"], int):
        raise ValidationError("must be int", path="SKILL_STATE.submissions_used_total")
    if not isinstance(obj["selected_submissions"], list):
        raise ValidationError("must be list", path="SKILL_STATE.selected_submissions")
    return obj


def validate_challenge_config(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValidationError("challenge_config must be a JSON object")
    _require_keys(obj, CHALLENGE_CONFIG_KEYS, path="challenge_config")

    # Minimal sanity checks; do not over-constrain (Zindi pages vary).
    if obj.get("metric_direction") not in (None, "minimize", "maximize"):
        raise ValidationError("must be 'minimize' or 'maximize' or null", path="challenge_config.metric_direction")
    if not isinstance(obj.get("use_probabilities"), bool):
        raise ValidationError("must be boolean", path="challenge_config.use_probabilities")
    if not isinstance(obj.get("allowed_external_data"), bool):
        raise ValidationError("must be boolean", path="challenge_config.allowed_external_data")
    if not isinstance(obj.get("automl_permitted"), bool):
        raise ValidationError("must be boolean", path="challenge_config.automl_permitted")
    return obj


def skill_state_skeleton() -> Dict[str, Any]:
    return {
        "competition": None,
        "md5_target_hash": None,
        "anchor_oof_f1": None,
        # Legacy compatibility field; downstream logic should prefer anchor_oof_f1.
        "anchor_oof_rmse": None,
        "anchor_lb_score": None,
        "submissions_used_today": 0,
        "submissions_used_total": 0,
        "remaining_submissions": None,
        "dag_phase": "uninitialized",
        "selected_submissions": [],
        "last_updated": None,
    }

