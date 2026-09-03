"""Shared metric helpers.

Single source of truth for competition composite scoring. The Zindi
leaderboard composite for this competition family is

    composite = 0.6 * F1 + 0.4 * ROC-AUC

(verified against every submission in
`reports/submissions_manifest.json` to <5e-7 residual). Gate
comparisons, anchor confirmation, and variant scoring must all use
this helper so the internal metric matches the leaderboard metric.
"""


from typing import NamedTuple, Union


class ScoreProvenance(NamedTuple):
    value: float
    origin: str  # "oof" or "lb"


def oof_score(value: float) -> ScoreProvenance:
    return ScoreProvenance(float(value), "oof")


def lb_score(value: float) -> ScoreProvenance:
    return ScoreProvenance(float(value), "lb")


def composite_metric(
    oof_f1: Union[float, ScoreProvenance],
    oof_auc: Union[float, ScoreProvenance],
    *,
    f1_origin: str = "oof",
    auc_origin: str = "oof",
) -> float:
    """Return the leaderboard-aligned composite score.

    Parameters must be genuine OOF values. Never pass leaderboard
    metrics here — the composite is used as an internal gate baseline,
    and mixing LB data into it breaks OOF/LB separation.

    Raises:
        ValueError: If any input score is tagged with LB provenance ('lb').
    """
    if isinstance(oof_f1, ScoreProvenance):
        f1_val = oof_f1.value
        f1_orig = oof_f1.origin
    elif hasattr(oof_f1, "origin"):
        f1_val = float(getattr(oof_f1, "value", oof_f1))
        f1_orig = getattr(oof_f1, "origin")
    else:
        f1_val = float(oof_f1)
        f1_orig = f1_origin

    if isinstance(oof_auc, ScoreProvenance):
        auc_val = oof_auc.value
        auc_orig = oof_auc.origin
    elif hasattr(oof_auc, "origin"):
        auc_val = float(getattr(oof_auc, "value", oof_auc))
        auc_orig = getattr(oof_auc, "origin")
    else:
        auc_val = float(oof_auc)
        auc_orig = auc_origin

    if f1_orig == "lb" or auc_orig == "lb":
        raise ValueError(
            "composite_metric prohibits Leaderboard (LB) scores. Only OOF scores are allowed."
        )
    if f1_orig != "oof" or auc_orig != "oof":
        raise ValueError(
            f"composite_metric received invalid provenance: f1={f1_orig}, auc={auc_orig}"
        )

    return 0.6 * f1_val + 0.4 * auc_val
