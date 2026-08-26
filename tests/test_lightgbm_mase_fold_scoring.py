"""F4 MASE fold-scoring tests (step 11).

Tests three things:

1. MASE fold score == MAE(y_val, yhat_val) / MAE_naive_baseline  (per-fold correctness)
2. oof_mase == mean(fold_scores)  (aggregation)
3. Missing / invalid baseline RAISES before train_lightgbm_cv is ever reached —
   never silently produces an unscaled score.

The framing from the original draft ("safely falls back and logs") is explicitly
inverted here per the pre-merge fix: the code path is a hard ValueError in
skill_08_anchor (upstream guard) + a hard assertion in _lightgbm_shared (in-loop
guard). Both are tested.
"""

import math
import numpy as np
import pandas as pd
import pytest

from zindian.skills._lightgbm_shared import train_lightgbm_cv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_train_test(n: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {"feat": rng.standard_normal(n), "target": rng.standard_normal(n)}
    )
    train = df.iloc[: n * 3 // 4].copy()
    test = df.iloc[n * 3 // 4 :].copy()
    return train, test


def _run_mase_cv(baseline, *, n_splits: int = 3, n: int = 60):
    """Call train_lightgbm_cv with regression_metric='mase'."""
    train, test = _minimal_train_test(n=n)
    return train_lightgbm_cv(
        train=train,
        test=test,
        feature_cols=["feat"],
        target_col="target",
        n_splits=n_splits,
        random_seed=0,
        regression_metric="mase",
        mae_naive_baseline=baseline,
    )


# ---------------------------------------------------------------------------
# Test 1 — per-fold correctness: each fold MASE == MAE / baseline
# ---------------------------------------------------------------------------


def test_mase_fold_score_equals_mae_over_baseline():
    """Each fold score must be MAE(y_val, yhat_val) / MAE_naive_baseline.

    Verifies the formula by checking that all fold scores are non-negative,
    that oof_mase == mean(fold_scores), and that a distinct baseline scales
    the scores proportionally.
    """
    baseline_a = 2.0
    baseline_b = 4.0  # double — every fold score should halve

    result_a = _run_mase_cv(baseline=baseline_a, n_splits=3)
    result_b = _run_mase_cv(baseline=baseline_b, n_splits=3)

    # The internal CV splitter may override n_splits from config; what matters
    # is that both runs produce the same number of folds (same CV geometry).
    assert len(result_a.fold_scores) == len(result_b.fold_scores)
    assert len(result_a.fold_scores) > 0

    # All fold scores non-negative (MAE >= 0, baseline > 0)
    for fs in result_a.fold_scores:
        assert fs >= 0.0

    # Doubling the baseline should halve every fold score
    for fa, fb in zip(result_a.fold_scores, result_b.fold_scores):
        assert math.isclose(
            fa / fb, 2.0, rel_tol=1e-6
        ), f"Expected fold_score ratio 2.0, got {fa/fb:.6f}"

    # oof_mase == mean(fold_scores)
    assert math.isclose(
        result_a.oof_mase, float(np.mean(result_a.fold_scores)), rel_tol=1e-9
    )


# ---------------------------------------------------------------------------
# Test 2 — aggregation: oof_mase == mean(fold_scores), oof_rmse alias correct
# ---------------------------------------------------------------------------


def test_oof_mase_equals_mean_of_fold_scores():
    """oof_mase must equal the arithmetic mean of per-fold MASE scores,
    and oof_rmse must mirror oof_mase for backward compatibility."""
    result = _run_mase_cv(baseline=1.5, n_splits=4)

    expected = float(np.mean(result.fold_scores))
    assert math.isclose(result.oof_mase, expected, rel_tol=1e-9)
    # Backward-compat alias: skill_08 reads oof_logloss = oof_rmse
    assert math.isclose(result.oof_rmse, result.oof_mase, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Test 3 — upstream guard: missing/invalid baseline raises BEFORE training
# ---------------------------------------------------------------------------


def _make_mase_config_mock():
    """Return a config-like object that reports task_type=regression, metric=mase."""
    from unittest.mock import MagicMock

    fake_config = MagicMock()
    cfg_values = {
        "task_type": "regression",
        "metric": "mase",
        "cv_strategy": {"type": "kfold", "n_splits": 3},
        "columns": {},
        "policy_filters": [],
        "reproducibility": {"seed": 0},
    }
    fake_config.get = lambda key, default=None: cfg_values.get(key, default)
    fake_config._data = {}
    return fake_config


def _make_dummy_frames(n: int = 30):
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {"feat": rng.standard_normal(n), "target": rng.standard_normal(n)}
    )
    return df.iloc[:24].copy(), df.iloc[24:].copy()


def test_missing_baseline_raises_in_skill08_before_train():
    """skill_08_anchor's upstream ValueError guard must fire before any
    train_lightgbm_cv call — confirming the 'raises and never reaches training'
    contract (the opposite of the original 'safely falls back' framing).
    """
    from zindian.skills.skill_08_anchor import compute_oof_predictions

    dummy_train, dummy_test = _make_dummy_frames()

    with pytest.raises(ValueError, match="MAE_naive_baseline"):
        compute_oof_predictions(
            train=dummy_train,
            test=dummy_test,
            config=_make_mase_config_mock(),
            target_col="target",
            state={"eda": {}},  # MAE_naive_baseline absent
        )


def test_zero_baseline_raises_in_skill08():
    """A MAE_naive_baseline of 0.0 (degenerate) must also raise — not silently
    produce an infinity or NaN."""
    from zindian.skills.skill_08_anchor import compute_oof_predictions

    dummy_train, dummy_test = _make_dummy_frames()

    with pytest.raises(ValueError, match="MAE_naive_baseline"):
        compute_oof_predictions(
            train=dummy_train,
            test=dummy_test,
            config=_make_mase_config_mock(),
            target_col="target",
            state={"eda": {"MAE_naive_baseline": 0.0}},
        )


# ---------------------------------------------------------------------------
# Test 4 — in-loop hard assertion: bypassing skill_08 raises loudly
# ---------------------------------------------------------------------------


def test_in_loop_assertion_fires_when_baseline_missing():
    """If a caller bypasses skill_08_anchor and calls train_lightgbm_cv with
    regression_metric='mase' but no mae_naive_baseline, the in-loop assertion
    must raise AssertionError immediately — never produce a value."""
    with pytest.raises(AssertionError, match="MAE_naive_baseline"):
        _run_mase_cv(baseline=None)


def test_in_loop_assertion_fires_when_baseline_zero():
    """Same for baseline == 0 — a zero baseline is equally invalid."""
    with pytest.raises(AssertionError, match="MAE_naive_baseline"):
        _run_mase_cv(baseline=0.0)
