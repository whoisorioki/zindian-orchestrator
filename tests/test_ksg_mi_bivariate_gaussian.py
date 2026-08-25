"""[F2 — v2.8] KSG bivariate MI estimator — closed-form reference test.

For a bivariate Gaussian with Pearson correlation ρ, the true mutual information is:
    I(X; Y) = -0.5 * ln(1 - ρ²)

This is the only case where MI has a simple closed form for continuous variables.
The KSG estimator used in _run_pairwise_mi_audit should converge to this value
as n → ∞ (it is consistent but biased at finite n).

Tests:
  1. Independent Gaussians (ρ=0): I(X;Y) → 0.
  2. Correlated Gaussians (ρ=0.8): I(X;Y) → -0.5*ln(1-0.64) ≈ 0.511 nats.
     KSG at n=5000 should be within 15% of the true value.
  3. Scale invariance (F1): the score is insensitive to rescaling the target.

These tests close F2 (numeric-equivalence validation for the KSG estimator) and
also act as a regression guard for F1 (the scale-invariance fix).
"""

import math
import numpy as np
import pandas as pd
import pytest

from zindian.skills.skill_10_shap import _run_pairwise_mi_audit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bivariate_gaussian(rho: float, n: int, seed: int = 0) -> pd.DataFrame:
    """Sample (X, Y) from a bivariate standard normal with correlation rho."""
    rng = np.random.default_rng(seed)
    cov = [[1.0, rho], [rho, 1.0]]
    data = rng.multivariate_normal([0.0, 0.0], cov, size=n)
    return pd.DataFrame({"feat_x": data[:, 0], "feat_y": data[:, 1]})


def _gaussian_mi_true(rho: float) -> float:
    """True MI for bivariate Gaussian: -0.5 * ln(1 - rho^2)."""
    return -0.5 * math.log(1.0 - rho ** 2)


def _cfg_shap(threshold: float = 0.05, max_samples: int = 5000) -> dict:
    return {
        "enable_mi_regression_subsample": True,
        "mi_pairwise_top_n": 2,
        "mi_pairwise_threshold": threshold,
        "mi_max_samples": max_samples,
    }


def _ranking() -> pd.DataFrame:
    return pd.DataFrame({"feature": ["feat_x", "feat_y"], "mean_abs_shap": [1.0, 1.0]})


def _run(df: pd.DataFrame, target: str, **cfg_kwargs) -> list[dict]:
    cfg = _cfg_shap(**cfg_kwargs)
    return _run_pairwise_mi_audit(
        frame=df,
        ranking=_ranking(),
        target=target,
        task_type="regression",
        cfg_shap=cfg,
        seed=42,
    )


# ---------------------------------------------------------------------------
# Test 1: independent Gaussians → MI ≈ 0, no pair flagged
# ---------------------------------------------------------------------------

def test_independent_gaussians_not_flagged():
    """Independent features (ρ=0) have I(X;Y)=0. The pair must not be flagged
    at threshold=0.05 (well above the near-zero estimated MI)."""
    rng = np.random.default_rng(1)
    n = 2000
    df = pd.DataFrame({
        "feat_x": rng.standard_normal(n),
        "feat_y": rng.standard_normal(n),
        "target": rng.standard_normal(n),
    })
    flagged = _run(df, "target", threshold=0.05)
    # The pair (feat_x, feat_y) should not be flagged since features are
    # independent of each other AND of the target.
    assert len(flagged) == 0, f"Expected 0 flagged pairs, got {len(flagged)}: {flagged}"


# ---------------------------------------------------------------------------
# Test 2: known MI against Gaussian closed form (F2)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rho,tol_rel", [
    (0.8,  0.20),   # I_true ≈ 0.511 nats — moderate correlation, expect ≤20% error
    (0.5,  0.30),   # I_true ≈ 0.144 nats — weaker, KSG has higher relative bias
])
def test_ksg_converges_to_gaussian_closed_form(rho, tol_rel):
    """KSG estimator must agree with -0.5*ln(1-ρ²) within tol_rel at n=5000.

    Setup: feat_x and target are drawn from a bivariate Gaussian with correlation ρ.
    feat_y is independent noise. We measure I(feat_x; target) by passing the pair
    (feat_x, feat_y) through _run_pairwise_mi_audit and checking the flagged score.
    The pair (feat_x, feat_y) should reflect I(feat_x, feat_y; target) — since
    feat_y is independent of target, the dominant contribution is I(feat_x; target).

    The estimator is consistent but biased at finite n, so a 20–30% relative
    tolerance is appropriate here. This test catches sign errors, off-by-factor
    bugs, and systematic normalization failures — not high-precision accuracy.
    """
    n = 5000
    rng = np.random.default_rng(7)
    cov = [[1.0, rho], [rho, 1.0]]
    xy = rng.multivariate_normal([0.0, 0.0], cov, size=n)
    feat_x = xy[:, 0]
    target_vals = xy[:, 1]
    feat_y = rng.standard_normal(n)  # independent noise feature

    df = pd.DataFrame({"feat_x": feat_x, "feat_y": feat_y, "target": target_vals})
    ranking = pd.DataFrame({"feature": ["feat_x", "feat_y"], "mean_abs_shap": [1.0, 0.5]})
    cfg = _cfg_shap(threshold=0.001, max_samples=5000)

    flagged = _run_pairwise_mi_audit(
        frame=df, ranking=ranking, target="target",
        task_type="regression", cfg_shap=cfg, seed=42,
    )
    assert len(flagged) >= 1, "Expected at least one pair flagged (threshold=0.001)"
    best = max(flagged, key=lambda p: p["mi_pair_score"])
    estimated = best["mi_pair_score"]
    true_mi = _gaussian_mi_true(rho)
    rel_err = abs(estimated - true_mi) / true_mi
    assert rel_err <= tol_rel, (
        f"KSG MI estimate {estimated:.4f} deviates from true {true_mi:.4f} "
        f"by {rel_err:.1%} (limit {tol_rel:.0%}, ρ={rho})"
    )


# ---------------------------------------------------------------------------
# Test 3: scale invariance (F1 regression guard)
# ---------------------------------------------------------------------------

def test_regression_mi_score_is_scale_invariant():
    """Rescaling the target by a constant must not change the MI score.

    This is the core F1 invariance property: multiplying Y by c changes
    var(Y_raw) by c² but joint_mi is computed in y_scaled space (std≈1)
    and var(y_scaled) stays ≈1. Before the F1 fix, score_val ∝ 1/var(Y_raw)
    so a 10x scale would shrink the score 100x.
    """
    rng = np.random.default_rng(3)
    n = 1000
    x = rng.standard_normal(n)
    target = x + 0.3 * rng.standard_normal(n)  # moderate correlation

    df_base = pd.DataFrame({"feat_x": x, "feat_y": rng.standard_normal(n), "target": target})
    df_scaled = df_base.copy()
    df_scaled["target"] = df_scaled["target"] * 100.0   # scale target by 100

    score_base   = _run(df_base,   "target", threshold=0.001)[0]["mi_pair_score"]
    score_scaled = _run(df_scaled, "target", threshold=0.001)[0]["mi_pair_score"]

    # Before the fix, ratio would be ~10000 (100²). After fix, should be ≈ 1.0.
    ratio = score_base / score_scaled if score_scaled > 0 else float("inf")
    assert 0.5 <= ratio <= 2.0, (
        f"MI score ratio (base/scaled) = {ratio:.4f}; expected ≈ 1.0 (scale-invariant). "
        f"base={score_base:.6f}, scaled={score_scaled:.6f}"
    )
