import numpy as np
from scipy.stats import pearsonr, spearmanr
from typing import cast
from zindian.oracle_fusion_core import _correlation


def test_pearson_for_classification():
    # Classification uses Pearson correlation (standard linear correlation)
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([2, 4, 6, 8, 10], dtype=float)  # Perfect linear relationship

    corr_class = _correlation(x, y, "classification")
    expected_pearson = cast(tuple[float, float], pearsonr(x, y))[0]

    assert np.isclose(corr_class, expected_pearson)


def test_spearman_for_regression():
    # Regression uses Spearman rank correlation
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    # Non-linear monotonic relationship (Pearson will be different, Spearman will be 1.0)
    y = np.array([1, 10, 100, 1000, 10000], dtype=float)

    corr_reg = _correlation(x, y, "regression")
    expected_spearman = cast(tuple[float, float], spearmanr(x, y))[0]

    assert np.isclose(corr_reg, expected_spearman)
    assert np.isclose(corr_reg, 1.0)


def test_prune_collinear_residuals():
    from zindian.oracle_fusion_core import _prune_collinear

    # 2 candidates with highly correlated predictions but different error residuals when y_true is active
    # y_true is [1.0, 1.0, 1.0]
    # pred_a = [1.1, 1.2, 1.3] -> residual_a = [0.1, 0.2, 0.3]
    # pred_b = [1.2, 1.1, 1.3] -> residual_b = [0.2, 0.1, 0.3]
    # Let's verify _prune_collinear behavior
    candidates = [
        {"name": "cand_a", "score": 0.8, "probs": np.array([1.1, 1.2, 1.3])},
        {"name": "cand_b", "score": 0.75, "probs": np.array([1.2, 1.1, 1.3])},
    ]

    # Without y_true: raw Pearson correlation between [1.1, 1.2, 1.3] and [1.2, 1.1, 1.3]
    # mean_a = 1.2, dev_a = [-0.1, 0.0, 0.1]
    # mean_b = 1.2, dev_b = [0.0, -0.1, 0.1]
    # corr = 0.5 (below 0.95), neither is dropped
    pruned, dropped = _prune_collinear(
        candidates, task_type="classification", direction="maximize", y_true=None
    )
    assert len(pruned) == 2
    assert len(dropped) == 0

    # Let's construct a pair that IS collinear on raw predictions (> 0.95),
    # but when y_true is subtracted, their residuals are NOT collinear.
    # y_true: [0.0, 0.0, 0.0] -> residuals = raw. Let's make raw collinear.
    # pred_a = [1.0, 2.0, 3.0]
    # pred_b = [1.1, 2.1, 3.1] (perfectly correlated raw: corr = 1.0)
    # y_true = [1.0, 3.0, 2.0]
    # residual_a = [0.0, -1.0, 1.0]
    # residual_b = [0.1, -0.9, 1.1] -> wait, these are still perfectly correlated because b = a + 0.1
    # Let's make residuals different:
    # y_true = [1.0, 1.0, 2.0]
    # residual_a = [0.0, 1.0, 1.0]
    # residual_b = [0.1, 1.1, 1.1] -> still perfectly correlated.
    # To make residuals non-correlated while raw is highly correlated, we need:
    # raw_a = [1.0, 2.0, 3.0]
    # raw_b = [1.05, 2.05, 3.05] (perfectly correlated, r = 1.0)
    # If we subtract y_true = [1.0, 1.9, 3.2]
    # residual_a = [0.0, 0.1, -0.2]
    # residual_b = [0.05, 0.15, -0.15] -> b = a + 0.05 (still perfectly correlated).
    # Ah! Subtracting the same constant vector y_true preserves Pearson/Spearman correlation!
    # Wait, correlation is shift-invariant (corr(X - C, Y - C) == corr(X, Y) is NOT true!
    # Let's check: Cov(X - C, Y - C) = Cov(X, Y).
    # Var(X - C) = Var(X), Var(Y - C) = Var(Y).
    # So Pearson correlation of residuals is EXACTLY EQUAL to Pearson correlation of raw values!
    # Wait, is that true?
    # Yes! Pearson correlation of (X - Z) and (Y - Z) where Z is a vector (y_true) is NOT equal to corr(X, Y).
    # Because Z is a vector, not a scalar constant! Z varies across samples.
    # Yes, Z (y_true) varies! E.g. y_true = [1.0, 5.0, -2.0].
    # So corr(X - Z, Y - Z) != corr(X, Y).
    # Let's verify this mathematically:
    # X = [1.0, 2.0, 3.0], Y = [1.1, 2.2, 3.3] -> corr(X, Y) = 1.0
    # Z = [1.0, 5.0, 10.0]
    # X - Z = [0.0, -3.0, -7.0]
    # Y - Z = [0.1, -2.8, -6.7]
    # Let's compute their correlation:
    # X - Z: mean = -3.333, dev = [3.333, 0.333, -3.667]
    # Y - Z: mean = -3.133, dev = [3.233, 0.333, -3.567]
    # Wait, these are still extremely close to 1.0 because Y is just 1.1 * X, and they both have Z subtracted.
    # What if X and Y are highly correlated, but their errors are independent?
    # e.g., X = Z + error_a, Y = Z + error_b.
    # Since Z has large variance, X and Y will be highly correlated due to sharing the target Z.
    # But X - Z = error_a, and Y - Z = error_b.
    # If error_a and error_b are independent, the residuals (errors) will have 0 correlation!
    # This is exactly the core concept of residual diversity / Kuncheva pruning!
    # Excellent! Let's write the test:
    Z = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    error_a = np.array([0.1, -0.1, 0.2, -0.2, 0.0])
    error_b = np.array([-0.2, 0.2, -0.1, 0.1, 0.0])
    # raw predictions:
    X = Z + error_a
    Y = Z + error_b

    # Without y_true, X and Y are dominated by Z (variance of Z is ~250, variance of errors is ~0.05)
    # So corr(X, Y) is extremely close to 1.0.
    candidates = [
        {"name": "cand_a", "score": 0.8, "probs": X},
        {"name": "cand_b", "score": 0.75, "probs": Y},
    ]

    # Without y_true, they should be pruned (dropped) because corr > 0.95
    pruned_no_y, dropped_no_y = _prune_collinear(
        candidates, task_type="classification", direction="maximize", y_true=None
    )
    assert len(pruned_no_y) == 1
    assert len(dropped_no_y) == 1
    assert dropped_no_y[0]["dropped"] == "cand_b"

    # With y_true = Z, residuals are error_a and error_b, which are negatively correlated, so they shouldn't be pruned!
    pruned_y, dropped_y = _prune_collinear(
        candidates, task_type="classification", direction="maximize", y_true=Z
    )
    assert len(pruned_y) == 2
    assert len(dropped_y) == 0


def test_prune_collinear_residuals_inverse():
    from zindian.oracle_fusion_core import _prune_collinear

    # Inverse case: low raw correlation, but high residual correlation.
    # U = residual_a, V = residual_b. corr(U, V) > 0.95 (should be pruned when y_true is active).
    # Z = y_true. X = Z + U, Y = Z + V. corr(X, Y) < 0.95 (should NOT be pruned when y_true is None).
    U = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    V = np.array([1.0, 2.0, 3.0, 4.0, 6.0])  # corr(U, V) = 0.986
    Z = -1.05 * U
    X = Z + U
    Y = Z + V

    candidates = [
        {"name": "cand_a", "score": 0.8, "probs": X},
        {"name": "cand_b", "score": 0.75, "probs": Y},
    ]

    # Without y_true: raw predictions have correlation -0.6, so they should NOT be pruned.
    pruned_no_y, dropped_no_y = _prune_collinear(
        candidates, task_type="classification", direction="maximize", y_true=None
    )
    assert len(pruned_no_y) == 2
    assert len(dropped_no_y) == 0

    # With y_true = Z: residuals have correlation 0.986 > 0.95, so they SHOULD be pruned!
    pruned_y, dropped_y = _prune_collinear(
        candidates, task_type="classification", direction="maximize", y_true=Z
    )
    assert len(pruned_y) == 1
    assert len(dropped_y) == 1
    assert dropped_y[0]["dropped"] == "cand_b"


def test_prune_collinear_regression_residual_spearman():
    """T2: regression residual pruning must use Spearman rank correlation.

    Branches sharing the true signal Z have raw Pearson ~1.0 and are pruned by
    the shared-Signal collinearity when y_true is absent; once the target is
    subtracted their residuals are independent and must NOT be pruned. In the
    inverse direction, residuals correlated in rank (Spearman) must be pruned
    even when raw Pearson is low — proving _prune_collinear delegates to the
    Spearman branch for task_type="regression" (per AGENTS.md correlation note).
    """
    from zindian.oracle_fusion_core import _prune_collinear

    # Residual-diversity case: Z dominates predictions, errors are independent.
    Z = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    e_a = np.array([0.1, -0.1, 0.2, -0.2, 0.0])
    e_b = np.array([-0.2, 0.2, -0.1, 0.1, 0.0])
    X = Z + e_a
    Y = Z + e_b
    cands = [
        {"name": "cand_a", "score": 0.8, "probs": X},
        {"name": "cand_b", "score": 0.75, "probs": Y},
    ]

    # Without y_true, X and Y are dominated by Z (corr ~1.0) -> prune lower scorer.
    pruned, dropped = _prune_collinear(
        cands, task_type="regression", direction="maximize", y_true=None
    )
    assert len(dropped) == 1
    assert dropped[0]["dropped"] == "cand_b"

    # With y_true = Z, residuals are e_a/e_b (independent) -> NOT pruned.
    pruned_y, dropped_y = _prune_collinear(
        cands, task_type="regression", direction="maximize", y_true=Z
    )
    assert len(pruned_y) == 2
    assert len(dropped_y) == 0

    # Inverse: rank-correlated residuals (monotone, non-linear) must be pruned
    # via Spearman even when the residual Pearson correlation is below the
    # 0.95 threshold. r_b = r_a^3 preserves rank (Spearman = 1) but is modelled
    # by the cube transform, so the linear residual Pearson drops below 0.95.
    # This discriminates the regression Spearman branch from a naive Pearson one.
    r_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    r_b = np.power(r_a, 3)  # 1,8,27,64,125 -> Spearman=1, Pearson rho < 0.95
    Z2 = -r_a  # shared target
    cands2 = [
        {"name": "cand_a", "score": 0.8, "probs": Z2 + r_a},          # residual = r_a
        {"name": "cand_b", "score": 0.75, "probs": Z2 + (r_b - 3.0)},  # residual = r_b-3
    ]
    pruned2, dropped2 = _prune_collinear(
        cands2, task_type="regression", direction="maximize", y_true=Z2
    )
    assert len(dropped2) == 1
    assert dropped2[0]["dropped"] == "cand_b"
