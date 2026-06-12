import numpy as np
from scipy.stats import pearsonr, spearmanr
from zindian.oracle_fusion_core import _correlation


def test_pearson_for_classification():
    # Classification uses Pearson correlation (standard linear correlation)
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([2, 4, 6, 8, 10], dtype=float)  # Perfect linear relationship

    corr_class = _correlation(x, y, "classification")
    expected_pearson, _ = pearsonr(x, y)

    assert np.isclose(corr_class, expected_pearson)


def test_spearman_for_regression():
    # Regression uses Spearman rank correlation
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    # Non-linear monotonic relationship (Pearson will be different, Spearman will be 1.0)
    y = np.array([1, 10, 100, 1000, 10000], dtype=float)

    corr_reg = _correlation(x, y, "regression")
    expected_spearman, _ = spearmanr(x, y)

    assert np.isclose(corr_reg, expected_spearman)
    assert np.isclose(corr_reg, 1.0)
