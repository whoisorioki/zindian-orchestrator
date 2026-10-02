import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from zindian.skills._lightgbm_shared import train_lightgbm_cv


def _kf(n_splits, seed):
    """Explicit KFold so these tests never depend on the live competition.

    With cv=None the trainer reads the live config, whose strategy is
    BufferedSpatialCV -- a strategy it cannot construct and must not silently
    replace (SoT S-4). These tests are about seed discipline, so passing the
    splitter explicitly keeps their intent and removes that dependency.
    """
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


def make_dummy_data(n=100, features=5, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, features)
    y = (X[:, 0] + X[:, 1] > 0.0).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(features)])
    df["target"] = y
    test_df = pd.DataFrame(
        rng.randn(int(n / 5), features), columns=[f"f{i}" for i in range(features)]
    )
    return df, test_df


def test_train_lightgbm_cv_deterministic_with_seed():
    train, test = make_dummy_data(n=80, features=4, seed=1)
    feature_cols = [c for c in train.columns if c != "target"]

    res1 = train_lightgbm_cv(
        train, test, feature_cols, "target", n_splits=4, random_seed=42,
        cv=_kf(4, 42),
    )
    res2 = train_lightgbm_cv(
        train, test, feature_cols, "target", n_splits=4, random_seed=42,
        cv=_kf(4, 42),
    )

    assert np.allclose(res1.oof_probs, res2.oof_probs)
    assert np.allclose(res1.test_probs, res2.test_probs)


def test_train_lightgbm_cv_varies_with_different_seed():
    train, test = make_dummy_data(n=80, features=4, seed=2)
    feature_cols = [c for c in train.columns if c != "target"]

    res1 = train_lightgbm_cv(
        train, test, feature_cols, "target", n_splits=4, random_seed=1,
        cv=_kf(4, 1),
    )
    res2 = train_lightgbm_cv(
        train, test, feature_cols, "target", n_splits=4, random_seed=7,
        cv=_kf(4, 7),
    )

    # Different seeds may produce different models; expect outputs not identical.
    assert not np.allclose(res1.oof_probs, res2.oof_probs)
