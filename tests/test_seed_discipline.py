import numpy as np
import pandas as pd
from zindian.skills._lightgbm_shared import train_lightgbm_cv


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
        train, test, feature_cols, "target", n_splits=4, random_seed=42
    )
    res2 = train_lightgbm_cv(
        train, test, feature_cols, "target", n_splits=4, random_seed=42
    )

    assert np.allclose(res1.oof_probs, res2.oof_probs)
    assert np.allclose(res1.test_probs, res2.test_probs)


def test_train_lightgbm_cv_varies_with_different_seed():
    train, test = make_dummy_data(n=80, features=4, seed=2)
    feature_cols = [c for c in train.columns if c != "target"]

    res1 = train_lightgbm_cv(
        train, test, feature_cols, "target", n_splits=4, random_seed=1
    )
    res2 = train_lightgbm_cv(
        train, test, feature_cols, "target", n_splits=4, random_seed=7
    )

    # Different seeds may produce different models; expect outputs not identical.
    assert not np.allclose(res1.oof_probs, res2.oof_probs)
