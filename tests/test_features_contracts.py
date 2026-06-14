import pandas as pd
import numpy as np

from zindian.skills.skill_07_features import build_hypothesis_features


def test_build_hypothesis_features_adds_columns():
    # minimal train/test frames with required base columns
    cols = [
        "tmax_mean",
        "aet_mean",
        "pet_mean",
        "vpd_mean",
        "tmin_min",
        "ppt_mean",
        "tmin_mean",
        "tmin_std",
        "tmin_max",
        "soil_mean",
    ]
    n = 4
    train = pd.DataFrame({c: np.linspace(0.1, 1.0, n) for c in cols})
    test = pd.DataFrame({c: np.linspace(1.0, 2.0, n) for c in cols})

    tr, te = build_hypothesis_features(train.copy(), test.copy(), mode="inference")

    expected = [
        "tmax_mean_sq",
        "aet_pet_ratio",
        "tmax_vpd_stress",
        "frost_risk",
        "aridity_index",
        "warm_wet_index",
    ]
    for col in expected:
        assert col in tr.columns
        assert col in te.columns

def test_target_dependent_feature_modes():
    cols = [
        "tmax_mean",
        "aet_mean",
        "pet_mean",
        "vpd_mean",
        "tmin_min",
        "ppt_mean",
        "tmin_mean",
        "tmin_std",
        "tmin_max",
        "soil_mean",
    ]
    n = 4
    train = pd.DataFrame({c: [0.1, 0.2, 0.3, 0.4] for c in cols})
    test = pd.DataFrame({c: [0.5, 0.6] for c in cols})
    target_array = np.array([1, 0, 1, 0])

    # 1. Missing train_idx in mode="cv" raises ValueError
    import pytest
    with pytest.raises(ValueError, match="train_idx must be provided"):
        build_hypothesis_features(
            train, test, mode="cv", target_array=target_array, train_idx=None
        )

    # 2. Correct computation in mode="cv" with train_idx
    train_idx = np.array([0, 1])
    tr_cv, te_cv = build_hypothesis_features(
        train, test, mode="cv", target_array=target_array, train_idx=train_idx
    )
    assert "tmin_bin_target_mean" in tr_cv.columns
    assert "tmin_bin_target_mean" in te_cv.columns

    # 3. Correct computation in mode="inference"
    tr_inf, te_inf = build_hypothesis_features(
        train, test, mode="inference", target_array=target_array
    )
    assert "tmin_bin_target_mean" in tr_inf.columns
    assert "tmin_bin_target_mean" in te_inf.columns

