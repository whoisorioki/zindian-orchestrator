import pandas as pd
import numpy as np

from zindian.skills.skill_07_features import build_hypothesis_features


def test_build_hypothesis_features_adds_columns():
    # minimal train/test frames with required base columns
    cols = [
        "tmax_mean", "aet_mean", "pet_mean", "vpd_mean",
        "tmin_min", "ppt_mean", "tmin_mean", "tmin_std", "tmin_max",
        "soil_mean",
    ]
    n = 4
    train = pd.DataFrame({c: np.linspace(0.1, 1.0, n) for c in cols})
    test = pd.DataFrame({c: np.linspace(1.0, 2.0, n) for c in cols})

    tr, te = build_hypothesis_features(train.copy(), test.copy(), mode="inference")

    expected = [
        "tmax_mean_sq", "aet_pet_ratio", "tmax_vpd_stress",
        "frost_risk", "aridity_index", "warm_wet_index",
    ]
    for col in expected:
        assert col in tr.columns
        assert col in te.columns
