import numpy as np
from zindian.skills._lightgbm_shared import LightGBMRunResult


def test_lightgbm_runresult_schema_fields():
    # Construct a minimal plausible LightGBMRunResult
    res = LightGBMRunResult(
        oof_probs=np.array([0.1, 0.9]),
        test_probs=np.array([0.2, 0.8]),
        oof_auc=0.85,
        oof_f1=0.7,
        threshold=0.5,
        fold_aucs=[0.8, 0.9],
    )

    assert hasattr(res, "oof_probs") and isinstance(res.oof_probs, np.ndarray)
    assert hasattr(res, "test_probs") and isinstance(res.test_probs, np.ndarray)
    assert isinstance(res.oof_auc, float)
    assert isinstance(res.oof_f1, float)
    assert isinstance(res.threshold, float)
    assert isinstance(res.fold_aucs, list)
