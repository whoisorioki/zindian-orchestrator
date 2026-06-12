import numpy as np
import pandas as pd

from zindian.skills import skill_10_shap as shap_mod


def test_compute_shap_audit_monkeypatch(monkeypatch):
    # Create small synthetic dataset
    n = 12
    df = pd.DataFrame(
        {
            "feat1": np.linspace(0.0, 1.0, n),
            "feat2": np.linspace(1.0, 2.0, n),
            "feat3": np.linspace(2.0, 3.0, n),
            "Occurrence Status": [0, 1] * (n // 2),
        }
    )
    feature_cols = ["feat1", "feat2", "feat3"]

    # Fake fold model that returns deterministic probabilities
    class FakeModel:
        def predict_proba(self, X):
            probs = np.tile([0.3, 0.7], (X.shape[0], 1))
            return probs

    monkeypatch.setattr(
        shap_mod, "_train_shap_fold_model", lambda a, b, c, d, seed: FakeModel()
    )

    # Fake SHAP explainer that returns constant positive values
    class FakeExplainer:
        def __init__(self, model):
            pass

        def shap_values(self, X, check_additivity=False):
            return np.ones((X.shape[0], X.shape[1]))

    monkeypatch.setattr(shap_mod.shap, "TreeExplainer", FakeExplainer)

    result = shap_mod._compute_shap_audit(
        df, feature_cols, "Occurrence Status", n_splits=3, seed=42
    )

    assert "oof_probs" in result and len(result["oof_probs"]) == len(df)
    assert len(result["fold_aucs"]) == 3
    assert "ranking" in result and not result["ranking"].empty
