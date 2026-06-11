import numpy as np
import pandas as pd

from zindian.skills import skill_07_features as features


def test_train_variant_calls_shared_trainer(monkeypatch):
    # small synthetic train/test
    train = pd.DataFrame({
        "ID": [1, 2, 3, 4],
        "Occurrence Status": [0, 1, 0, 1],
        "f1": [0.1, 0.2, 0.3, 0.4],
    })
    test = pd.DataFrame({"ID": [5, 6], "f1": [0.5, 0.6]})

    class FakeResult:
        def __init__(self):
            self.oof_f1 = 0.7
            self.oof_auc = 0.85
            self.threshold = 0.5
            self.oof_probs = np.array([0.1, 0.9, 0.2, 0.8])
            self.test_probs = np.array([0.6, 0.4])

    def fake_trainer(*args, **kwargs):
        return FakeResult()

    monkeypatch.setattr(features, "train_lightgbm_cv", fake_trainer)

    res = features.train_variant(train, test, ["f1"], "variant-06", anchor_f1=0.5, seed=42)
    assert res["variant"] == "variant-06"
    assert "oof_f1" in res and res["oof_f1"] == 0.7
    assert res["gate"] in {"PASS", "PRUNE"}
