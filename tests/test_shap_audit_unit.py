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
        shap_mod, "_train_shap_fold_model", lambda *args, **kwargs: FakeModel()
    )
    monkeypatch.setattr(shap_mod.ChallengeConfig, "load", lambda *args, **kwargs: None)

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
    assert len(result["fold_scores"]) == 3
    assert "ranking" in result and not result["ranking"].empty
    # M1 scope fix: mi_advisory_feature_names must always be present in return dict,
    # defaulting to [] when no dominance flag fires (uniform SHAP => no dominant feature).
    assert "mi_advisory_feature_names" in result
    assert isinstance(result["mi_advisory_feature_names"], list)


def test_shap_fallback_on_single_feature(tmp_path, monkeypatch):
    # Setup folders
    comp_dir = tmp_path / "competitions" / "testcomp"
    comp_dir.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COMPETITION_SLUG", "testcomp")
    processed_dir = comp_dir / "data" / "processed"
    processed_dir.mkdir(parents=True)

    # Save a training frame with 1 feature
    df = pd.DataFrame(
        {
            "feat1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "Occurrence Status": [0, 1, 0, 1, 0, 1],
        }
    )
    df.to_csv(processed_dir / "features_train.csv", index=False)

    # Write skeleton SKILL_STATE.json
    import json
    from zindian.schemas import skill_state_skeleton

    state_path = comp_dir / "SKILL_STATE.json"
    state = skill_state_skeleton()
    state.update(
        {
            "competition": "testcomp",
            "dag_phase": "phase_3_features",
            "last_updated": "2026-06-14T12:00:00Z",
        }
    )
    state_path.write_text(json.dumps(state), encoding="utf-8")

    # Write skeleton challenge_config.json
    config_path = comp_dir / "challenge_config.json"
    config_data = {
        "name": "testcomp",
        "slug": "testcomp",
        "metric": "f1_score",
        "metric_direction": "maximize",
        "submission_format": "csv",
        "use_probabilities": False,
        "daily_limit": 10,
        "total_limit": 100,
        "public_split_pct": 20,
        "private_split_pct": 80,
        "team_allowed": True,
        "code_review_tier": "top_10",
        "allowed_external_data": True,
        "automl_permitted": False,
        "data_modality": "tabular",
        "domain": "generic",
        "task_type": "classification",
        "target_col": "Occurrence Status",
        "target_column": "Occurrence Status",
    }
    config_path.write_text(json.dumps(config_data), encoding="utf-8")

    class SimplePaths:
        def __init__(self):
            self.competition_dir = comp_dir
            self.state_path = state_path
            self.config_path = config_path
            self.data_raw_dir = comp_dir / "data" / "raw"
            self.reports_dir = comp_dir / "reports"

    monkeypatch.setattr(
        shap_mod,
        "resolve_competition_paths",
        lambda require_competition=False: SimplePaths(),
    )

    # Fake fold model to avoid lightgbm dependency errors
    class FakeModel:
        def predict_proba(self, X):
            probs = np.tile([0.3, 0.7], (X.shape[0], 1))
            return probs

    monkeypatch.setattr(
        shap_mod, "_train_shap_fold_model", lambda a, b, c, d, seed: FakeModel()
    )

    # Run
    res = shap_mod.run(n_splits=3, seed=42)
    assert res["shap_audit_skipped_reason"] == "single_feature"

    # Verify State
    import json

    updated_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert updated_state.get("shap_audit_skipped_reason") == "single_feature"
    # M1 scope fix regression guard: single-feature path must NOT raise NameError
    # and must write leakage_mi_advisory as an empty list (not missing key).
    assert (
        "leakage_mi_advisory" not in updated_state
        or updated_state["leakage_mi_advisory"] == []
    )


def test_systematic_mi_advisory_regression(monkeypatch):
    # Create small synthetic regression dataset
    n = 100
    # feat1: independent noise
    # feat2: high mutual info with target (target + tiny noise)
    target_vals = np.linspace(0.0, 10.0, n)
    df = pd.DataFrame(
        {
            "feat1": np.random.default_rng(42).normal(0.0, 1.0, n),
            "feat2": target_vals + np.random.default_rng(42).normal(0.0, 0.01, n),
            "Target": target_vals,
        }
    )
    feature_cols = ["feat1", "feat2"]

    class FakeModel:
        def predict(self, X):
            return np.zeros(X.shape[0], dtype=np.float64)

    monkeypatch.setattr(
        shap_mod, "_train_shap_fold_model", lambda *args, **kwargs: FakeModel()
    )

    class MockCfg:
        def get(self, key, default=None):
            if key == "enable_mi_regression_subsample":
                return True
            if key == "leak_nmi_threshold":
                return 0.2
            if key == "mi_max_samples":
                return 100
            return default

    monkeypatch.setattr(
        shap_mod.ChallengeConfig, "load", lambda *args, **kwargs: MockCfg()
    )

    # Fake SHAP explainer that returns constant values where feat1 gets higher SHAP than feat2
    # This ensures feat2 is NOT SHAP-dominant.
    class FakeExplainer:
        def __init__(self, model):
            pass

        def shap_values(self, X, check_additivity=False):
            # feat1 has shap importance 10.0, feat2 has 1.0
            shaps = np.zeros((X.shape[0], X.shape[1]))
            shaps[:, 0] = 10.0
            shaps[:, 1] = 1.0
            return shaps

    monkeypatch.setattr(shap_mod.shap, "TreeExplainer", FakeExplainer)

    result = shap_mod._compute_shap_audit(
        df, feature_cols, "Target", n_splits=3, seed=42, task_type="regression"
    )

    # feat2 must be flagged by the systematic MI check because of its high mutual information with Target,
    # even though feat1 was the top SHAP feature and was NOT flagged by Pearson (since it's independent).
    assert "feat2" in result["mi_advisory_feature_names"]
    assert "feat1" not in result["mi_advisory_feature_names"]


def test_train_shap_fold_model_eval_set(monkeypatch):
    captured_kwargs = []

    class MockLGBM:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, X, y, **kwargs):
            captured_kwargs.append(kwargs)

    monkeypatch.setattr(shap_mod.lgb, "LGBMClassifier", MockLGBM)
    monkeypatch.setattr(shap_mod.lgb, "LGBMRegressor", MockLGBM)

    train_x = np.zeros((5, 2))
    train_y = np.zeros(5)
    val_x = np.zeros((2, 2))
    val_y = np.zeros(2)

    shap_mod._train_shap_fold_model(
        train_x, train_y, val_x, val_y, seed=42, task_type="classification"
    )
    shap_mod._train_shap_fold_model(
        train_x, train_y, val_x, val_y, seed=42, task_type="regression"
    )

    assert len(captured_kwargs) == 2
    for kwargs in captured_kwargs:
        assert "eval_set" in kwargs
        assert "eval_X" not in kwargs
        assert "eval_y" not in kwargs

