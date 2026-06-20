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
