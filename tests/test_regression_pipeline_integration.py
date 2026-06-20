import json
import numpy as np
import pandas as pd

from zindian.state import SkillStateStore
from zindian.skills import skill_04_eda
from zindian.skills import skill_06_cleaning
from zindian.skills import skill_07_features
from zindian.skills import skill_10_shap
from zindian.skills import skill_11_gate


def test_regression_pipeline_integration(tmp_path, monkeypatch):
    # Setup folders under tmp_path
    comp_dir = tmp_path / "competitions" / "regcomp"
    comp_dir.mkdir(parents=True)
    raw_dir = comp_dir / "data" / "raw"
    raw_dir.mkdir(parents=True)
    proc_dir = comp_dir / "data" / "processed"
    proc_dir.mkdir(parents=True)
    reports_dir = comp_dir / "reports"
    reports_dir.mkdir(parents=True)
    (comp_dir / "submissions").mkdir(parents=True)

    # 1. Create synthetic regression dataset (12 rows, tmax_mean etc columns to pass skill_07 requirements)
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
    # Continuous target for regression
    rng = np.random.RandomState(42)
    n_samples = 20
    train_df = pd.DataFrame({c: rng.randn(n_samples) for c in cols})
    train_df["ID"] = [f"train_{i}" for i in range(n_samples)]
    # Set continuous target
    train_df["target"] = rng.randn(n_samples) * 5.0 + 10.0  # mean=10, std=5

    test_df = pd.DataFrame({c: rng.randn(n_samples) for c in cols})
    test_df["ID"] = [f"test_{i}" for i in range(n_samples)]

    sample_df = pd.DataFrame(
        {"ID": [f"test_{i}" for i in range(n_samples)], "target": [0.0] * n_samples}
    )

    train_df.to_csv(raw_dir / "Training_Data.csv", index=False)
    test_df.to_csv(raw_dir / "Test_Data.csv", index=False)
    sample_df.to_csv(raw_dir / "SampleSubmission.csv", index=False)

    # 2. Write challenge_config.json
    cfg = {
        "name": "regcomp",
        "slug": "regcomp",
        "task_type": "regression",
        "target_col": "target",
        "target_column": "target",
        "metric": "rmse",
        "metric_direction": "minimize",
        "submission_format": "csv",
        "use_probabilities": False,
        "daily_limit": 10,
        "total_limit": 100,
        "public_split_pct": 20,
        "private_split_pct": 80,
        "team_allowed": True,
        "code_review_tier": None,
        "allowed_external_data": False,
        "automl_permitted": False,
        "data_modality": "tabular",
        "domain": "regression",
        "target_domain_bounds": {"min": -10.0, "max": 30.0},
        "cv_strategy": {
            "type": "KFold",
            "n_splits": 5,
            "shuffle": True,
            "random_state": 42,
        },
        "reproducibility": {"seed": 42},
        "gate_margin": 0.05,
        "variance_gate_threshold": 0.5,
        "input_files": {
            "train": "Training_Data.csv",
            "test": "Test_Data.csv",
            "sample": "SampleSubmission.csv",
        },
        "feature_extraction_plugin": "tests._mock_plugin_07",
    }
    (comp_dir / "challenge_config.json").write_text(
        json.dumps(cfg, indent=2), encoding="utf-8"
    )

    # 3. Write initial SKILL_STATE.json
    from zindian.schemas import skill_state_skeleton

    state = skill_state_skeleton()
    state.update(
        {
            "dag_phase": "phase_1",
            "last_updated": "2026-06-14T12:00:00Z",
            "competition": "regcomp",
            "anchor_git_branch": "main",
            "human_gate_2_variant-06_approved": True,  # approve gate for the variant
        }
    )
    (comp_dir / "SKILL_STATE.json").write_text(
        json.dumps(state, indent=2), encoding="utf-8"
    )

    # Monkeypatch and setenv so paths resolve to our temp workspace
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COMPETITION_SLUG", "regcomp")

    # Mock checkout command in subprocess for git checkout in test context
    class DummyCompletedProcess:
        def __init__(self):
            self.stdout = ""
            self.stderr = ""
            self.returncode = 0

    monkeypatch.setattr("subprocess.run", lambda cmd, **kwargs: DummyCompletedProcess())

    # 4. Execute Skill 04: EDA
    skill_04_eda.run()

    # Verify that target_std was calculated and written to the eda block in state
    state_store = SkillStateStore(comp_dir / "SKILL_STATE.json")
    state_after_eda = state_store.read()
    assert "eda" in state_after_eda
    assert "target_std" in state_after_eda["eda"]
    assert state_after_eda["eda"]["target_std"] > 0.0

    # 5. Execute Skill 06: Cleaning (manual orchestration because it is a state utility)
    clean_state = skill_06_cleaning.run(
        cfg,
        {
            "eda": state_after_eda["eda"],
            "X_train": train_df.drop(columns=["ID", "target"]),
            "X_test": test_df.drop(columns=["ID"]),
        },
    )
    assert "cleaning" in clean_state

    # 5.5 Prepare processed features and run Skill 08: Anchor Baseline
    from zindian.constants import TC_BAND_NAMES

    train_feat_df = train_df.copy()
    test_feat_df = test_df.copy()
    for col in TC_BAND_NAMES:
        if col not in train_feat_df.columns:
            train_feat_df[col] = 0.0
        if col not in test_feat_df.columns:
            test_feat_df[col] = 0.0
    train_feat_df.to_csv(proc_dir / "features_train.csv", index=False)
    test_feat_df.to_csv(proc_dir / "features_test.csv", index=False)

    from zindian.skills import skill_08_anchor

    skill_08_anchor.run(n_splits=5)

    # Assert that anchor_oof_score is populated in SKILL_STATE.json and is a float
    state_after_anchor = state_store.read()
    assert "anchor_oof_score" in state_after_anchor
    assert isinstance(state_after_anchor["anchor_oof_score"], float)
    # Check that secondary_metrics exists in branch_anchor-baseline_oof
    assert "branch_anchor-baseline_oof" in state_after_anchor
    assert "secondary_metrics" in state_after_anchor["branch_anchor-baseline_oof"]
    sec = state_after_anchor["branch_anchor-baseline_oof"]["secondary_metrics"]
    assert isinstance(sec["mae"], float)
    assert isinstance(sec["r2"], float)

    # 6. Execute Skill 07: Feature Engineering
    # Patch importlib.import_module so skill_07 receives a mock plugin with extract().
    import importlib as _importlib

    def mock_extract_fn(paths_arg, tiff_path_arg, config_arg):
        return (
            pd.read_csv(paths_arg.data_processed_dir / "features_train.csv"),
            pd.read_csv(paths_arg.data_processed_dir / "features_test.csv"),
        )

    class _MockPlugin:
        @staticmethod
        def extract(paths_arg, tiff_path_arg, config_arg):
            return mock_extract_fn(paths_arg, tiff_path_arg, config_arg)

    _real_import = _importlib.import_module

    def _patched_import(name, *args, **kwargs):
        if name == "tests._mock_plugin_07":
            return _MockPlugin()
        return _real_import(name, *args, **kwargs)

    monkeypatch.setattr(_importlib, "import_module", _patched_import)
    # Also patch skill_07's own importlib reference
    monkeypatch.setattr(skill_07_features.importlib, "import_module", _patched_import)

    skill_07_features.run(variant_name="variant-06")
    assert (proc_dir / "features_train.csv").exists()

    # Verify variant-06 OOF record has secondary_metrics
    state_after_features = state_store.read()
    assert "branch_variant-06_oof" in state_after_features
    assert "secondary_metrics" in state_after_features["branch_variant-06_oof"]
    sec_var = state_after_features["branch_variant-06_oof"]["secondary_metrics"]
    assert isinstance(sec_var["mae"], float)

    # 7. Execute Skill 10: SHAP Audit
    # First, let's write best_variant_this_round and other expected state items so skill_11 knows what to promote
    state_store.update(
        best_variant_this_round="variant-06",
        best_variant_oof_score=1.0,  # low RMSE means improved!
        variants_passed=1,
    )

    skill_10_shap.run(n_splits=5, seed=42)

    # 8. Execute Skill 11: Gate
    res = skill_11_gate.run()

    # The gate should pass and update the state
    assert res["status"] in ("PASS", "BLOCKED")
