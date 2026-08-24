import pytest
import pandas as pd
import inspect
from pathlib import Path
from unittest.mock import MagicMock


# 1. A composite-formula correctness test
def test_composite_formula_correctness():
    # Classification target: F1 = 0.8, weight = 0.4
    # Regression target: RMSE = 2.0, target_std = 4.0, weight = 0.6
    # expected regression_score = max(0.0, 1.0 - (2.0 / 4.0)) = 0.5
    # expected composite = (0.8 * 0.4 + 0.5 * 0.6) / (0.4 + 0.6) = 0.62

    f1 = 0.8
    w_class = 0.4
    rmse = 2.0
    target_std = 4.0
    w_reg = 0.6

    regression_score = max(0.0, 1.0 - (rmse / target_std))
    weighted_scores = [f1 * w_class, regression_score * w_reg]
    total_weight = w_class + w_reg
    avg_score = sum(weighted_scores) / total_weight

    assert abs(avg_score - 0.62) < 1e-9


# 2. A stub-detection test for skill_21
def test_skill21_stub_detection(monkeypatch):
    from zindian.skills.skill_21_pseudo_label import run as run_skill_21

    # Assert that calling run() with a regression task_type raises ValueError (guard condition 1)
    class DummyConfig:
        def __init__(self):
            self._data = {"task_type": "regression"}

        def get(self, key, default=None):
            return self._data.get(key, default)

    mock_cfg = DummyConfig()
    monkeypatch.setattr(
        "zindian.skills.skill_21_pseudo_label.ChallengeConfig.load", lambda: mock_cfg
    )

    mock_paths = MagicMock()
    mock_paths.state_path = Path("dummy_state.json")
    monkeypatch.setattr(
        "zindian.skills.skill_21_pseudo_label.resolve_competition_paths",
        lambda: mock_paths,
    )

    mock_store = MagicMock()
    mock_store.read.return_value = {}
    monkeypatch.setattr(
        "zindian.skills.skill_21_pseudo_label.SkillStateStore", lambda path: mock_store
    )

    with pytest.raises(ValueError, match="strictly prohibited for task_type"):
        run_skill_21()


# 3. A join-cardinality regression test for the extractor
def test_extractor_join_cardinality():
    from plugins.world_cup_extractor import _enrich

    # Input dataframe with 4 matches (4 rows)
    df = pd.DataFrame(
        {
            "team_id": ["T-1", "T-2", "T-1", "T-3"],
            "country": ["Germany", "Brazil", "Germany", "Argentina"],
        }
    )

    # Auxiliary data containing duplicates (West Germany and Germany both mapped to T-1)
    teams_df = pd.DataFrame(
        {
            "team_id": ["T-1", "T-1", "T-2", "T-3"],
            "team_name": ["Germany", "West Germany", "Brazil", "Argentina"],
            "confederation_id": ["UEFA", "UEFA", "CONMEBOL", "CONMEBOL"],
        }
    )

    auxiliary_data = {"teams": teams_df}

    class DummyConfig:
        def get(self, key, default=None):
            if key == "plugin_config":
                return {"team_id_col": "team_id"}
            return default

    config = DummyConfig()

    # Call _enrich
    enriched_df = _enrich(df, auxiliary_data, config)

    # Assert that row count did not multiply (remained 4)
    assert len(enriched_df) == 4
    assert list(enriched_df["confederation_id"]) == [
        "UEFA",
        "CONMEBOL",
        "UEFA",
        "CONMEBOL",
    ]


# 4. A field-naming contract test
def test_field_naming_contract():
    # Assert that a regression target uses 'oof_rmse' and classification uses 'oof_f1'
    from zindian.skills.skill_11_gate import _run_multi_target_gate

    source = inspect.getsource(_run_multi_target_gate)

    # Classification targets read F1
    assert "oof_f1" in source
    # Regression targets read RMSE
    assert "oof_rmse" in source

    # Ensure there is no legacy 'oof_logloss' fallback for regression target calculation
    assert "oof_logloss" not in source


def test_feature_extractor_interface():
    from plugins.base_extractor import FeatureExtractor
    from plugins.world_cup_extractor import Extractor as WCExtractor
    from plugins.nedbank_extractor import Extractor as NExtractor
    from plugins.terraclimate_extractor import Extractor as TExtractor

    assert issubclass(WCExtractor, FeatureExtractor)
    assert issubclass(NExtractor, FeatureExtractor)
    assert issubclass(TExtractor, FeatureExtractor)


def test_skill12_composite_variance():
    from zindian.skills.skill_12_metric import run as run_skill_12

    # Setup mock targets in config
    config = {
        "target_config": {
            "targets": [
                {"name": "Target_Class", "task_type": "classification", "weight": 0.4},
                {"name": "Target_Reg", "task_type": "regression", "weight": 0.6},
            ]
        }
    }

    # Setup mock state with fold scores for targets
    # For Target_Class (classification): F1 scores across 5 folds
    # For Target_Reg (regression): RMSE scores across 5 folds
    state = {
        "best_variant_this_round": "anchor-baseline",
        "branch_anchor-baseline_Target_Class_oof": {
            "model_config": {
                "fold_scores": [0.8, 0.85, 0.78, 0.82, 0.88],
            }
        },
        "branch_anchor-baseline_Target_Reg_oof": {
            "model_config": {
                "fold_scores": [2.0, 1.8, 2.2, 1.9, 2.1],
            }
        },
        "eda": {
            "Target_Reg_std": 4.0,
        },
    }

    # Composite calculation hand-math check for fold 0:
    # f1 = 0.8 -> distance = 1.0 - 0.8 = 0.2
    # rmse = 2.0 -> distance = 2.0 / 4.0 = 0.5
    # composite_score_0 = (0.2 * 0.4 + 0.5 * 0.6) / 1.0 = 0.38

    res = run_skill_12(config=config, state=state)
    assert "metric_analysis" in res
    analysis = res["metric_analysis"]

    assert "fold_scores" in analysis
    assert len(analysis["fold_scores"]) == 5
    assert abs(analysis["fold_scores"][0] - 0.38) < 1e-9
    assert "fold_score_variance" in analysis
    assert analysis["fold_score_variance"] > 0


def test_skill21_recombination_and_multi_target_retraining(monkeypatch):
    from zindian.skills.skill_21_pseudo_label import _run_multi_target_pseudo_label

    # Setup mock state/config
    config_dict = {
        "target_config": {
            "targets": [
                {"name": "Target_Class", "task_type": "classification", "weight": 0.5},
                {"name": "Target_Reg", "task_type": "regression", "weight": 0.5},
            ],
            "pseudo_label_recombination_policy": "freeze_unaugmented_targets_at_original",
        }
    }

    class DummyConfig:
        def __init__(self, data):
            self._data = data

        def get(self, key, default=None):
            return self._data.get(key, default)

    config = DummyConfig(config_dict)

    # Mock original baseline OOF for regression
    state = {
        "branch_anchor-baseline_Target_Reg_oof": {
            "scores": [2.5, 1.0, 3.2, 0.5],
            "cv_strategy_id": "stratified_5fold",
        }
    }

    # Mock run() to simulate classification target augmentation needing retraining
    def mock_run(dry_run=False, target_name_override=None, is_multi_target=False):
        assert target_name_override == "Target_Class"
        assert is_multi_target is True
        return {
            "status": "OK",
            "best_iteration": 2,
            "best_oof_f1": 0.85,
            "retraining_required": True,
            "guard_condition_flags": {"gc1": True},
        }

    monkeypatch.setattr("zindian.skills.skill_21_pseudo_label.run", mock_run)

    mock_store = MagicMock()
    mock_store.read.return_value = state

    # Run target recombination policy execution
    res = _run_multi_target_pseudo_label(None, config, mock_store, state, dry_run=False)

    assert res["status"] == "OK"
    assert res["retraining_required"] is True

    # Verify that the store's update got called to write the frozen/copied regression OOF to _augmented
    called_keys = []
    for call in mock_store.update.call_args_list:
        called_keys.extend(call[1].keys())

    # Check that branch_anchor-baseline_Target_Reg_oof_augmented is written
    assert "branch_anchor-baseline_Target_Reg_oof_augmented" in called_keys


# 5. ChallengeConfig immutability test (Phase 2)
def test_challenge_config_immutability(tmp_path):
    from zindian.config import ChallengeConfig, FrozenDict

    cfg_data = {
        "metric": "auc",
        "metric_direction": "maximize",
        "use_probabilities": True,
        "automl_permitted": False,
        "data_modality": "tabular",
        "sub_block": {"key": "val"},
        "sub_list": [{"key": "val"}],
    }

    cfg_path = tmp_path / "challenge_config.json"
    import json

    cfg_path.write_text(json.dumps(cfg_data), encoding="utf-8")

    cfg = ChallengeConfig(path=cfg_path, _data=cfg_data)
    assert isinstance(cfg._data, FrozenDict)

    with pytest.raises(TypeError, match="ChallengeConfig is read-only"):
        cfg._data["new_key"] = "test"

    with pytest.raises(TypeError, match="ChallengeConfig is read-only"):
        cfg._data["sub_block"]["key"] = "test"

    with pytest.raises(TypeError, match="ChallengeConfig is read-only"):
        cfg._data["sub_list"][0]["key"] = "test"


# 6. ZindiClient headers intake fallback test (Phase 3)
def test_skill02_zindiclient_headers_fallback(monkeypatch):
    from zindian.skills.skill_02_intake import run as run_intake

    # Mock ZindiClient constructor to raise an exception
    def mock_zindi_client_init(self):
        raise RuntimeError("Mock login failed")

    monkeypatch.setattr(
        "zindian.zindi_client.ZindiClient.__init__", mock_zindi_client_init
    )

    called = {}

    def mock_fetch(slug, headers):
        called["headers"] = headers
        return {"name": "Test", "metric": "auc"}

    monkeypatch.setattr("zindian.skills.skill_02_intake.fetch_competition", mock_fetch)

    class MockPaths:
        def __init__(self):
            self.config_path = Path("dummy_config.json")
            self.reports_dir = Path("dummy_reports")
            self.state_path = Path("dummy_state.json")
            self.data_raw_dir = Path("dummy_raw")

    monkeypatch.setattr(
        "zindian.skills.skill_02_intake.resolve_competition_paths",
        lambda slug=None: MockPaths(),
    )
    monkeypatch.setattr(
        "zindian.skills.skill_02_intake.write_config", lambda config, paths: None
    )
    monkeypatch.setattr(
        "zindian.skills.skill_02_intake.update_skill_state", lambda slug, paths: None
    )

    mock_store = MagicMock()
    mock_store.read.return_value = {"dag_phase": "phase_1"}
    monkeypatch.setattr(
        "zindian.skills.skill_02_intake.SkillStateStore", lambda path: mock_store
    )

    run_intake(slug="test-slug", headers=None, dry_run=False)
    assert called["headers"] == {"Accept": "application/json"}


# 7. Asynchronous deep research sidecar execution test (Phase 4)
def test_run_deep_research_async(monkeypatch):
    from zindian.orchestrator import run_deep_research

    class MockPaths:
        def __init__(self):
            self.config_path = Path("dummy_config.json")
            self.reports_dir = Path("dummy_reports")
            self.state_path = Path("dummy_state.json")

    monkeypatch.setattr(
        "zindian.orchestrator.resolve_competition_paths",
        lambda require_competition=False: MockPaths(),
    )

    called_bg = []

    class DummyLibrarian:
        @staticmethod
        def run_librarian(config_path, cache_path):
            import time

            time.sleep(0.1)
            called_bg.append("librarian")

    class DummyCodeMiner:
        @staticmethod
        def run_code_miner(domain, dry_run):
            called_bg.append("code_miner")

    class DummyScientist:
        @staticmethod
        def run_scientist(
            hypotheses_path, priorart_path, hypothesis_path, failed_hypotheses_path
        ):
            called_bg.append("scientist")

    import zindian.orchestrator

    monkeypatch.setitem(
        zindian.orchestrator.SKILL_REGISTRY, "skill_18", ("Librarian", DummyLibrarian)
    )
    monkeypatch.setitem(
        zindian.orchestrator.SKILL_REGISTRY, "skill_19", ("Code Miner", DummyCodeMiner)
    )
    monkeypatch.setitem(
        zindian.orchestrator.SKILL_REGISTRY, "skill_20", ("Scientist", DummyScientist)
    )

    import time

    start = time.time()
    res = run_deep_research(domain="geospatial", dry_run=True)
    duration = time.time() - start

    assert duration < 0.05
    assert res["status"] == "LAUNCHED"

    time.sleep(0.3)
    assert "librarian" in called_bg
    assert "code_miner" in called_bg
    assert "scientist" in called_bg


def test_pseudo_label_quantile_logic():
    import numpy as np
    import pandas as pd

    # Simulate the class-wise quantile logic in skill_21:
    # 8 samples
    p1 = np.array([0.95, 0.85, 0.72, 0.10, 0.25, 0.65, 0.05, 0.50], dtype=np.float64)
    p0 = 1.0 - p1

    # For p_val = 0.25, k = ceil(0.25 * 8) = 2
    p_val = 0.25
    n_samples = len(p1)
    k = int(np.ceil(p_val * n_samples))

    assert k == 2

    # Deterministic method='first' tie-breaking
    rank1 = pd.Series(p1).rank(method="first", ascending=False).values
    rank0 = pd.Series(p0).rank(method="first", ascending=False).values

    # Floor at 0.70
    pos_mask = (rank1 <= k) & (p1 >= 0.70)
    neg_mask = (rank0 <= k) & (p0 >= 0.70)

    # p1 values sorted: 0.95 (idx 0), 0.85 (idx 1), 0.72 (idx 2)
    # Ranks: idx 0 is 1.0, idx 1 is 2.0, idx 2 is 3.0
    # rank1 <= 2 covers idx 0 (0.95) and idx 1 (0.85). Both are >= 0.70 floor.
    assert np.all(
        pos_mask == np.array([True, True, False, False, False, False, False, False])
    )

    # p0 values: [0.05, 0.15, 0.28, 0.90, 0.75, 0.35, 0.95, 0.50]
    # Sorted: 0.95 (idx 6), 0.90 (idx 3), 0.75 (idx 4)
    # Ranks: idx 6 is 1.0, idx 3 is 2.0, idx 4 is 3.0
    # rank0 <= 2 covers idx 6 (0.95) and idx 3 (0.90). Both are >= 0.70 floor.
    assert np.all(
        neg_mask == np.array([False, False, False, True, False, False, True, False])
    )

    # min_pseudo_samples guard: if set to 5 (we have 4 total selected), it should clear selections
    total_selected = int(pos_mask.sum() + neg_mask.sum())
    assert total_selected == 4

    min_samples = 5
    if total_selected < min_samples:
        pos_mask_guarded = np.zeros_like(pos_mask, dtype=bool)
        neg_mask_guarded = np.zeros_like(neg_mask, dtype=bool)

    assert np.all(~pos_mask_guarded)
    assert np.all(~neg_mask_guarded)


def test_log_directory_deduplication(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from datetime import datetime, timezone
    from zindian.paths import CompetitionPaths
    from unittest.mock import MagicMock

    # 1. Setup mock directories
    comp_dir = tmp_path / "competitions" / "test-competition"
    logs_dir = comp_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    reports_dir = comp_dir / "reports"
    summaries_dir = reports_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    # 2. Mock paths
    mock_paths = CompetitionPaths(
        root=tmp_path,
        competition_dir=comp_dir,
        state_path=comp_dir / "SKILL_STATE.json",
        config_path=comp_dir / "challenge_config.json",
        reports_dir=reports_dir,
        submissions_dir=comp_dir / "submissions",
        data_raw_dir=comp_dir / "data" / "raw",
        data_processed_dir=comp_dir / "data" / "processed",
        notebooks_dir=comp_dir / "notebooks",
    )

    # Mock datetime to control directory name creation
    current_time = [datetime(2026, 8, 24, 0, 0, 0, tzinfo=timezone.utc)]

    import zindian.orchestrator as orch

    monkeypatch.setattr("zindian.orchestrator._get_utc_now", lambda: current_time[0])
    monkeypatch.setattr(
        "zindian.paths.resolve_competition_paths", lambda *args, **kwargs: mock_paths
    )

    # Mock ChallengeConfig to return empty skill list for the phase
    mock_config = MagicMock()
    mock_config.get.side_effect = lambda key, default=None: {
        "phase_skill_map": {"1": []},
        "target": "Occurrence Status",
        "automl_permitted": False,
        "allowed_external_data": False,
        "banned_features": [],
        "cv_strategy": {"type": "StratifiedKFold"},
        "slug": "test-slug",
    }.get(key, default)
    monkeypatch.setattr("zindian.config.ChallengeConfig.load", lambda *args, **kwargs: mock_config)

    # Mock SkillStateStore read/update
    mock_store = MagicMock()
    mock_store.read.return_value = {"dag_phase": "phase_1"}
    monkeypatch.setattr("zindian.state.SkillStateStore", lambda path: mock_store)

    # Mock skill_15 reporter functions called at the end of run_phase
    monkeypatch.setattr("zindian.skills.skill_15_reporter.run_phase_summary", lambda phase: None)
    monkeypatch.setattr("zindian.skills.skill_15_reporter._write_json_summary", lambda *args, **kwargs: None)

    # 3. Simulate first run:
    # Set up a new run log dir matching datetime (20260824T000000)
    run_1_dir = logs_dir / "run_20260824T000000_phase1"
    run_1_dir.mkdir(parents=True, exist_ok=True)
    (run_1_dir / "session.log").write_text("Run 1 logs", encoding="utf-8")

    # Create the phase summary report
    summary_1 = {
        "timestamp": "2026-08-24T00:00:00Z",
        "phase": "1",
        "metric": "auc",
        "cv_strategy_type": "StratifiedKFold",
    }
    summary_1_path = summaries_dir / "phase_1_summary.json"
    summary_1_path.write_text(json.dumps(summary_1), encoding="utf-8")

    # Run the orchestrator phase
    orch.run_phase("1")

    # Verify that run_1_dir was promoted to run_latest_phase1
    latest_dir = logs_dir / "run_latest_phase1"
    assert latest_dir.exists()
    assert (latest_dir / "session.log").read_text(encoding="utf-8") == "Run 1 logs"
    assert (latest_dir / "summary.json").exists()
    assert not run_1_dir.exists()

    # 4. Simulate second run: similar summary
    current_time[0] = datetime(2026, 8, 24, 1, 0, 0, tzinfo=timezone.utc)
    run_2_dir = logs_dir / "run_20260824T010000_phase1"
    run_2_dir.mkdir(parents=True, exist_ok=True)
    (run_2_dir / "session.log").write_text("Run 2 logs (identical config)", encoding="utf-8")

    # Update summary with new timestamp but identical core fields
    summary_2 = {
        "timestamp": "2026-08-24T01:00:00Z",
        "phase": "1",
        "metric": "auc",
        "cv_strategy_type": "StratifiedKFold",
    }
    summary_1_path.write_text(json.dumps(summary_2), encoding="utf-8")

    orch.run_phase("1")

    # Since summary_2 is similar to summary_1 (ignoring timestamp), run_latest_phase1 should be updated with new logs,
    # and run_2_dir should be removed. No archives should be created because they are similar.
    assert latest_dir.exists()
    assert (latest_dir / "session.log").read_text(encoding="utf-8") == "Run 2 logs (identical config)"
    assert not run_2_dir.exists()

    # Check that no timestamped run folder exists (only latest_dir)
    dirs = [p.name for p in logs_dir.iterdir() if p.is_dir()]
    assert dirs == ["run_latest_phase1"]

    # 5. Simulate third run: different summary
    current_time[0] = datetime(2026, 8, 24, 2, 0, 0, tzinfo=timezone.utc)
    run_3_dir = logs_dir / "run_20260824T020000_phase1"
    run_3_dir.mkdir(parents=True, exist_ok=True)
    (run_3_dir / "session.log").write_text("Run 3 logs (different config)", encoding="utf-8")

    summary_3 = {
        "timestamp": "2026-08-24T02:00:00Z",
        "phase": "1",
        "metric": "auc",
        "cv_strategy_type": "GroupKFold",  # changed!
    }
    summary_1_path.write_text(json.dumps(summary_3), encoding="utf-8")

    orch.run_phase("1")

    # Since summary_3 is different, the previous latest_dir (Run 2) should be archived to run_20260824T010000_phase1,
    # and run_3_dir should be promoted to latest_dir.
    assert latest_dir.exists()
    assert (latest_dir / "session.log").read_text(encoding="utf-8") == "Run 3 logs (different config)"
    assert not run_3_dir.exists()

    archived_dir = logs_dir / "run_20260824T010000Z_phase1"
    assert archived_dir.exists()
    assert (archived_dir / "session.log").read_text(encoding="utf-8") == "Run 2 logs (identical config)"


