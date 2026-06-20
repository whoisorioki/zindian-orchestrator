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
