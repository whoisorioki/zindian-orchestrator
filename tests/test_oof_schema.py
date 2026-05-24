import pytest


def sample_oof():
    return {
        "scores": [0.1, 0.9, 0.2],
        "cv_strategy_id": "stratified",
        "seed": 42,
        "branch_name": "anchor",
        "model_config": {"num_leaves": 31},
    }


def test_oof_schema_keys_present():
    oof = sample_oof()
    required = {"scores", "cv_strategy_id", "seed", "branch_name", "model_config"}
    assert required.issubset(set(oof.keys()))


def test_oof_scores_type_and_length():
    oof = sample_oof()
    assert isinstance(oof["scores"], list)
    assert len(oof["scores"]) > 0
    assert all(isinstance(x, (float, int)) for x in oof["scores"]) 
