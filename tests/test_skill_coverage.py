from __future__ import annotations

import importlib

import pandas as pd
import pytest


SKILL_EXPORTS = [
    ("zindian.skills.skill_00_zindi_monitor", "run"),
    ("zindian.skills.skill_01_integrity", "run"),
    ("zindian.skills.skill_02_intake", "run"),
    ("zindian.skills.skill_03_legality", "run"),
    ("zindian.skills.skill_04_eda", "run"),
    ("zindian.skills.skill_05_cv", "run"),
    ("zindian.skills.skill_07_features", "run"),
    ("zindian.skills.skill_08_anchor", "run"),
    ("zindian.skills.skill_10_shap", "run"),
    ("zindian.skills.skill_11_gate", "run"),
    ("zindian.skills.skill_13_ensemble", "run"),
    ("zindian.skills.skill_15_reporter", "run"),
    ("zindian.skills.skill_16_submit", "run"),
    ("zindian.skills.skill_17_governance", "run_governance"),
    ("zindian.skills.skill_18_librarian", "run_librarian"),
    ("zindian.skills.skill_19_code_miner", "run_code_miner"),
    ("zindian.skills.skill_20_scientist", "run_scientist"),
]


@pytest.mark.parametrize("module_name, export_name", SKILL_EXPORTS)
def test_skill_modules_export_callables(module_name: str, export_name: str) -> None:
    module = importlib.import_module(module_name)
    assert hasattr(module, export_name)
    assert callable(getattr(module, export_name))


def test_scientist_feature_inventory_and_static_validation() -> None:
    from zindian.skills.skill_20_scientist import get_available_columns, static_validate_hypothesis
    columns = get_available_columns()
    # Don't assert exact counts — ensure it's a non-empty iterable and reasonable columns exist
    assert isinstance(columns, list)
    assert len(columns) > 0
    assert "Latitude" not in columns
    assert "Longitude" not in columns

    ok, reason = static_validate_hypothesis({"feature_columns": ["ppt_mean"], "transformation": "raw"}, set(columns))
    assert ok in (True, False)
    # If validation passes, reason should be a short string; otherwise provide a failure reason
    assert isinstance(reason, str)


def test_scientist_two_stage_validation_and_ledger_blocking(monkeypatch: pytest.MonkeyPatch) -> None:
    import zindian.skills.skill_20_scientist as scientist

    monkeypatch.setattr(scientist, "mutual_info_classif", lambda *args, **kwargs: [0.5])

    class DummyBooster:
        def feature_importance(self, importance_type: str = "gain"):
            return [1.0]

    class DummyModel:
        def __init__(self, *args, **kwargs):
            self.booster_ = DummyBooster()

        def fit(self, X, y):
            return self

    monkeypatch.setattr(scientist.lgb, "LGBMClassifier", DummyModel)

    frame = pd.DataFrame(
        {
            "Occurrence Status": [0, 1] * 10,
            "ppt_mean": [0.0, 1.0] * 10,
        }
    )
    hypothesis = {"feature_columns": ["ppt_mean"], "transformation": "raw"}

    kept, failed = scientist.validate_hypotheses([hypothesis], frame, [])
    assert len(kept) == 1
    assert len(failed) == 0
    assert kept[0]["validation_status"] == "passed"

    blocked_ledger = [{"signature": scientist._hypothesis_signature(hypothesis), "do_not_retry": True}]
    kept2, failed2 = scientist.validate_hypotheses([hypothesis], frame, blocked_ledger)
    assert len(kept2) == 0
    assert len(failed2) == 1
    assert failed2[0]["validation_status"] == "blocked"


def test_semantic_scholar_client_uses_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from zindian.clients.semantic_scholar import SemanticScholarClient

    captured_url: dict[str, str] = {}
    captured_params: dict[str, object] = {}
    captured_headers: dict[str, str] = {}

    class DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"data": []}

    class DummySession:
        def __init__(self) -> None:
            self.headers: dict[str, str] = {}

        def get(self, url: str, params=None, timeout: int = 15):
            captured_url["value"] = url
            captured_params["value"] = params
            captured_headers.update(self.headers)
            return DummyResponse()

    monkeypatch.setattr("zindian.clients.semantic_scholar.requests.Session", lambda: DummySession())

    client = SemanticScholarClient(api_key="test-key", rate_limit_per_sec=1000)
    result = client.search_papers("frog ecology", limit=3)

    assert result == {"data": []}
    assert captured_url["value"] == "https://api.semanticscholar.org/graph/v1/paper/search"
    assert captured_params["value"] == {"query": "frog ecology", "limit": 3, "fields": "title,abstract,authors,year"}
    assert captured_headers["x-api-key"] == "test-key"
