import json
import pytest
from pathlib import Path

REPORTS = Path("competitions/ey-frogs/reports")

# If the reports directory is not populated in this environment, skip
# the artifact existence tests — these are integration checks, not unit tests.
REPORTS_AVAILABLE = REPORTS.exists() and (REPORTS / "literature_cache.json").exists()

# ── Import smoke tests ────────────────────────────────────────────────────────


def test_librarian_imports():
    from zindian.skills.skill_18_librarian import run_librarian

    assert callable(run_librarian)


def test_scientist_imports():
    from zindian.skills.skill_20_scientist import run_scientist

    assert callable(run_scientist)


def test_code_miner_imports():
    from zindian.skills.skill_19_code_miner import run_code_miner

    assert callable(run_code_miner)


def test_deep_research_orchestrator_imports():
    from zindian.orchestrator import run_deep_research

    assert callable(run_deep_research)


def test_governance_imports():
    from zindian.skills.skill_17_governance import run_governance

    assert callable(run_governance)


# ── Artifact existence & schema tests ────────────────────────────────────────


def test_literature_cache_exists_and_valid_json():
    if not REPORTS_AVAILABLE:
        pytest.skip("reports not available in this environment")
    p = REPORTS / "literature_cache.json"
    assert p.exists(), "literature_cache.json not found"
    data = json.loads(p.read_text())
    assert "status" in data or "entries" in data, "Unexpected schema"


def test_feature_hypothesis_exists_and_valid_json():
    if not REPORTS_AVAILABLE:
        pytest.skip("reports not available in this environment")
    p = REPORTS / "feature_hypothesis.json"
    assert p.exists(), "feature_hypothesis.json not found"
    data = json.loads(p.read_text())
    # Accepts scaffold (dict with status) or real output (list of hypotheses)
    assert isinstance(data, (dict, list))


def test_feature_policy_exists_and_has_required_keys():
    if not REPORTS_AVAILABLE:
        pytest.skip("reports not available in this environment")
    p = REPORTS / "feature_policy.json"
    assert p.exists(), "feature_policy.json not found"
    policy = json.loads(p.read_text())
    for key in (
        "allowed_data_sources",
        "external_data_permitted",
        "coordinate_features_permitted",
    ):
        assert key in policy, f"Missing key: {key}"


def test_legality_report_exists():
    if not REPORTS_AVAILABLE:
        pytest.skip("reports not available in this environment")
    p = REPORTS / "legality_report.md"
    assert p.exists(), "legality_report.md not found"
    content = p.read_text()
    assert "PASS" in content or "BLOCK" in content, "Report missing status"


def test_code_miner_cache_exists_and_has_schema():
    if not REPORTS_AVAILABLE:
        pytest.skip("reports not available in this environment")
    p = REPORTS / "code_miner_cache.json"
    assert p.exists(), "code_miner_cache.json not found"
    cache = json.loads(p.read_text())
    for key in (
        "generated_at",
        "status",
        "model",
        "query_count",
        "queries",
        "raw_count",
    ):
        assert key in cache, f"Missing key: {key}"
    assert isinstance(cache["queries"], list)
    assert isinstance(cache["raw_count"], int)


def test_code_miner_patterns_exists_and_has_schema():
    if not REPORTS_AVAILABLE:
        pytest.skip("reports not available in this environment")
    p = REPORTS / "code_miner_patterns.json"
    assert p.exists(), "code_miner_patterns.json not found"
    patterns = json.loads(p.read_text())
    for key in ("generated_at", "status", "patterns_count", "patterns"):
        assert key in patterns, f"Missing key: {key}"
    assert isinstance(patterns["patterns"], list)
    if patterns["patterns"]:
        first = patterns["patterns"][0]
        for key in (
            "pattern_id",
            "source_type",
            "technique_name",
            "implementation_steps",
            "confidence",
        ):
            assert key in first, f"Missing pattern key: {key}"
