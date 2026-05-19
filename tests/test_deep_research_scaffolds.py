import json, pytest
from pathlib import Path

REPORTS = Path("competitions/ey-frogs/reports")

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

def test_governance_imports():
    from zindian.skills.skill_17_governance import run_governance
    assert callable(run_governance)

# ── Artifact existence & schema tests ────────────────────────────────────────

def test_literature_cache_exists_and_valid_json():
    p = REPORTS / "literature_cache.json"
    assert p.exists(), "literature_cache.json not found"
    data = json.loads(p.read_text())
    assert "status" in data or "entries" in data, "Unexpected schema"

def test_feature_hypothesis_exists_and_valid_json():
    p = REPORTS / "feature_hypothesis.json"
    assert p.exists(), "feature_hypothesis.json not found"
    data = json.loads(p.read_text())
    # Accepts scaffold (dict with status) or real output (list of hypotheses)
    assert isinstance(data, (dict, list))

def test_feature_policy_exists_and_has_required_keys():
    p = REPORTS / "feature_policy.json"
    assert p.exists(), "feature_policy.json not found"
    policy = json.loads(p.read_text())
    for key in ("allowed_data_sources", "external_data_permitted", "coordinate_features_permitted"):
        assert key in policy, f"Missing key: {key}"

def test_legality_report_exists():
    p = REPORTS / "legality_report.md"
    assert p.exists(), "legality_report.md not found"
    content = p.read_text()
    assert "PASS" in content or "BLOCK" in content, "Report missing status"

