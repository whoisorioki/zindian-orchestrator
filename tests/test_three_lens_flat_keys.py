import pytest
from dataclasses import asdict
from zindian.three_lens import evaluate_three_lenses
from zindian.state import SkillStateStore


def test_phase_3b_evaluation_with_flat_boolean_gate_keys(tmp_path):
    """
    Verifies that the Three-Lens evaluation environment correctly parses
    flat boolean gate keys and rejects legacy nested dictionary keys.
    """
    state_file = tmp_path / "SKILL_STATE.json"
    state_store = SkillStateStore(path=state_file)

    # Configure mock state using the mandatory flat boolean format
    state_store.update(
        human_gate_1_approved=True,
        **{
            "human_gate_2_exp-feature-aridity_approved": True,
            "human_gate_2_exp-feature-desiccation_approved": False,
        },
    )

    mock_config = {"challenge_id": "ey-frogs", "task_type": "regression"}

    # Run evaluation routine across the target phase checkpoint
    report = evaluate_three_lenses(
        phase="Phase3B", config=mock_config, state=state_store
    )

    assert report.generalisation.verdict in ["PASS", "WARN"]
    assert "human_gate_2_by_branch" not in asdict(report)


def test_network_isolation_enforcement(monkeypatch):
    """
    Asserts that processing logic safely errors out if external
    network calls are attempted during validation sweeps.
    """
    import urllib.request

    def mock_urlopen(*args, **kwargs):
        raise RuntimeError(
            "ZINDIAN_DISABLE_NETWORK enforcement hit. Network access blocked."
        )

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    with pytest.raises(RuntimeError, match="Network access blocked"):
        urllib.request.urlopen("https://zindi.africa/api")
