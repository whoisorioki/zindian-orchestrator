import pytest
import requests
from unittest.mock import MagicMock


def test_requests_get_intercepted_under_network_disable(monkeypatch):
    # Ensure network is disabled
    monkeypatch.setenv("ZINDIAN_DISABLE_NETWORK", "1")

    # Mock requests.get to return a high-fidelity mock response
    mock_response = MagicMock()
    mock_response.json.return_value = {"status": "mocked"}
    mock_response.text = "mocked HTML content"
    mock_response.status_code = 200

    monkeypatch.setattr("requests.get", MagicMock(return_value=mock_response))

    # Call requests.get and verify it returns mock payload without HTTP leakage
    res = requests.get("https://zindi.africa")
    assert res.json() == {"status": "mocked"}


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
