from unittest.mock import patch

from zindian.orchestrator import prompt_human_gate


class DummyStore:
    def __init__(self, data=None):
        self.data = data or {}

    def read(self):
        return self.data

    def update(self, **kwargs):
        self.data.update(kwargs)


class DummyConfig:
    def __init__(self, data=None):
        self._data = data or {}

    def get(self, key, default=None):
        return self._data.get(key, default)


def test_prompt_human_gate_already_approved():
    store = DummyStore({"human_gate_1_approved": True})
    config = DummyConfig()
    # Should return True immediately without prompting
    assert (
        prompt_human_gate("Gate 1", store, store.read(), config, non_interactive=False)
        is True
    )


def test_prompt_human_gate_non_interactive_fallback():
    store = DummyStore({"human_gate_3_approved": False})
    config = DummyConfig()
    # In non-interactive mode, if gate is not approved, should return False
    assert (
        prompt_human_gate("Gate 3", store, store.read(), config, non_interactive=True)
        is False
    )


@patch("builtins.input", return_value="A")
@patch("sys.stdin.isatty", return_value=True)
def test_prompt_gate_1_approve(mock_isatty, mock_input):
    store = DummyStore({"human_gate_1_approved": False})
    config = DummyConfig({"cv_strategy": {"type": "KFold"}})

    assert (
        prompt_human_gate("Gate 1", store, store.read(), config, non_interactive=False)
        is True
    )
    assert store.read().get("human_gate_1_approved") is True


@patch("builtins.input", return_value="B")
@patch("sys.stdin.isatty", return_value=True)
def test_prompt_gate_1_reject(mock_isatty, mock_input):
    store = DummyStore({"human_gate_1_approved": False})
    config = DummyConfig({"cv_strategy": {"type": "KFold"}})

    assert (
        prompt_human_gate("Gate 1", store, store.read(), config, non_interactive=False)
        is False
    )
    assert store.read().get("human_gate_1_approved") is not True


@patch("builtins.input", return_value="YES")
@patch("sys.stdin.isatty", return_value=True)
def test_prompt_gate_2_approve(mock_isatty, mock_input):
    store = DummyStore()
    config = DummyConfig()

    assert (
        prompt_human_gate(
            "Gate 2",
            store,
            store.read(),
            config,
            variant_name="var1",
            non_interactive=False,
        )
        is True
    )
    assert store.read().get("human_gate_2_var1_approved") is True


@patch("builtins.input", return_value="NO")
@patch("sys.stdin.isatty", return_value=True)
def test_prompt_gate_3_reject(mock_isatty, mock_input):
    store = DummyStore()
    config = DummyConfig()

    assert (
        prompt_human_gate("Gate 3", store, store.read(), config, non_interactive=False)
        is False
    )
    assert store.read().get("human_gate_3_approved") is not True
