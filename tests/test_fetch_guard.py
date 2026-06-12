import os
import pytest

from plugins import terraclimate_extractor as tc


def test_fetch_guard_raises_on_no_network(tmp_path):
    # Ensure env var disables network
    os.environ["ZINDIAN_DISABLE_NETWORK"] = "1"

    class P:
        data_processed_dir = tmp_path / "data" / "processed"

    P.data_processed_dir.mkdir(parents=True)

    class DummyConfig:
        def get(self, key, default=None):
            return default

    with pytest.raises(RuntimeError, match="Network fetch disabled"):
        tc.fetch(P(), DummyConfig())
    # cleanup
    del os.environ["ZINDIAN_DISABLE_NETWORK"]
