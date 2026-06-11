import os
import pytest
from pathlib import Path

from zindian.skills import skill_07_features as feat


def test_fetch_guard_raises_on_no_network(tmp_path):
    # Ensure env var disables network
    os.environ["ZINDIAN_DISABLE_NETWORK"] = "1"
    class P:
        data_processed_dir = tmp_path / "data" / "processed"
        data_processed_dir.mkdir(parents=True)
        # rest not used by fetch_terraclimate
    with pytest.raises(RuntimeError):
        feat.fetch_terraclimate(P())
    # cleanup
    del os.environ["ZINDIAN_DISABLE_NETWORK"]
