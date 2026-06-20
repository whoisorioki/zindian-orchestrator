import os
import json
import tempfile
from pathlib import Path
import pytest

from plugins import terraclimate_extractor as tc
from zindian.config import ChallengeConfig


def test_fetch_guard_raises_on_no_network(tmp_path):
    # Ensure env var disables network
    os.environ["ZINDIAN_DISABLE_NETWORK"] = "1"

    class P:
        data_processed_dir = tmp_path / "data" / "processed"

    P.data_processed_dir.mkdir(parents=True)

    # Build a minimal ChallengeConfig for the test
    cfg_data: dict = {}
    cfg_path = Path(tempfile.mktemp(suffix=".json"))
    cfg_path.write_text(
        json.dumps(
            {
                "metric": "f1",
                "metric_direction": "maximize",
                "use_probabilities": False,
                "automl_permitted": False,
                "data_modality": "tabular",
            }
        )
    )
    dummy_config = ChallengeConfig(path=cfg_path, _data=cfg_data)

    with pytest.raises(RuntimeError, match="Network fetch disabled"):
        tc.fetch(P(), dummy_config)
    # cleanup
    del os.environ["ZINDIAN_DISABLE_NETWORK"]
