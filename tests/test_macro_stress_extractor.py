"""Unit tests for plugins/macro_stress_extractor.py and scripts/build_macro_stress_proxies.py."""

from pathlib import Path
import pandas as pd
import pytest

from scripts.build_macro_stress_proxies import build_macro_stress_proxies
from plugins.macro_stress_extractor import Extractor


def test_macro_stress_extractor_end_to_end(tmp_path):
    raw_dir = tmp_path / "data" / "raw"
    ext_dir = tmp_path / "data" / "external"
    proc_dir = tmp_path / "data" / "processed"
    raw_dir.mkdir(parents=True)
    ext_dir.mkdir(parents=True)
    proc_dir.mkdir(parents=True)

    # Make synthetic train/test
    train_df = pd.DataFrame(
        {
            "ID": [1, 2, 3],
            "latitude": [10.0, 10.0, 12.0],
            "longitude": [30.0, 30.0, 32.0],
            "date": ["2020-01-01", "2020-02-01", "2020-01-01"],
            "target": [0, 1, 0],
        }
    )
    test_df = pd.DataFrame(
        {
            "ID": [4, 5],
            "latitude": [10.0, 12.0],
            "longitude": [30.0, 32.0],
            "date": ["2020-03-01", "2020-02-01"],
        }
    )

    train_df.to_csv(raw_dir / "Train.csv", index=False)
    test_df.to_csv(raw_dir / "Test.csv", index=False)

    config = {
        "input_files": {"train": "Train.csv", "test": "Test.csv"},
        "id_col": "ID",
        "temporal_col": "date",
        "columns": {"latitude": "latitude", "longitude": "longitude"},
        "plugin_config": {
            "external_dataset_path": "data/external/macro_stress_proxies.parquet"
        },
    }

    class MockPaths:
        root = tmp_path
        data_raw_dir = raw_dir
        data_processed_dir = proc_dir

    paths = MockPaths()

    extractor = Extractor(config)
    ext_path = extractor.fetch(paths, config)
    assert ext_path.exists()

    tr_out, te_out = extractor.extract(paths, ext_path, config, branch_name="test")

    assert "spei_6" in tr_out.columns
    assert "viirs_radiance_lag1m" in tr_out.columns
    assert "spei_6_is_imputed" in tr_out.columns
    assert "viirs_radiance_lag1m_is_imputed" in tr_out.columns
    assert "ID" not in tr_out.columns
