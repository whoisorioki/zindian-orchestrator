"""GeoAI Aquaculture Pond Feature Extractor.

Pass-through for this tabular competition: reads raw Train.csv/Test.csv,
drops ID and targets, writes features_train.csv and features_test.csv
to data/processed/.
"""

from pathlib import Path
from typing import Any, Tuple
import pandas as pd


def extract(
    paths, tiff_path: Path, config
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    input_files = config.get("input_files", {}) or {}
    train_file = input_files.get("train", "Train.csv")
    test_file = input_files.get("test", "Test.csv")
    id_col = config.get("id_col", "ID")

    train = pd.read_csv(paths.data_raw_dir / train_file)
    test = pd.read_csv(paths.data_raw_dir / test_file)

    target_config = config.get("target_config", {}) or {}
    targets = [t["name"] for t in target_config.get("targets", [])]
    drop_cols = [id_col] + targets
    # Drop ID, targets, and banned features from config
    banned = list(config.get("banned_features") or [])
    drop_cols = list(set(drop_cols + banned))
    train = train.drop(columns=drop_cols, errors="ignore")
    test = test.drop(columns=list(set([id_col] + banned)), errors="ignore")

    # Normalize masked observations (-9999 -> NaN) so downstream skills can handle partial windows
    plugin_config = (config.get("plugin_config") or {})
    masked_value = plugin_config.get("masked_value", -9999)
    if masked_value is not None:
        train = train.replace(masked_value, float("nan"))
        test = test.replace(masked_value, float("nan"))

    paths.data_processed_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(paths.data_processed_dir / "features_train.csv", index=False)
    test.to_csv(paths.data_processed_dir / "features_test.csv", index=False)
    return train, test