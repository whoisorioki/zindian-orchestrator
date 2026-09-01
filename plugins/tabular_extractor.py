"""General Tabular Feature Extractor.

Pass-through for tabular competitions: reads raw Train.csv / Test.csv,
drops ID and banned features (retains targets in train), then writes
features_train_{branch}.csv and features_test_{branch}.csv to data/processed/
for reproducibility contract.
"""

from pathlib import Path
from typing import Tuple
import pandas as pd
from plugins.base_extractor import FeatureExtractor


class Extractor(FeatureExtractor):
    """General tabular feature extractor implementing FeatureExtractor ABC."""

    def fetch(self, paths, config, allow_network: bool = True):
        """No external data needed by default."""
        return None

    def extract(
        self, paths, data_path: Path, config, branch_name: str | None = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        input_files = config.get("input_files", {}) or {}
        train_file = input_files.get("train", "Train.csv")
        test_file = input_files.get("test", "Test.csv")
        id_col = config.get("id_col", "ID")

        train = pd.read_csv(paths.data_raw_dir / train_file)
        test = pd.read_csv(paths.data_raw_dir / test_file)

        banned = list(config.get("banned_features") or [])
        drop_cols_train = list(set([c for c in [id_col] + banned if c in train.columns]))
        drop_cols_test = list(set([c for c in [id_col] + banned if c in test.columns]))

        train_feat = train.drop(columns=drop_cols_train, errors="ignore") if drop_cols_train else train
        test_feat = test.drop(columns=drop_cols_test, errors="ignore") if drop_cols_test else test

        branch = branch_name or "anchor-baseline"
        paths.data_processed_dir.mkdir(parents=True, exist_ok=True)
        train_feat.to_csv(
            paths.data_processed_dir / f"features_train_{branch}.csv", index=False
        )
        test_feat.to_csv(
            paths.data_processed_dir / f"features_test_{branch}.csv", index=False
        )
        return train_feat, test_feat


def extract(
    paths, data_path: Path, config, branch_name: str = "anchor-baseline"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Legacy function for backward compatibility."""
    extractor = Extractor(config)
    return extractor.extract(paths, data_path, config, branch_name)
