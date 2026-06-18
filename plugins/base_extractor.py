"""Base Feature Extractor ABC for multi-target competitions.

All column names read from config — no string literals for competition-specific
column names (A5). Must NOT include target columns in output. Must handle missing
test data gracefully. Must log feature names to reports/feature_manifest.json.
Must be deterministic (A7) — same input -> same output, no API calls, no randomness,
no filesystem side effects beyond data/processed/.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


class FeatureExtractor(ABC):
    """Abstract base class for competition-specific feature extraction plugins."""

    @abstractmethod
    def extract_features(
        self, raw_data_dir: Path, config: Dict
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract features from raw competition data.

        Args:
            raw_data_dir: Path to directory containing raw data files
            config: challenge_config.json dict containing:
                - file_manifest: {"train": "train.csv", "test": "test.csv"}
                - plugin_config: plugin-specific parameters
                - target_config: (multi-target only) target specifications

        Returns:
            Tuple of (train_features, test_features) DataFrames

        Contract:
            - All column names read from config — no hardcoded strings (A5)
            - Must NOT include target columns in output
            - Must handle missing test data gracefully
            - Must log feature names to reports/feature_manifest.json
            - Must be deterministic (A7) — no API calls, no randomness
        """
        pass
