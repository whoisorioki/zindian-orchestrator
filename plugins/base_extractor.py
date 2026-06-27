"""Base Feature Extractor ABC for multi-target competitions.

All column names read from config — no string literals for competition-specific
column names (A5). Must NOT include target columns in output. Must handle missing
test data gracefully. Must log feature names to reports/feature_manifest.json.
Must be deterministic (A7) — same input -> same output, no API calls, no randomness,
no filesystem side effects beyond data/processed/.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Tuple, Any

import pandas as pd


class FeatureExtractor(ABC):
    """Abstract base class for competition-specific feature extraction plugins."""

    def fetch(
        self, paths: Any, config: Any, allow_network: bool = True
    ) -> Path:
        """Fetch raw external source data (e.g. satellite imagery tiff).

        Args:
            paths: CompetitionPaths object
            config: ChallengeConfig object
            allow_network: Whether network requests are allowed

        Returns:
            Path to the fetched data file
        """
        return paths.data_processed_dir / "plugin_data.tiff"

    @abstractmethod
    def extract(
        self, paths: Any, tiff_path: Path, config: Any
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract features from raw competition data.

        Args:
            paths: CompetitionPaths object
            tiff_path: Path to fetched imagery or auxiliary data
            config: ChallengeConfig object

        Returns:
            Tuple of (train_features, test_features) DataFrames
        """
        pass
