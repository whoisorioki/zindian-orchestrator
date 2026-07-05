"""Base class for competition-specific feature extractors."""
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import pandas as pd


class FeatureExtractor(ABC):
    """
    Abstract base class for competition-specific feature extractors.
    
    All plugins must inherit from this class and implement:
    - fetch(): Download/load external data
    - extract(): Transform raw data into features
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize extractor with config.
        
        Args:
            config: challenge_config.json data
        """
        self.config = config or {}
        self.plugin_config = self.config.get("plugin_config", {})
    
    @abstractmethod
    def fetch(self, paths, config, allow_network: bool = True):
        """
        Fetch external data sources.
        
        Args:
            paths: Competition paths object
            config: Challenge config object
            allow_network: Whether network access is allowed
        
        Returns:
            Path to fetched data or None
        """
        pass
    
    @abstractmethod
    def extract(self, paths, data_path, config) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract features from raw data.
        
        Args:
            paths: Competition paths object
            data_path: Path to data file
            config: Challenge config object
        
        Returns:
            (train_features, test_features) tuple of DataFrames
        """
        pass
