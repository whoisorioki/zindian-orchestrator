"""World Cup Feature Extractor - tabular plugin.

Zero hardcoding (A5) - all file names and columns from config.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple


def extract(paths: Any, tiff_path: Path, config: Any) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract features from World Cup raw data.
    
    Args:
        paths: CompetitionPaths object
        tiff_path: Unused for tabular data
        config: ChallengeConfig object
        
    Returns:
        Tuple of (train_features, test_features)
    """
    out_train = paths.data_processed_dir / "features_train.csv"
    out_test = paths.data_processed_dir / "features_test.csv"
    
    if out_train.exists() and out_test.exists():
        print("  ✅ Feature CSVs already exist — skipping extraction")
        return pd.read_csv(out_train), pd.read_csv(out_test)
    
    # Load train and test
    input_files = config.get("input_files", {}) or {}
    train_file = input_files.get("train", "Train.csv")
    test_file = input_files.get("test", "Test.csv")
    
    train = pd.read_csv(paths.data_raw_dir / train_file)
    test = pd.read_csv(paths.data_raw_dir / test_file)
    
    # Get ID and target columns to drop
    id_col = config.get("id_col", "ID")
    target_cols = []
    target_config = config.get("target_config")
    if target_config:
        targets = target_config.get("targets", [])
        target_cols = [t["name"] for t in targets]
    
    # Load auxiliary data
    auxiliary_data = _load_auxiliary(paths, config)
    
    # Enrich with auxiliary data
    train_feat = _enrich(train, auxiliary_data, config)
    test_feat = _enrich(test, auxiliary_data, config)
    
    # Drop ID and targets
    train_feat = train_feat.drop(columns=[id_col] + target_cols, errors="ignore")
    test_feat = test_feat.drop(columns=[id_col], errors="ignore")
    
    # Ordinal encode categorical columns
    train_feat, test_feat = _encode_categoricals(train_feat, test_feat)
    
    # Save
    paths.data_processed_dir.mkdir(parents=True, exist_ok=True)
    train_feat.to_csv(out_train, index=False)
    test_feat.to_csv(out_test, index=False)
    print(f"  ✅ Feature CSVs successfully extracted and saved ({train_feat.shape[1]} features)")
    
    return train_feat, test_feat


def _load_auxiliary(paths: Any, config: Any) -> Dict[str, pd.DataFrame]:
    """Load auxiliary data files from manifest."""
    auxiliary_data = {}
    plugin_config = config.get("plugin_config", {}) or {}
    file_manifest = plugin_config.get("file_manifest", {}) or {}
    
    data_dir = paths.data_raw_dir / "data"
    
    for key, filename in file_manifest.items():
        filepath = data_dir / filename
        if filepath.exists():
            auxiliary_data[key] = pd.read_csv(filepath)
    
    return auxiliary_data


def _enrich(df: pd.DataFrame, auxiliary_data: Dict[str, pd.DataFrame], config: Any) -> pd.DataFrame:
    """Enrich df with features from auxiliary data."""
    df = df.copy()
    
    plugin_config = config.get("plugin_config", {}) or {}
    team_id_col = plugin_config.get("team_id_col", "team_id")
    
    # Merge teams
    if "teams" in auxiliary_data and team_id_col in df.columns:
        teams = auxiliary_data["teams"][[team_id_col, "confederation_id"]].drop_duplicates()
        df = df.merge(teams, on=team_id_col, how="left", suffixes=("", "_aux"))
    
    # Merge confederations
    if "confederations" in auxiliary_data and "confederation_id" in df.columns:
        confed = auxiliary_data["confederations"][["confederation_id", "confederation_name"]].drop_duplicates()
        df = df.merge(confed, on="confederation_id", how="left", suffixes=("", "_aux"))
    
    return df


def _encode_categoricals(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply ordinal encoding to categorical columns consistently across train and test."""
    # Identify categorical columns
    cat_cols = [col for col in train_df.columns if train_df[col].dtype == 'object']
    
    if not cat_cols:
        return train_df, test_df
    
    # Combine for consistent encoding
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    train_len = len(train_df)
    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    
    # Encode each categorical column
    for col in cat_cols:
        combined[col] = pd.factorize(combined[col])[0]
    
    # Split back
    train_encoded = combined.iloc[:train_len].reset_index(drop=True)
    test_encoded = combined.iloc[train_len:].reset_index(drop=True)
    
    return train_encoded, test_encoded
