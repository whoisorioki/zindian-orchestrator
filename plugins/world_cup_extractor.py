"""World Cup Feature Extractor - tabular plugin.

Zero hardcoding (A5) - all file names and columns from config.
Generates: team_win_rate, has_tournament_history
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import numpy as np
from plugins.base_extractor import FeatureExtractor


def extract(
    paths: Any, tiff_path: Path, config: Any
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

    # Map country codes, team IDs and compute base features (win_rate, history)
    train = _map_keys_and_compute_features(train, auxiliary_data, config)
    test = _map_keys_and_compute_features(test, auxiliary_data, config)

    # Enrich with auxiliary data
    train_feat = _enrich(train, auxiliary_data, config)
    test_feat = _enrich(test, auxiliary_data, config)

    # Drop ID, targets, and matches_played (which is a target leak / missing in test)
    # We also drop year, team_code, region_name, tournament_id, tournament_name to prevent OOD splits and category mismatch
    drop_cols = [
        id_col,
        "matches_played",
        "team_code",
        "region_name",
        "tournament_id",
        "tournament_name",
        "year",
    ]
    train_feat = train_feat.drop(columns=drop_cols + target_cols, errors="ignore")
    test_feat = test_feat.drop(columns=drop_cols, errors="ignore")

    # Ordinal encode categorical columns
    train_feat, test_feat = _encode_categoricals(train_feat, test_feat)

    # Save
    paths.data_processed_dir.mkdir(parents=True, exist_ok=True)
    train_feat.to_csv(out_train, index=False)
    test_feat.to_csv(out_test, index=False)
    print(
        f"  ✅ Feature CSVs successfully extracted and saved ({train_feat.shape[1]} features)"
    )

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


def _map_keys_and_compute_features(
    df: pd.DataFrame, auxiliary_data: Dict[str, pd.DataFrame], config: Any
) -> pd.DataFrame:
    """Enrich dataframe with team_id, year, confederation, team_win_rate, and has_tournament_history."""
    df = df.copy()

    plugin_config = config.get("plugin_config", {}) or {}
    team_id_col = plugin_config.get("team_id_col", "team_id")
    id_col = config.get("id_col") or config.get("id_column") or "ID"

    teams_df = auxiliary_data.get("teams", pd.DataFrame())
    matches_df = auxiliary_data.get("matches", pd.DataFrame())

    # Country alias mapping to align Test.csv and teams.csv
    country_map = {
        "Czechia": "Czech Republic",
        "Turkiye": "Turkey",
        "Cote d'Ivoire": "Ivory Coast",
        "DR Congo": "Zaire",
    }

    team_name_to_id = {}
    team_name_to_conf = {}
    if not teams_df.empty:
        team_name_to_id = dict(zip(teams_df["team_name"], teams_df[team_id_col]))
        team_name_to_conf = dict(
            zip(teams_df["team_name"], teams_df["confederation_name"])
        )

    new_team_ids = {
        "Cabo Verde": "T-89",
        "Jordan": "T-90",
        "Uzbekistan": "T-91",
        "Curacao": "T-92",
    }
    new_team_confs = {
        "Cabo Verde": "Confederation of African Football",
        "Jordan": "Asian Football Confederation",
        "Uzbekistan": "Asian Football Confederation",
        "Curacao": "Confederation of North, Central American and Caribbean Association Football",
    }

    # Populate team_id
    if team_id_col not in df.columns:
        df[team_id_col] = (
            df["country"]
            .map(lambda c: country_map.get(c, c))
            .map(lambda c: team_name_to_id.get(c))
        )
        df[team_id_col] = df[team_id_col].fillna(
            df["country"].map(lambda c: new_team_ids.get(c))
        )

    # Populate year
    if "year" not in df.columns:
        try:
            df["year"] = (
                df[id_col].str.split("_").str[0].str.split("-").str[1].astype(int)
            )
        except Exception:
            df["year"] = 2026

    # Populate confederation_name
    if "confederation_name" not in df.columns:
        df["confederation_name"] = (
            df["country"]
            .map(lambda c: country_map.get(c, c))
            .map(lambda c: team_name_to_conf.get(c))
        )
        df["confederation_name"] = df["confederation_name"].fillna(
            df["country"].map(lambda c: new_team_confs.get(c))
        )

    # Helper to parse matches year
    def get_match_year(tour_id):
        try:
            return int(tour_id.split("-")[1])
        except Exception:
            return 1930

    if not matches_df.empty:
        matches_df = matches_df.copy()
        matches_df["match_year"] = matches_df["tournament_id"].map(get_match_year)

    win_rates = []
    has_histories = []

    for idx, row in df.iterrows():
        t_id = row[team_id_col]
        y = row["year"]

        if matches_df.empty:
            win_rates.append(np.nan)
            has_histories.append(0.0)
            continue

        # Filter matches played before current tournament year (temporal separation)
        hist_matches = matches_df[
            (matches_df["match_year"] < y)
            & (
                (matches_df["home_team_id"] == t_id)
                | (matches_df["away_team_id"] == t_id)
            )
        ]

        if len(hist_matches) > 0:
            outcomes = []
            for _, m in hist_matches.iterrows():
                if m["home_team_id"] == t_id:
                    outcome = (
                        1.0
                        if m["home_team_win"] == 1
                        else (0.5 if m["draw"] == 1 else 0.0)
                    )
                else:
                    outcome = (
                        1.0
                        if m["away_team_win"] == 1
                        else (0.5 if m["draw"] == 1 else 0.0)
                    )
                outcomes.append(outcome)

            win_rates.append(float(np.mean(outcomes)))
            has_histories.append(1.0)
        else:
            win_rates.append(np.nan)
            has_histories.append(0.0)

    df["team_win_rate"] = win_rates
    df["has_tournament_history"] = has_histories

    # Compute confederation fallbacks for debutant/missing teams
    for conf_name in df["confederation_name"].dropna().unique():
        conf_mask = df["confederation_name"] == conf_name
        conf_df = df[conf_mask]

        for idx, row in conf_df.iterrows():
            y = row["year"]

            # Confederation fallback: average of other teams in same conf before or at year Y
            other_conf = df[
                (df["confederation_name"] == conf_name)
                & (df["year"] <= y)
                & (df["team_win_rate"].notna())
            ]

            if len(other_conf) > 0:
                fallback_wr = other_conf["team_win_rate"].mean()
            else:
                # Global fallback
                global_before = df[(df["year"] <= y) & (df["team_win_rate"].notna())]
                if len(global_before) > 0:
                    fallback_wr = global_before["team_win_rate"].mean()
                else:
                    fallback_wr = 0.35

            if pd.isna(df.loc[idx, "team_win_rate"]):
                df.loc[idx, "team_win_rate"] = fallback_wr

    return df


def _enrich(
    df: pd.DataFrame, auxiliary_data: Dict[str, pd.DataFrame], config: Any
) -> pd.DataFrame:
    """Enrich df with features from auxiliary data."""
    df = df.copy()

    plugin_config = config.get("plugin_config", {}) or {}
    team_id_col = plugin_config.get("team_id_col", "team_id")

    # Merge teams
    if "teams" in auxiliary_data and team_id_col in df.columns:
        teams = auxiliary_data["teams"][[team_id_col, "confederation_id"]].drop_duplicates()
        df = df.merge(teams, on=team_id_col, how="left", suffixes=("", "_aux"))

        # Fill new team confederation_id
        new_team_conf_ids = {
            "T-89": "CAF",
            "T-90": "AFC",
            "T-91": "AFC",
            "T-92": "CONCACAF",
        }
        df["confederation_id"] = df["confederation_id"].fillna(
            df[team_id_col].map(lambda c: new_team_conf_ids.get(c))
        )

    # Merge confederations
    if "confederations" in auxiliary_data and "confederation_id" in df.columns:
        confed = auxiliary_data["confederations"][
            ["confederation_id", "confederation_name"]
        ].drop_duplicates()
        df = df.merge(confed, on="confederation_id", how="left", suffixes=("", "_aux"))

        # Fill new team confederation_name
        new_conf_names = {
            "CAF": "Confederation of African Football",
            "AFC": "Asian Football Confederation",
            "CONCACAF": "Confederation of North, Central American and Caribbean Association Football",
        }
        df["confederation_name_aux"] = df["confederation_name_aux"].fillna(
            df["confederation_id"].map(lambda c: new_conf_names.get(c))
        )

    return df


def _encode_categoricals(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply ordinal encoding to categorical columns consistently across train and test."""
    # Identify categorical columns
    cat_cols = [
        col
        for col in train_df.columns
        if not pd.api.types.is_numeric_dtype(train_df[col])
    ]

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

class Extractor(FeatureExtractor):
    """World Cup Extractor implementing the formal FeatureExtractor ABC."""

    def extract(
        self, paths: Any, tiff_path: Path, config: Any, branch_name: str | None = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return extract(paths, tiff_path, config)
