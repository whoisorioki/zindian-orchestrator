"""Nedbank feature extractor plugin.

Loads and aggregates transactions, financials, and demographics parquet files
using DuckDB and pandas, factorizes categorical columns, joins everything,
and saves the output to the processed data directory.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Tuple
import pandas as pd

try:
    orig_path = sys.path.copy()
    sys.path = [
        p
        for p in sys.path
        if p not in ("", ".", os.getcwd(), os.path.abspath(os.getcwd()))
    ]
    if "duckdb" in sys.modules:
        del sys.modules["duckdb"]
    import duckdb
finally:
    sys.path = orig_path

from zindian.config import ChallengeConfig
from plugins.base_extractor import FeatureExtractor


class NedbankExtractor(FeatureExtractor):
    """Nedbank-specific feature extractor implementing FeatureExtractor ABC."""

    def extract(
        self, paths: Any, tiff_path: Path, config: Any
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.extract_features(paths.data_raw_dir, config)

    def extract_features(
        self, raw_data_dir: Path, config: dict
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Extract features from Nedbank raw data."""
        from zindian.config import ChallengeConfig

        _paths = type(
            "Paths",
            (),
            {
                "data_raw_dir": raw_data_dir,
                "data_processed_dir": raw_data_dir.parent / "processed",
            },
        )()
        # Load ChallengeConfig from the default path; pass config dict as fallback
        try:
            _cfg = ChallengeConfig.load()
        except Exception:
            # Construct a minimal ChallengeConfig from the provided dict
            import tempfile
            import json as _json
            from pathlib import Path as _Path

            _tmp = _Path(tempfile.mktemp(suffix=".json"))
            _tmp.write_text(_json.dumps(config))
            _cfg = ChallengeConfig(path=_tmp, _data=config)
        tiff_path = raw_data_dir / "plugin_data.tiff"
        return extract(_paths, tiff_path, _cfg)


def fetch(paths, config: ChallengeConfig, allow_network: bool = True) -> Path:
    """Touch dummy tiff file to satisfy plugin interface contract and return it."""
    tiff_path = paths.data_processed_dir / "plugin_data.tiff"
    tiff_path.parent.mkdir(parents=True, exist_ok=True)
    tiff_path.touch()
    return tiff_path


def extract(
    paths, tiff_path: Path, config: ChallengeConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw datasets, aggregate via DuckDB, join, and save to processed directory."""
    out_train = paths.data_processed_dir / "features_train.csv"
    out_test = paths.data_processed_dir / "features_test.csv"

    if out_train.exists() and out_test.exists():
        print("  ✅ Feature CSVs already exist — skipping extraction")
        return pd.read_csv(out_train), pd.read_csv(out_test)

    raw_dir = paths.data_raw_dir

    print("Initializing DuckDB...")
    con = duckdb.connect()

    print("Aggregating transactions...")
    txn_df = con.execute(f"""
        SELECT
            UniqueID,
            COUNT(*) AS txn_count,
            SUM(TransactionAmount) AS txn_amount_sum,
            AVG(TransactionAmount) AS txn_amount_avg,
            MIN(TransactionAmount) AS txn_amount_min,
            MAX(TransactionAmount) AS txn_amount_max,
            SUM(CASE WHEN IsDebitCredit = 'D' THEN TransactionAmount ELSE 0 END) AS debit_amount_sum,
            SUM(CASE WHEN IsDebitCredit = 'C' THEN TransactionAmount ELSE 0 END) AS credit_amount_sum,
            AVG(StatementBalance) AS statement_balance_avg
        FROM read_parquet('{raw_dir}/transactions_features.parquet')
        GROUP BY UniqueID
    """).df()

    print("Aggregating financials...")
    fin_df = con.execute(f"""
        SELECT
            UniqueID,
            SUM(NetInterestIncome) AS nii_sum,
            AVG(NetInterestIncome) AS nii_avg,
            SUM(NetInterestRevenue) AS nir_sum,
            AVG(NetInterestRevenue) AS nir_avg
        FROM read_parquet('{raw_dir}/financials_features.parquet')
        GROUP BY UniqueID
    """).df()

    print("Loading demographics...")
    demo_df = pd.read_parquet(f"{raw_dir}/demographics_clean.parquet")

    # Calculate Age from BirthDate
    demo_df["Age"] = (
        2015 - pd.to_datetime(demo_df["BirthDate"], errors="coerce").dt.year
    )
    demo_df["Age"] = demo_df["Age"].fillna(demo_df["Age"].median())
    demo_df = demo_df.drop(columns=["BirthDate"])

    # Factorize non-numeric columns
    for col in demo_df.columns:
        if col != "UniqueID" and not pd.api.types.is_numeric_dtype(demo_df[col]):
            demo_df[col] = pd.factorize(demo_df[col])[0]

    # Ingest target, id columns from config
    input_files = config.get("input_files", {}) or {}
    train_file = input_files.get("train", "Train.csv")
    test_file = input_files.get("test", "Test.csv")

    train = pd.read_csv(raw_dir / train_file)
    test = pd.read_csv(raw_dir / test_file)

    # Merge demographics, transactions, financials
    train_feat = train.merge(demo_df, on="UniqueID", how="left")
    test_feat = test.merge(demo_df, on="UniqueID", how="left")

    train_feat = train_feat.merge(txn_df, on="UniqueID", how="left")
    test_feat = test_feat.merge(txn_df, on="UniqueID", how="left")

    train_feat = train_feat.merge(fin_df, on="UniqueID", how="left")
    test_feat = test_feat.merge(fin_df, on="UniqueID", how="left")

    # Fill NaNs with 0
    target_col = (
        config.get("target_col") or config.get("target_column") or "next_3m_txn_count"
    )
    for col in train_feat.columns:
        if col not in ("UniqueID", target_col):
            train_feat[col] = train_feat[col].fillna(0.0)
            test_feat[col] = test_feat[col].fillna(0.0)

    paths.data_processed_dir.mkdir(parents=True, exist_ok=True)
    train_feat.to_csv(out_train, index=False)
    test_feat.to_csv(out_test, index=False)
    print("  ✅ Feature CSVs successfully extracted and saved to processed folder")

    return train_feat, test_feat


from typing import Any, Tuple
from plugins.base_extractor import FeatureExtractor


class Extractor(FeatureExtractor):
    """Nedbank Extractor implementing the formal FeatureExtractor ABC."""

    def extract(
        self, paths: Any, tiff_path: Path, config: Any
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return extract(paths, tiff_path, config)

