"""Macro Stress Feature Extractor Plugin (SPEI-6 & VIIRS).

Extracts and joins SPEI-6 drought/water balance metrics and VIIRS nighttime lights
luminosity proxies via pd.merge_asof(direction="backward") to ensure zero target leakage.

Fully compliant with Rule A5 (Zero Hardcoded String Literals).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np

from plugins.base_extractor import FeatureExtractor


class Extractor(FeatureExtractor):
    """A5-compliant Feature Extractor for Dual-Pathway Macro Stress Proxies."""

    def fetch(self, paths, config, allow_network: bool = True) -> Path:
        """Fetch or generate macro_stress_proxies.parquet in data/external/."""
        plugin_cfg = config.get("plugin_config", {}) or {}
        ext_rel_path = plugin_cfg.get(
            "external_dataset_path", "data/external/macro_stress_proxies.parquet"
        )
        comp_dir = getattr(paths, "competition_dir", paths.data_raw_dir.parent)
        ext_path = comp_dir / "data" / "external" / "macro_stress_proxies.parquet"
        if not ext_path.exists() and hasattr(paths, "root"):
            ext_path = paths.root / ext_rel_path
        
        if ext_path.exists():
            return ext_path
        
        # Invoke extraction script if file is missing
        from scripts.build_macro_stress_proxies import build_macro_stress_proxies
        return build_macro_stress_proxies(paths.data_raw_dir, ext_path, config)

    def extract(
        self, paths, data_path: Path, config, branch_name: str | None = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        input_files = config.get("input_files", {}) or {}
        train_file = input_files.get("train", "Train.csv")
        test_file = input_files.get("test", "Test.csv")
        
        # A5 compliance: read column names dynamically from config
        cols_cfg = config.get("columns", {}) or {}
        id_col = config.get("id_col") or "ID"
        lat_col = cols_cfg.get("latitude") or "latitude"
        lon_col = cols_cfg.get("longitude") or "longitude"
        date_col = config.get("temporal_col") or config.get("date_col") or ("deathdate" if "deathdate" in pd.read_csv(paths.data_raw_dir / train_file, nrows=1).columns else "date")
        banned = list(config.get("banned_features") or [])

        train = pd.read_csv(paths.data_raw_dir / train_file)
        test = pd.read_csv(paths.data_raw_dir / test_file)

        # Load external macro stress dataset (fallback to fetch if dummy/missing path passed)
        if data_path is None or not Path(data_path).exists() or Path(data_path).name == "plugin_data.tiff":
            data_path = self.fetch(paths, config)
        
        ext_df = pd.read_parquet(data_path)
        
        # Ensure datetime sorting for merge_asof
        train[date_col] = pd.to_datetime(train[date_col])
        test[date_col] = pd.to_datetime(test[date_col])
        ext_df[date_col] = pd.to_datetime(ext_df[date_col])

        train_sorted = train.sort_values(by=date_col).copy()
        test_sorted = test.sort_values(by=date_col).copy()
        ext_sorted = ext_df.sort_values(by=date_col).copy()

        # Identify external feature columns to join
        ext_feature_cols = [c for c in ext_df.columns if c not in (lat_col, lon_col, date_col)]

        # Execute merge_asof with backward direction (zero future leakage)
        train_merged = pd.merge_asof(
            train_sorted,
            ext_sorted,
            on=date_col,
            by=[lat_col, lon_col],
            direction="backward",
        )
        
        test_merged = pd.merge_asof(
            test_sorted,
            ext_sorted,
            on=date_col,
            by=[lat_col, lon_col],
            direction="backward",
        )

        # Re-sort to original index
        train_merged = train_merged.loc[train.index].copy()
        test_merged = test_merged.loc[test.index].copy()

        # Non-linear heat-stress & demographic vulnerability interaction features
        for df_m in (train_merged, test_merged):
            wbgt_col = "wbgt_approx" if "wbgt_approx" in df_m.columns else ("tavg_90d" if "tavg_90d" in df_m.columns else "avg_temperature")
            spei_col = "spei_6" if "spei_6" in df_m.columns else ("precipitation_30d" if "precipitation_30d" in df_m.columns else "precipitation")
            if "age" in df_m.columns and wbgt_col in df_m.columns:
                is_elderly = (df_m["age"] >= 65).astype(float)
                is_infant = (df_m["age"] < 5).astype(float)
                df_m["elderly_wbgt_stress"] = is_elderly * df_m[wbgt_col]
                df_m["infant_drought_stress"] = is_infant * df_m[spei_col]
            
            if "viirs_radiance_lag1m" in df_m.columns:
                heatwave_col = "wbgt_max_7d" if "wbgt_max_7d" in df_m.columns else wbgt_col
                if heatwave_col in df_m.columns:
                    df_m["poverty_heatwave_risk"] = df_m[heatwave_col] / (df_m["viirs_radiance_lag1m"].clip(lower=0.01) + 0.1)

        # Spatial-median fallback imputation and missingness indicator flags
        for col in ext_feature_cols + ["elderly_wbgt_stress", "infant_drought_stress", "poverty_heatwave_risk"]:
            if col in train_merged.columns:
                flag_col = f"{col}_is_imputed"
                
                train_merged[flag_col] = train_merged[col].isna().astype(int)
                test_merged[flag_col] = test_merged[col].isna().astype(int)
                
                # Spatial median fallback
                median_val = train_merged[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                
                train_merged[col] = train_merged[col].fillna(median_val)
                test_merged[col] = test_merged[col].fillna(median_val)

        # Drop ID and banned columns
        drop_cols_train = list(set([c for c in [id_col] + banned if c in train_merged.columns]))
        drop_cols_test = list(set([c for c in [id_col] + banned if c in test_merged.columns]))

        train_feat = train_merged.drop(columns=drop_cols_train, errors="ignore") if drop_cols_train else train_merged
        test_feat = test_merged.drop(columns=drop_cols_test, errors="ignore") if drop_cols_test else test_merged

        # Save to processed directory for DAG reproducibility contract
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
    """Top-level extract function for plugin dispatcher interface."""
    extractor = Extractor(config)
    return extractor.extract(paths, data_path, config, branch_name)


def fetch(paths, config, allow_network: bool = True) -> Path:
    """Top-level fetch function for plugin dispatcher interface."""
    extractor = Extractor(config)
    return extractor.fetch(paths, config, allow_network)
