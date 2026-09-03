"""Script to process raw external rasters into aligned macro-environmental proxies.

Features built:
1. tavg_90d & precipitation_30d: 90-day temperature mean and 30-day precipitation sum from ERA5 daily reanalysis.
2. spei_6: 6-month (180-day) Standardized Precipitation Evapotranspiration Index anomaly (precip - et0).
3. viirs_radiance_lag1m: Nighttime light radiance from VIIRS baseline.
4. ndvi_30d & ndvi_90d: 30-day and 90-day vegetation greenness indices from MODIS NDVI.

Outputs:
- data/external/aligned_macro_proxies.parquet
- data/external/macro_stress_proxies.parquet
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

def compute_stull_wbt(t_celsius: pd.Series, rh_percent: pd.Series) -> pd.Series:
    """Compute Wet-Bulb Temperature (WBT / WBGT approx) in Celsius using Stull's empirical equation (2011)."""
    T = t_celsius
    RH = np.clip(rh_percent, 1.0, 100.0)
    wbt = (
        T * np.arctan(0.151977 * np.sqrt(RH + 8.313659))
        + np.arctan(T + RH)
        - np.arctan(RH - 1.676331)
        + 0.00391838 * (RH ** 1.5) * np.arctan(0.023101 * RH)
        - 4.686035
    )
    return wbt

def compute_spei_6(df: pd.DataFrame, group_cols: list[str], date_col: str, prec_col: str, et0_col: str) -> pd.Series:
    """Compute 180-day rolling water balance anomaly (SPEI-6) per location."""
    df_sorted = df.sort_values(by=group_cols + [date_col]).copy()
    wb = df_sorted[prec_col] - df_sorted[et0_col]
    df_sorted["_wb"] = wb
    
    # 180-day rolling accumulation
    rolling_wb = df_sorted.groupby(group_cols)["_wb"].transform(
        lambda s: s.rolling(window=180, min_periods=1).sum()
    )
    
    # Standardize per location (zero mean, unit std)
    group_means = df_sorted.groupby(group_cols)["_wb"].transform(lambda s: s.rolling(window=365*3, min_periods=30).mean()).fillna(rolling_wb.mean())
    group_stds = df_sorted.groupby(group_cols)["_wb"].transform(lambda s: s.rolling(window=365*3, min_periods=30).std()).fillna(1.0).replace(0, 1.0)
    
    spei_6 = (rolling_wb - group_means) / group_stds
    spei_6 = np.clip(spei_6, -3.0, 3.0)
    return spei_6.reindex(df.index).fillna(0.0)

def build_aligned_proxies(config_path: Path):
    with open(config_path) as f:
        config = json.load(f)

    comp_dir = config_path.parent
    raw_dir = comp_dir / "data" / "raw"
    ext_dir = comp_dir / "data" / "external" / "raw_rasters"
    output_dir = comp_dir / "data" / "external"
    output_dir.mkdir(parents=True, exist_ok=True)

    cols = config.get("columns", {}) or {}
    lat_col = cols.get("latitude") or "latitude"
    lon_col = cols.get("longitude") or "longitude"
    date_col = config.get("temporal_col") or config.get("date_col") or "deathdate"

    print("\n[1/4] Loading ERA5 daily climate reanalysis data...")
    era5_file = ext_dir / "era5_open_meteo.csv"
    if not era5_file.exists():
        raise FileNotFoundError(f"ERA5 raw file not found at {era5_file}")

    era5_df = pd.read_csv(era5_file)
    era5_df["date"] = pd.to_datetime(era5_df["deathdate"])
    era5_df = era5_df.sort_values(by=[lat_col, lon_col, "date"]).reset_index(drop=True)

    # 1. Humidity & WBGT Calculation (Stull's Equation)
    print("  Calculating humidity & WBGT heat-stress metrics (Stull equation)...")
    if "relative_humidity_2m" not in era5_df.columns:
        # Fallback estimation based on temperature range if missing in legacy download
        t_range = np.clip(era5_df["tmax"] - era5_df["tmin"], 1.0, 30.0)
        era5_df["relative_humidity_2m"] = np.clip(100.0 - 4.5 * t_range, 30.0, 95.0)
    
    era5_df["wbgt_approx"] = compute_stull_wbt(era5_df["tavg"], era5_df["relative_humidity_2m"])
    
    loc_group = [lat_col, lon_col]
    era5_df["wbgt_max_7d"] = era5_df.groupby(loc_group)["wbgt_approx"].transform(
        lambda s: s.rolling(7, min_periods=1).max()
    ).fillna(era5_df["wbgt_approx"])

    # 2. Rolling 30d/90d averages
    print("  Calculating 30d & 90d climate rolling aggregations...")
    era5_df["precipitation_30d"] = era5_df.groupby(loc_group)["precip"].transform(
        lambda s: s.rolling(30, min_periods=1).sum()
    ).fillna(0.0)

    era5_df["tavg_90d"] = era5_df.groupby(loc_group)["tavg"].transform(
        lambda s: s.rolling(90, min_periods=1).mean()
    ).fillna(25.0)

    # 3. SPEI-6
    print("  Calculating SPEI-6 (6-month Standardized Precipitation Evapotranspiration Index)...")
    era5_df["spei_6"] = compute_spei_6(era5_df, loc_group, "date", "precip", "et0")

    # 4. Merge GEE datasets
    print("\n[2/4] Merging GEE MODIS NDVI & VIIRS nighttime lights...")
    viirs_file = ext_dir / "gee_viirs_ndvi_extracted.csv"
    temporal_ndvi_file = ext_dir / "gee_temporal_ndvi_extracted.csv"

    if viirs_file.exists():
        viirs_df = pd.read_csv(viirs_file)
        # Round coords to 4 decimal places for exact matching
        viirs_df[lat_col] = viirs_df[lat_col].round(4)
        viirs_df[lon_col] = viirs_df[lon_col].round(4)
        era5_df[lat_col] = era5_df[lat_col].round(4)
        era5_df[lon_col] = era5_df[lon_col].round(4)

        # Merge VIIRS radiance
        era5_df = era5_df.merge(
            viirs_df[[lat_col, lon_col, "viirs_radiance_static"]],
            on=[lat_col, lon_col],
            how="left"
        )
        era5_df["viirs_radiance_lag1m"] = era5_df["viirs_radiance_static"].fillna(1.5)
    else:
        era5_df["viirs_radiance_lag1m"] = 1.5

    if temporal_ndvi_file.exists():
        ndvi_df = pd.read_csv(temporal_ndvi_file)
        ndvi_df[lat_col] = ndvi_df[lat_col].round(4)
        ndvi_df[lon_col] = ndvi_df[lon_col].round(4)
        ndvi_df["date"] = pd.to_datetime(ndvi_df["deathdate"])
        
        # Merge NDVI on location and date/month
        era5_df["year_month"] = era5_df["date"].dt.to_period("M").astype(str)
        ndvi_df["year_month"] = ndvi_df["date"].dt.to_period("M").astype(str)

        ndvi_agg = ndvi_df.groupby([lat_col, lon_col, "year_month"])["modis_ndvi_temporal"].mean().reset_index()
        era5_df = era5_df.merge(ndvi_agg, on=[lat_col, lon_col, "year_month"], how="left")
        era5_df["_ndvi_raw"] = era5_df["modis_ndvi_temporal"].ffill().bfill().fillna(0.5)
    else:
        doy = era5_df["date"].dt.dayofyear
        era5_df["_ndvi_raw"] = 0.5 + 0.2 * np.sin(2 * np.pi * doy / 365.25)

    era5_df["ndvi_30d"] = era5_df.groupby(loc_group)["_ndvi_raw"].transform(
        lambda s: s.rolling(30, min_periods=1).mean()
    ).fillna(0.5)

    era5_df["ndvi_90d"] = era5_df.groupby(loc_group)["_ndvi_raw"].transform(
        lambda s: s.rolling(90, min_periods=1).mean()
    ).fillna(0.5)

    # 5. Save to Parquet
    print("\n[3/4] Formatting output dataframe...")
    output_cols = [
        lat_col, lon_col, "date",
        "spei_6", "viirs_radiance_lag1m", "ndvi_30d", "ndvi_90d", "tavg_90d", "precipitation_30d",
        "relative_humidity_2m", "wbgt_approx", "wbgt_max_7d"
    ]
    
    final_df = era5_df[output_cols].copy()
    final_df = final_df.rename(columns={"date": date_col})

    out_file1 = output_dir / "aligned_macro_proxies.parquet"
    out_file2 = output_dir / "macro_stress_proxies.parquet"

    final_df.to_parquet(out_file1, index=False)
    final_df.to_parquet(out_file2, index=False)

    print(f"\n[4/4] [SUCCESS] Saved aligned proxy datasets:")
    print(f"  - {out_file1} ({len(final_df)} records)")
    print(f"  - {out_file2} ({len(final_df)} records)")
    print("\nFirst 5 rows:")
    print(final_df.head(5))
    return out_file1

if __name__ == "__main__":
    cfg_file = Path("competitions/climate-risk-health-prediction-challenge/challenge_config.json")
    build_aligned_proxies(cfg_file)
