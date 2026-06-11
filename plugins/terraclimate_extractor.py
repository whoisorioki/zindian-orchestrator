"""TerraClimate plugin: fetch and extract TerraClimate bands.

This module follows the plugin API used by the core skill:
- fetch(paths, config, allow_network=True) -> Path to tiff
- extract(paths, tiff_path, config) -> (train_df, test_df)

This file contains the dataset-specific logic and is imported dynamically
by `skill_07_features` when `feature_extraction_plugin` is set in
`challenge_config.json`.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from zindian.config import ChallengeConfig


def fetch(paths, config: ChallengeConfig, allow_network: bool = True) -> Path:
    import pystac_client
    import planetary_computer
    import xarray as xr
    import rasterio
    from rasterio.transform import from_bounds

    tiff_path = paths.data_processed_dir / "TerraClimate_14band.tiff"
    cache_dir = paths.data_processed_dir / "tc_cache"

    if tiff_path.exists():
        return tiff_path

    if os.environ.get("ZINDIAN_DISABLE_NETWORK") or not allow_network:
        raise RuntimeError("Network fetch disabled and TerraClimate tiff missing")

    cache_dir.mkdir(parents=True, exist_ok=True)
    tiff_path.parent.mkdir(parents=True, exist_ok=True)

    def connect():
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
        collection = catalog.get_collection("terraclimate")
        asset = collection.assets["zarr-abfs"]
        ds = xr.open_dataset(asset.href, **asset.extra_fields["xarray:open_kwargs"])
        ds = ds.drop("crs", dim=None)
        # TIME_SLICE is provided by zindian.constants in the main project; read from config as fallback
        time_slice = config.get("time_slice")
        if time_slice:
            ds = ds.sel(time=slice(*time_slice))

        spatial_cfg = config.get("spatial_signal", {}) or {}
        min_lon_v = spatial_cfg.get("min_lon")
        max_lon_v = spatial_cfg.get("max_lon")
        min_lat_v = spatial_cfg.get("min_lat")
        max_lat_v = spatial_cfg.get("max_lat")
        if None in (min_lon_v, max_lon_v, min_lat_v, max_lat_v):
            raise RuntimeError("spatial_signal bounding box not populated in config")
        assert min_lon_v is not None and max_lon_v is not None and min_lat_v is not None and max_lat_v is not None
        min_lon = float(min_lon_v)
        max_lon = float(max_lon_v)
        min_lat = float(min_lat_v)
        max_lat = float(max_lat_v)

        mask_lon = (ds.lon >= min_lon) & (ds.lon <= max_lon)
        mask_lat = (ds.lat >= min_lat) & (ds.lat <= max_lat)
        return ds.where(mask_lon & mask_lat, drop=True)

    ds = connect()

    # TC_VARIABLES and TC_STATS are expected to be present in project constants
    from zindian.constants import TC_VARIABLES, TC_STATS

    bands, band_names = [], []

    MAX_RETRIES = int(config.get("fetch_retries", 5))
    RETRY_WAIT = int(config.get("fetch_retry_wait", 15))

    for var in TC_VARIABLES:
        if var not in ds:
            continue
        for stat in TC_STATS:
            key = f"{var}_{stat}"
            cache_file = cache_dir / f"{key}.npy"
            if cache_file.exists():
                bands.append(np.load(cache_file))
                band_names.append(key)
                continue
            fn = {
                "mean": lambda a: a.mean(dim="time"),
                "std": lambda a: a.std(dim="time"),
                "min": lambda a: a.min(dim="time"),
                "max": lambda a: a.max(dim="time"),
            }[stat]
            result = None
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    result = fn(ds[var]).compute().values
                    break
                except Exception:
                    if attempt == MAX_RETRIES:
                        raise
                    time.sleep(RETRY_WAIT)
                    try:
                        ds = connect()
                    except Exception:
                        pass

            if result is None:
                raise RuntimeError(f"Failed to compute {key}")
            np.save(cache_file, result)
            bands.append(result)
            band_names.append(key)

    bands_array = np.stack(bands, axis=0)
    height, width = bands_array.shape[1], bands_array.shape[2]
    # reuse spatial_cfg values for transform
    spatial_cfg = config.get("spatial_signal", {}) or {}
    min_lon_v = spatial_cfg.get("min_lon")
    max_lon_v = spatial_cfg.get("max_lon")
    min_lat_v = spatial_cfg.get("min_lat")
    max_lat_v = spatial_cfg.get("max_lat")
    assert min_lon_v is not None and max_lon_v is not None and min_lat_v is not None and max_lat_v is not None
    min_lon = float(min_lon_v)
    max_lon = float(max_lon_v)
    min_lat = float(min_lat_v)
    max_lat = float(max_lat_v)

    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)

    import rasterio

    with rasterio.open(
        tiff_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=len(band_names),
        dtype=bands_array.dtype,
        crs="EPSG:4326",
        transform=transform,
        compress="lzw",
    ) as dst:
        for i, (band, name) in enumerate(zip(bands_array, band_names)):
            dst.write(band, i + 1)
            dst.update_tags(i + 1, name=name)

    return tiff_path


def extract(paths, tiff_path: Path, config: ChallengeConfig):
    import rasterio
    from rasterio.transform import rowcol

    out_train = paths.data_processed_dir / "features_train.csv"
    out_test = paths.data_processed_dir / "features_test.csv"

    if out_train.exists() and out_test.exists():
        return pd.read_csv(out_train), pd.read_csv(out_test)

    def spiral_search(data, row, col, max_radius=10):
        h, w = data.shape[1], data.shape[2]
        for radius in range(1, max_radius + 1):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if abs(dr) != radius and abs(dc) != radius:
                        continue
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        vals = data[:, nr, nc]
                        if not np.isnan(vals).any():
                            return vals
        return np.full(data.shape[0], np.nan)

    cols_cfg = config.get("columns", {}) or {}
    lon_col = cols_cfg.get("longitude", "Longitude")
    lat_col = cols_cfg.get("latitude", "Latitude")

    def extract_df(df, src, band_names, data):
        coords = list(zip(df[lon_col], df[lat_col]))
        values = np.array(list(src.sample(coords, masked=False)), dtype=np.float64)
        nan_mask = np.isnan(values).any(axis=1)
        if nan_mask.sum() > 0:
            for i in np.where(nan_mask)[0]:
                lon, lat = coords[i]
                r, c = rowcol(src.transform, lon, lat)
                r = int(max(0, min(src.height - 1, r)))
                c = int(max(0, min(src.width - 1, c)))
                values[i] = spiral_search(data, r, c)
        return pd.concat([df.reset_index(drop=True), pd.DataFrame(values, columns=band_names)], axis=1)

    input_files = config.get("input_files", {}) or {}
    train_file = input_files.get("train", "Training_Data.csv")
    test_file = input_files.get("test", "Test.csv")

    train = pd.read_csv(paths.data_raw_dir / train_file)
    test = pd.read_csv(paths.data_raw_dir / test_file)

    with rasterio.open(tiff_path) as src:
        band_names = [src.tags(i).get("name", f"band_{i}") for i in range(1, src.count + 1)]
        data = src.read().astype(np.float64)

        train_feat = extract_df(train, src, band_names, data)
        test_feat = extract_df(test, src, band_names, data)

    nan_remaining = train_feat[band_names].isnull().sum().sum()
    if nan_remaining != 0:
        raise RuntimeError("NaNs remain after spiral search — investigate")

    paths.data_processed_dir.mkdir(parents=True, exist_ok=True)
    train_feat.to_csv(out_train, index=False)
    test_feat.to_csv(out_test, index=False)

    return train_feat, test_feat
