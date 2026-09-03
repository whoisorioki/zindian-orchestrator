"""Downloader for Google Earth Engine VIIRS Nighttime Lights & MODIS NDVI (Static + Temporal Date-Matched)."""

import ee
import json
from pathlib import Path
import pandas as pd
import numpy as np

def download_gee_data(config_path: Path):
    project_id = "project-maps-474408"
    print(f"\n[1/3] Initializing Google Earth Engine (Project: {project_id})...")
    ee.Initialize(project=project_id)
    
    with open(config_path) as f:
        config = json.load(f)
        
    comp_dir = config_path.parent
    raw_dir = comp_dir / "data" / "raw"
    ext_dir = comp_dir / "data" / "external" / "raw_rasters"
    ext_dir.mkdir(parents=True, exist_ok=True)
    
    cols = config.get("columns", {}) or {}
    lat_col = cols.get("latitude") or "latitude"
    lon_col = cols.get("longitude") or "longitude"
    date_col = config.get("temporal_col") or config.get("date_col") or "deathdate"
    
    train = pd.read_csv(raw_dir / "Train.csv")
    test = pd.read_csv(raw_dir / "Test.csv")
    combined = pd.concat([train, test], ignore_index=True)
    
    print(f"\n[2/3] Extracting Static Spatial & Temporal Remote Sensing for {len(combined)} rows across 55 locations...")
    
    # 1. Static Composites across 55 unique locations
    unique_locs = combined[[lat_col, lon_col]].drop_duplicates().reset_index(drop=True)
    features_static = []
    for idx, row in unique_locs.iterrows():
        lat, lon = float(row[lat_col]), float(row[lon_col])
        geom = ee.Geometry.Point([lon, lat])
        features_static.append(ee.Feature(geom, {'loc_id': idx, lat_col: lat, lon_col: lon}))
        
    pts_fc_static = ee.FeatureCollection(features_static)
    
    viirs_img_static = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG").select("avg_rad").median()
    viirs_sampled = viirs_img_static.reduceRegions(collection=pts_fc_static, reducer=ee.Reducer.mean(), scale=500).getInfo()
    
    modis_img_static = ee.ImageCollection("MODIS/061/MOD13Q1").select("NDVI").median().multiply(0.0001)
    modis_sampled = modis_img_static.reduceRegions(collection=pts_fc_static, reducer=ee.Reducer.mean(), scale=250).getInfo()
    
    results_static = []
    for vf, mf in zip(viirs_sampled['features'], modis_sampled['features']):
        props_v = vf['properties']
        props_m = mf['properties']
        results_static.append({
            lat_col: props_v[lat_col],
            lon_col: props_v[lon_col],
            'viirs_radiance_static': props_v.get('mean', 0.0),
            'modis_ndvi_static': props_m.get('mean', 0.0)
        })
        
    df_static = pd.DataFrame(results_static)
    out_static = ext_dir / "gee_viirs_ndvi_extracted.csv"
    df_static.to_csv(out_static, index=False)
    print(f"  ✓ Saved static baseline remote sensing features to {out_static}")
    
    # 2. Temporal Date-Matched MODIS & VIIRS per location-month
    print("\n[3/3] Extracting Date-Matched Temporal MODIS NDVI & VIIRS time-series...")
    combined['date'] = pd.to_datetime(combined[date_col])
    combined['year_month'] = combined['date'].dt.to_period('M')
    loc_months = combined[[lat_col, lon_col, 'year_month']].drop_duplicates().reset_index(drop=True)
    
    print(f"  Total unique location-month target windows: {len(loc_months)}")
    
    # Process in batches of 500 location-month points to avoid EE payload overflow
    temporal_records = []
    batch_size = 500
    for b in range(0, len(loc_months), batch_size):
        batch = loc_months.iloc[b:b+batch_size]
        ee_features = []
        for idx, row in batch.iterrows():
            lat, lon = float(row[lat_col]), float(row[lon_col])
            ym_str = str(row['year_month'])
            start_date = f"{ym_str}-01"
            end_date = str((pd.Period(ym_str) + 1).start_time.date())
            geom = ee.Geometry.Point([lon, lat])
            ee_features.append(ee.Feature(geom, {
                'batch_id': idx,
                lat_col: lat,
                lon_col: lon,
                'year_month': ym_str,
                'start_date': start_date,
                'end_date': end_date
            }))
            
        fc = ee.FeatureCollection(ee_features)
        
        # Sample MODIS NDVI for matching month
        modis_col = ee.ImageCollection("MODIS/061/MOD13Q1").select("NDVI")
        def sample_modis(feature):
            s_date = feature.get('start_date')
            e_date = feature.get('end_date')
            sub_col = modis_col.filterDate(s_date, e_date)
            mean_ndvi = ee.Algorithms.If(
                sub_col.size().gt(0),
                sub_col.mean().reduceRegion(reducer=ee.Reducer.mean(), geometry=feature.geometry(), scale=250).get('NDVI'),
                None
            )
            return feature.set('modis_ndvi_temporal', mean_ndvi)
            
        fc_modis = fc.map(sample_modis).getInfo()
        
        for f_item in fc_modis['features']:
            p = f_item['properties']
            raw_ndvi = p.get('modis_ndvi_temporal')
            val_ndvi = float(raw_ndvi * 0.0001) if raw_ndvi is not None else np.nan
            temporal_records.append({
                lat_col: p[lat_col],
                lon_col: p[lon_col],
                date_col: p['start_date'],
                'year_month': p['year_month'],
                'modis_ndvi_temporal': val_ndvi
            })
            
        print(f"  Processed temporal batch {b//batch_size + 1}/{(len(loc_months)-1)//batch_size + 1}...")
        
    df_temporal = pd.DataFrame(temporal_records)
    out_temporal = ext_dir / "gee_temporal_ndvi_extracted.csv"
    df_temporal.to_csv(out_temporal, index=False)
    print(f"\n[SUCCESS] Saved Date-Matched Temporal GEE Features to {out_temporal}")
    print(df_temporal.head(10))
    return out_temporal

if __name__ == "__main__":
    download_gee_data(Path("competitions/climate-risk-health-prediction-challenge/challenge_config.json"))

