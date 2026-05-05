"""
Skill 07 — Feature Engineering
Competition-aware, config-driven feature engineering.
Runs 1 anchor + 9 isolated variants per round.

Governed by:
  - competitions/<slug>/challenge_config.json
  - competitions/<slug>/SKILL_STATE.json

Writes to:
  - competitions/<slug>/data/processed/TerraClimate_14band.tiff
  - competitions/<slug>/data/processed/features_train.csv
  - competitions/<slug>/data/processed/features_test.csv
  - competitions/<slug>/SKILL_STATE.json
  - competitions/<slug>/reports/feature_round_<N>.md
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score

from zindian.config import ChallengeConfig
from zindian.paths import resolve_competition_paths
from zindian.state import SkillStateStore

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────
SEED        = 42
N_SPLITS    = 5
MIN_DELTA   = 0.005          # gate: variant must beat anchor by ≥ 0.5% AUC
MAX_RETRIES = 5
RETRY_WAIT  = 15

TC_VARIABLES = [
    "aet", "def", "pdsi", "pet", "ppt",
    "q", "soil", "srad", "swe",
    "tmax", "tmin", "vap", "vpd",
]
TC_STATS     = ["mean", "std", "min", "max"]
TC_BAND_NAMES = [f"{v}_{s}" for v in TC_VARIABLES for s in TC_STATS]  # 52 bands

# Bbox — SE Australia (confirmed 100% coverage of training data)
MIN_LON, MAX_LON = 139.94, 151.48
MIN_LAT, MAX_LAT = -39.74, -30.92
TIME_SLICE       = ("2011-01-01", "2021-12-01")   # 10-year modern normal


# ── State helpers ─────────────────────────────────────────────────────────────

def _write_state(state: dict, path: Path) -> None:
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
        json.dump(state, tmp, indent=2)
        tmp_path = tmp.name
    os.replace(tmp_path, path)


# ── Phase A: TerraClimate Fetch ───────────────────────────────────────────────

def fetch_terraclimate(paths) -> Path:
    """
    Fetch all 13 TerraClimate variables × 4 stats = 52 bands.
    Saves to data/processed/TerraClimate_14band.tiff.
    Uses per-band checkpointing to survive Azure timeouts.
    Returns path to tiff.
    """
    import pystac_client
    import planetary_computer
    import xarray as xr
    import rasterio
    from rasterio.transform import from_bounds

    tiff_path = paths.competition_dir / "data/processed/TerraClimate_14band.tiff"
    cache_dir = paths.competition_dir / "data/processed/tc_cache"

    if tiff_path.exists():
        print(f"  ✅ Tiff already exists ({tiff_path.stat().st_size/1024**2:.1f}MB) — skipping fetch")
        return tiff_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    tiff_path.parent.mkdir(parents=True, exist_ok=True)

    def connect():
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace
        )
        collection = catalog.get_collection("terraclimate")
        asset      = collection.assets["zarr-abfs"]
        ds = xr.open_dataset(asset.href, **asset.extra_fields["xarray:open_kwargs"])
        ds = ds.drop("crs", dim=None)
        ds = ds.sel(time=slice(*TIME_SLICE))
        mask_lon = (ds.lon >= MIN_LON) & (ds.lon <= MAX_LON)
        mask_lat = (ds.lat >= MIN_LAT) & (ds.lat <= MAX_LAT)
        return ds.where(mask_lon & mask_lat, drop=True)

    print("  Connecting to Planetary Computer...")
    ds = connect()
    print(f"  ✅ Connected — dims: {dict(ds.dims)}")

    bands, band_names = [], []

    for var in TC_VARIABLES:
        if var not in ds:
            print(f"  ⚠️  {var} not in dataset — skipping")
            continue
        for stat in TC_STATS:
            key        = f"{var}_{stat}"
            cache_file = cache_dir / f"{key}.npy"

            if cache_file.exists():
                print(f"  ✅ {key} — from cache")
                bands.append(np.load(cache_file))
                band_names.append(key)
                continue

            print(f"  ⏳ {key}...", end=" ", flush=True)
            fn = {"mean": lambda a: a.mean(dim="time"),
                  "std":  lambda a: a.std(dim="time"),
                  "min":  lambda a: a.min(dim="time"),
                  "max":  lambda a: a.max(dim="time")}[stat]

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    result = fn(ds[var]).compute().values
                    break
                except Exception:
                    if attempt == MAX_RETRIES:
                        raise
                    print(f"\n  ⚠️  Attempt {attempt} failed — reconnecting in {RETRY_WAIT}s...")
                    time.sleep(RETRY_WAIT)
                    try:
                        ds = connect()
                    except Exception:
                        pass

            np.save(cache_file, result)
            bands.append(result)
            band_names.append(key)
            print("done ✅")

    bands_array = np.stack(bands, axis=0)
    height, width = bands_array.shape[1], bands_array.shape[2]
    transform = from_bounds(MIN_LON, MIN_LAT, MAX_LON, MAX_LAT, width, height)

    with rasterio.open(
        tiff_path, "w", driver="GTiff",
        height=height, width=width,
        count=len(band_names),
        dtype=bands_array.dtype,
        crs="EPSG:4326", transform=transform, compress="lzw"
    ) as dst:
        for i, (band, name) in enumerate(zip(bands_array, band_names)):
            dst.write(band, i + 1)
            dst.update_tags(i + 1, name=name)

    print(f"  ✅ Tiff written → {tiff_path}")
    return tiff_path


# ── Phase B: Feature Extraction ───────────────────────────────────────────────

def extract_features(paths, tiff_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract TerraClimate features at each lat/lon point.
    Uses spiral nearest-neighbour search for off-grid (coastal) points.
    Returns (train_features, test_features).
    """
    import rasterio
    from rasterio.transform import rowcol

    out_train = paths.competition_dir / "data/processed/features_train.csv"
    out_test  = paths.competition_dir / "data/processed/features_test.csv"

    if out_train.exists() and out_test.exists():
        print("  ✅ Feature CSVs already exist — skipping extraction")
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

    def extract(df, src, band_names, data):
        coords = list(zip(df["Longitude"], df["Latitude"]))
        values = np.array(list(src.sample(coords, masked=False)), dtype=np.float64)
        nan_mask = np.isnan(values).any(axis=1)
        if nan_mask.sum() > 0:
            print(f"  Fixing {nan_mask.sum()} off-grid points via spiral search...")
            for i in np.where(nan_mask)[0]:
                lon, lat = coords[i]
                r, c = rowcol(src.transform, lon, lat)
                r = int(max(0, min(src.height - 1, r)))
                c = int(max(0, min(src.width  - 1, c)))
                values[i] = spiral_search(data, r, c)
        return pd.concat([
            df.reset_index(drop=True),
            pd.DataFrame(values, columns=band_names)
        ], axis=1)

    train = pd.read_csv(paths.data_raw_dir / "Training_Data.csv")
    test  = pd.read_csv(paths.data_raw_dir / "Test.csv")

    with rasterio.open(tiff_path) as src:
        band_names = [src.tags(i).get("name", f"band_{i}") for i in range(1, src.count + 1)]
        data       = src.read().astype(np.float64)

        print(f"  Extracting {len(train)} train points...")
        train_feat = extract(train, src, band_names, data)

        print(f"  Extracting {len(test)} test points...")
        test_feat  = extract(test, src, band_names, data)

    nan_remaining = train_feat[band_names].isnull().sum().sum()
    print(f"  NaNs remaining: {nan_remaining}")
    assert nan_remaining == 0, "NaNs remain after spiral search — investigate"

    train_feat.to_csv(out_train, index=False)
    test_feat.to_csv(out_test, index=False)
    print(f"  ✅ features_train.csv → {out_train}")
    print(f"  ✅ features_test.csv  → {out_test}")

    return train_feat, test_feat


# ── Phase C: Variant Training ─────────────────────────────────────────────────

def train_variant(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    variant_name: str,
    anchor_auc: float,
    seed: int = SEED,
) -> dict:
    """
    Train one LightGBM variant and evaluate against anchor gate.
    Returns result dict with status, AUC, F1, threshold, delta.
    """
    np.random.seed(seed)

    TARGET = "Occurrence Status"
    X      = train[feature_cols].values
    y      = train[TARGET].values
    X_test = test[feature_cols].values

    skf        = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    oof_probs  = np.zeros(len(y))
    test_probs = np.zeros(len(test))

    print(f"\n  Training {variant_name} ({len(feature_cols)} features)...")
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):

        # ── variant-13: tuned LightGBM ────────────────────
        if variant_name == 'variant-13':
            model = lgb.LGBMClassifier(
                n_estimators=1000, learning_rate=0.02,
                num_leaves=63, min_child_samples=20,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=0.1,
                random_state=SEED, verbose=-1
            )
            model.fit(
                X[tr_idx], y[tr_idx],
                eval_set=[(X[val_idx], y[val_idx])],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(-1)]
            )

        # ── variant-14: Random Forest ─────────────────────
        elif variant_name == 'variant-14':
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=500, max_depth=None,
                min_samples_leaf=2, max_features='sqrt',
                random_state=SEED, n_jobs=-1
            )
            model.fit(X[tr_idx], y[tr_idx])

        # ── variant-18: XGBoost ──────────────────────────
        elif variant_name == 'variant-18':
            from xgboost import XGBClassifier
            model = XGBClassifier(
                n_estimators=500, learning_rate=0.05,
                max_depth=6, subsample=0.8,
                colsample_bytree=0.8, use_label_encoder=False,
                eval_metric="logloss", random_state=seed,
                verbosity=0, n_jobs=-1
            )
            model.fit(
                X[tr_idx], y[tr_idx],
                eval_set=[(X[val_idx], y[val_idx])],
                verbose=False
            )

        # ── variant-19: LightGBM larger trees ────────────
        elif variant_name == 'variant-19':
            model = lgb.LGBMClassifier(
                n_estimators=1000, learning_rate=0.02,
                num_leaves=127, max_depth=8,
                min_child_samples=10,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.05, reg_lambda=0.05,
                random_state=seed, verbose=-1
            )
            model.fit(
                X[tr_idx], y[tr_idx],
                eval_set=[(X[val_idx], y[val_idx])],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(-1)]
            )

        # ── variant-25: LGB + RF blend ────────────────────
        elif variant_name == 'variant-25':
            from sklearn.ensemble import RandomForestClassifier
            # LGB probabilities
            lgb_model = lgb.LGBMClassifier(
                n_estimators=500, learning_rate=0.05,
                num_leaves=31, random_state=seed, verbose=-1
            )
            lgb_model.fit(
                X[tr_idx], y[tr_idx],
                eval_set=[(X[val_idx], y[val_idx])],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)]
            )
            # RF probabilities
            rf_model = RandomForestClassifier(
                n_estimators=500, max_depth=None,
                min_samples_leaf=2, max_features="sqrt",
                random_state=seed, n_jobs=-1
            )
            rf_model.fit(X[tr_idx], y[tr_idx])
            # Average blend
            lgb_val  = lgb_model.predict_proba(X[val_idx])[:, 1]
            rf_val   = rf_model.predict_proba(X[val_idx])[:, 1]
            lgb_test = lgb_model.predict_proba(X_test)[:, 1]
            rf_test  = rf_model.predict_proba(X_test)[:, 1]
            oof_probs[val_idx]  = 0.5 * lgb_val + 0.5 * rf_val
            test_probs         += (0.5 * lgb_test + 0.5 * rf_test) / N_SPLITS
            fold_auc = roc_auc_score(y[val_idx], oof_probs[val_idx])
            print(f"    Fold {fold+1}: AUC={fold_auc:.5f}")
            continue

        # ── variant-26: per-fold threshold optimization ───────
        elif variant_name == 'variant-26':
            model = lgb.LGBMClassifier(
                n_estimators=500, learning_rate=0.05,
                num_leaves=31, random_state=seed, verbose=-1
            )
            model.fit(
                X[tr_idx], y[tr_idx],
                eval_set=[(X[val_idx], y[val_idx])],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)]
            )

        # ── default: standard LightGBM ────────────────────
        else:
            model = lgb.LGBMClassifier(
                n_estimators=500, learning_rate=0.05,
                num_leaves=31, random_state=SEED, verbose=-1
            )
            model.fit(
                X[tr_idx], y[tr_idx],
                eval_set=[(X[val_idx], y[val_idx])],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)]
            )

        oof_probs[val_idx]  = model.predict_proba(X[val_idx])[:, 1]
        test_probs         += model.predict_proba(X_test)[:, 1] / N_SPLITS
        fold_auc = roc_auc_score(y[val_idx], oof_probs[val_idx])
        print(f"    Fold {fold+1}: AUC={fold_auc:.5f}")

    oof_auc   = roc_auc_score(y, oof_probs)
    thresholds = np.arange(0.3, 0.7, 0.01)
    best_t    = max(thresholds, key=lambda t: f1_score(y, (oof_probs >= t).astype(int)))
    oof_f1    = f1_score(y, (oof_probs >= best_t).astype(int))
    delta     = oof_auc - anchor_auc
    gate      = "PASS" if delta >= MIN_DELTA else "PRUNE"

    print(f"\n  {'='*50}")
    print(f"  {variant_name}")
    print(f"  OOF AUC  : {oof_auc:.5f}  (anchor: {anchor_auc:.5f})")
    print(f"  Delta    : {delta:+.5f}  → {gate}")
    print(f"  OOF F1   : {oof_f1:.5f}  (threshold: {best_t:.2f})")

    return {
        "variant":    variant_name,
        "features":   len(feature_cols),
        "oof_auc":    float(oof_auc),
        "oof_f1":     float(oof_f1),
        "threshold":  float(best_t),
        "delta":      float(delta),
        "gate":       gate,
        "oof_probs":  oof_probs,
        "test_probs": test_probs,
    }


# ── Phase D: Report Writer ────────────────────────────────────────────────────

def write_round_report(paths, results: list[dict], round_num: int, anchor_auc: float) -> None:
    passed  = [r for r in results if r["gate"] == "PASS"]
    pruned  = [r for r in results if r["gate"] == "PRUNE"]
    now     = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        f"# Feature Round {round_num} Report",
        f"**Generated**: {now}",
        f"**Anchor AUC**: {anchor_auc:.5f}",
        f"**Gate threshold**: anchor + {MIN_DELTA} = {anchor_auc + MIN_DELTA:.5f}",
        f"**Variants tested**: {len(results)}",
        f"**Passed**: {len(passed)}  |  **Pruned**: {len(pruned)}",
        "", "---", "",
        "## Results",
        "",
        f"| Variant | Features | OOF AUC | Delta | F1 | Gate |",
        f"|---|---|---|---|---|---|",
    ]
    for r in results:
        icon = "✅" if r["gate"] == "PASS" else "❌"
        lines.append(
            f"| {r['variant']} | {r['features']} | {r['oof_auc']:.5f} "
            f"| {r['delta']:+.5f} | {r['oof_f1']:.5f} | {icon} {r['gate']} |"
        )

    if passed:
        best = max(passed, key=lambda r: r["oof_auc"])
        lines += ["", "## Best Variant This Round", "",
                  f"**{best['variant']}** — AUC {best['oof_auc']:.5f} "
                  f"(Δ {best['delta']:+.5f})"]

    report_path = paths.reports_dir / f"feature_round_{round_num:02d}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    print(f"\n  ✅ Round report → {report_path}")


# ── Entry Point ───────────────────────────────────────────────────────────────

def run(variant_name: str | None = None, force_save: bool = False) -> dict:
    """
    Skill 07 — Feature Engineering entry point.

    If variant_name is None: runs fetch + extraction only.
    If variant_name is given: runs that specific variant against anchor.

    Variants defined:
      variant-06  : Lat + Lon + all 52 TerraClimate bands
      variant-07  : Lat + Lon + temperature only (tmax, tmin × 4 stats = 8)
      variant-08  : Lat + Lon + water balance (aet, def, pet, ppt, soil, q × 4 = 24)
      variant-09  : Lat + Lon + srad + vpd + vap (radiation + humidity × 4 = 12)
    """
    print(f"\n{'='*60}")
    print(f"SKILL 07 — Feature Engineering")
    print(f"{'='*60}\n")

    paths  = resolve_competition_paths()
    config = ChallengeConfig.load()
    store  = SkillStateStore(paths.state_path)
    state  = store.read()

    print(f"Competition : {config.slug}")
    print(f"DAG phase   : {state.get('dag_phase')}")
    print(f"Anchor AUC  : {state.get('anchor_oof_auc')}")

    anchor_auc = float(state.get("anchor_oof_auc") or 0.0)
    if anchor_auc == 0.0:
        raise RuntimeError("anchor_oof_auc not set in SKILL_STATE.json — run Skill 08 first")

    # ── Phase A: Fetch ────────────────────────────────────────
    print("\n[A] TerraClimate Fetch")
    tiff_path = fetch_terraclimate(paths)

    # ── Phase B: Extract ──────────────────────────────────────
    print("\n[B] Feature Extraction")
    train_feat, test_feat = extract_features(paths, tiff_path)

    if variant_name is None:
        print("\n✅ Fetch + extraction complete. Pass --variant <name> to run a variant.")
        return {"status": "extracted", "tiff": str(tiff_path)}

    # ── Phase C: Define variant feature sets ──────────────────
    tc_all   = TC_BAND_NAMES
    tc_temp  = [f"{v}_{s}" for v in ["tmax","tmin"] for s in TC_STATS]
    tc_water = [f"{v}_{s}" for v in ["aet","def","pet","ppt","soil","q"] for s in TC_STATS]
    tc_rad   = [f"{v}_{s}" for v in ["srad","vpd","vap"] for s in TC_STATS]

    # SHAP top features from variant-06
    shap_top5  = ["aet_min","tmin_mean","pet_mean","srad_mean","ppt_min"]
    shap_top10 = shap_top5 + ["vap_mean","pdsi_max","soil_max","srad_std","vpd_min"]
    shap_top20 = shap_top10 + ["vap_min","aet_mean","q_max","pet_min","pet_max",
                                "tmin_max","pdsi_min","pet_std","def_std","aet_max"]
    # Dead features by SHAP (near-zero importance)
    dead = ["swe_mean","swe_std","swe_min","swe_max","def_min","q_min"]
    tc_clean = [f for f in tc_all if f not in dead]  # 46 features

    VARIANTS = {
        # Round 1
        "variant-06": [tc_all,
        "variant-07": [tc_temp,
        "variant-08": [tc_water,
        "variant-09": [tc_rad,
        # Round 2 — SHAP-driven
        "variant-10": [shap_top10,
        "variant-11": [tc_clean,
        "variant-12": [shap_top20,
        "variant-13": [tc_all,    # hyperparams differ
        "variant-14": [tc_all,    # RF ensemble
        "variant-16": [shap_top5,

        # Round 3 — Science-driven (ecology: temp + moisture first)
        "variant-15": [[
            # Temperature — primary frog habitat driver
            "tmin_mean","tmin_std","tmin_min","tmin_max",
            "tmax_mean","tmax_std","tmax_min","tmax_max",
            # Moisture — critical for frog survival
            "soil_mean","soil_std","soil_min","soil_max",
            "ppt_mean","ppt_std","ppt_min","ppt_max",
            "vap_mean","vap_std","vap_min","vap_max",
        ],
        "variant-17": [[
            # All TC except swe (snow water = irrelevant for Australia)
            c for c in [
                "aet_mean","aet_std","aet_min","aet_max",
                "def_mean","def_std","def_min","def_max",
                "pdsi_mean","pdsi_std","pdsi_min","pdsi_max",
                "pet_mean","pet_std","pet_min","pet_max",
                "ppt_mean","ppt_std","ppt_min","ppt_max",
                "q_mean","q_std","q_min","q_max",
                "soil_mean","soil_std","soil_min","soil_max",
                "srad_mean","srad_std","srad_min","srad_max",
                "tmax_mean","tmax_std","tmax_min","tmax_max",
                "tmin_mean","tmin_std","tmin_min","tmin_max",
                "vap_mean","vap_std","vap_min","vap_max",
                "vpd_mean","vpd_std","vpd_min","vpd_max",
                "ws_mean","ws_std","ws_min","ws_max",
            ]
        ],
        "variant-18": [tc_all,  # XGBoost
        "variant-19": [tc_all,  # LGB larger trees
        "variant-20": [[
            # Top ecological predictors from SHAP + domain science
            "aet_min","tmin_mean","pet_mean","srad_mean",
            "soil_mean","ppt_mean","vap_mean","vpd_mean",
            "tmax_mean","aet_mean","def_mean","pdsi_mean",
        ],
        # Round 4 — Blend + threshold
        "variant-25": [tc_all,  # LGB+RF blend
        "variant-26": [tc_all,  # per-fold threshold
    }

    if variant_name not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant_name}'. Choose from: {list(VARIANTS)}")

    feature_cols = VARIANTS[variant_name]

    # ── Phase C: Train (multi-seed averaging) ────────────────
    SEEDS = [42, 123, 7]
    print(f"\n[C] Training {variant_name} over {len(SEEDS)} seeds: {SEEDS}")
    seed_results = []
    for s in SEEDS:
        print(f"\n  -- Seed {s} --")
        r = train_variant(train_feat, test_feat, feature_cols, variant_name, anchor_auc, seed=s)
        seed_results.append(r)

    # Average OOF AUC and test probabilities across seeds
    mean_auc   = float(np.mean([r["oof_auc"]    for r in seed_results]))
    std_auc    = float(np.std( [r["oof_auc"]    for r in seed_results]))
    mean_f1    = float(np.mean([r["oof_f1"]     for r in seed_results]))
    mean_thr   = float(np.mean([r["threshold"]  for r in seed_results]))
    mean_delta = float(np.mean([r["delta"]      for r in seed_results]))
    avg_test   = np.mean([r["test_probs"] for r in seed_results], axis=0)
    avg_oof    = np.mean([r["oof_probs"]  for r in seed_results], axis=0)
    gate       = "PASS" if mean_delta >= MIN_DELTA else "PRUNE"

    print(f"\n  {'='*50}")
    print(f"  {variant_name} — MULTI-SEED SUMMARY ({len(SEEDS)} seeds)")
    print(f"  Mean OOF AUC : {mean_auc:.5f}  ±{std_auc:.5f}")
    print(f"  Mean Delta   : {mean_delta:+.5f}  → {gate}")
    print(f"  Mean OOF F1  : {mean_f1:.5f}  (threshold: {mean_thr:.2f})")
    print(f"  Seed AUCs    : {[round(r['oof_auc'],5) for r in seed_results]}")

    result = {
        "variant":    variant_name,
        "features":   len(feature_cols),
        "oof_auc":    mean_auc,
        "oof_f1":     mean_f1,
        "threshold":  mean_thr,
        "delta":      mean_delta,
        "gate":       gate,
        "oof_probs":  avg_oof,
        "test_probs": avg_test,
        "seed_aucs":  [r["oof_auc"] for r in seed_results],
        "seed_std":   std_auc,
    }

    # ── Phase D: Save submission if PASS or force_save ──────────
    if result["gate"] == "PASS" or force_save:
        sample  = pd.read_csv(paths.data_raw_dir / "SampleSubmission.csv")
        sub_col = [c for c in sample.columns if c != "ID"][0]
        test_probs = result["test_probs"]
        preds   = (test_probs >= result["threshold"]).astype(int)
        sub     = pd.DataFrame({"ID": test_feat["ID"], sub_col: preds})
        sub     = sub.set_index("ID").reindex(sample["ID"]).reset_index()
        out     = paths.competition_dir / f"submissions/{variant_name}_submission.csv"
        sub.to_csv(out, index=False)
        print(f"  ✅ Submission saved → {out}")

    # ── Phase D: Update state ─────────────────────────────────
    variants_tested = int(state.get("variants_tested") or 0) + 1
    variants_passed = int(state.get("variants_passed") or 0) + (1 if result["gate"] == "PASS" else 0)
    best_auc = float(state.get("best_variant_oof_auc") or 0.0)

    update = {
        "dag_phase":              "phase_3_features",
        "variants_tested":        variants_tested,
        "variants_passed":        variants_passed,
        "last_updated":           datetime.now(timezone.utc).isoformat(),
    }
    if result["gate"] == "PASS" and result["oof_auc"] > best_auc:
        update["best_variant_this_round"] = variant_name
        update["best_variant_oof_auc"]    = result["oof_auc"]
        update["best_variant_oof_f1"]     = result["oof_f1"]
        update["best_variant_threshold"]  = result["threshold"]
        update["best_variant_features"]   = len(feature_cols)

    store.update(**update)
    print(f"  ✅ SKILL_STATE.json updated")

    # ── Phase D: Write report ─────────────────────────────────
    round_num = int(state.get("feature_round") or 1)
    write_round_report(paths, [result], round_num, anchor_auc)

    return {
        "status":   result["gate"],
        "variant":  variant_name,
        "oof_auc":  result["oof_auc"],
        "oof_f1":   result["oof_f1"],
        "delta":    result["delta"],
        "features": len(feature_cols),
    }


if __name__ == "__main__":
    import sys
    variant = None
    for arg in sys.argv[1:]:
        if arg.startswith("--variant="):
            variant = arg.split("=", 1)[1]
        elif arg == "--variant" and len(sys.argv) > sys.argv.index(arg) + 1:
            variant = sys.argv[sys.argv.index(arg) + 1]
    force_save = "--force-save" in sys.argv
    result = run(variant_name=variant, force_save=force_save)
    print(json.dumps({k: v for k, v in result.items() if k != "oof_probs"}, indent=2))
