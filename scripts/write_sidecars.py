"""Write sidecar variant JSONs with correct UTF-8 encoding (no BOM)."""

import json
import pathlib

comp = pathlib.Path("competitions/geoai-aquaculture-pond-identification-challenge")

sar_radar = {
    "_variant": "sar_radar_only",
    "_rationale": (
        "Restricts training to the 24 cloud-free Sentinel-1 SAR bands plus "
        "VH/VV ratio (cross-pol ratio, discriminates volume vs surface scatter) "
        "and VH*VV interaction (surface roughness proxy) per month. "
        "feature_engineering uses the generic ratios/interactions keys already "
        "consumed by build_hypothesis_features — no Python changes required. "
        "feature_columns restricts _resolve_variant_features to SAR-only columns."
    ),
    "feature_engineering": {
        "ratios": [
            ["VH_01", "VV_01"],
            ["VH_02", "VV_02"],
            ["VH_03", "VV_03"],
            ["VH_04", "VV_04"],
            ["VH_05", "VV_05"],
            ["VH_06", "VV_06"],
            ["VH_07", "VV_07"],
            ["VH_08", "VV_08"],
            ["VH_09", "VV_09"],
            ["VH_10", "VV_10"],
            ["VH_11", "VV_11"],
            ["VH_12", "VV_12"],
        ],
        "interactions": [
            ["VH_01", "VV_01"],
            ["VH_02", "VV_02"],
            ["VH_03", "VV_03"],
            ["VH_04", "VV_04"],
            ["VH_05", "VV_05"],
            ["VH_06", "VV_06"],
            ["VH_07", "VV_07"],
            ["VH_08", "VV_08"],
            ["VH_09", "VV_09"],
            ["VH_10", "VV_10"],
            ["VH_11", "VV_11"],
            ["VH_12", "VV_12"],
        ],
    },
    "feature_columns": [
        "VH_01",
        "VH_02",
        "VH_03",
        "VH_04",
        "VH_05",
        "VH_06",
        "VH_07",
        "VH_08",
        "VH_09",
        "VH_10",
        "VH_11",
        "VH_12",
        "VV_01",
        "VV_02",
        "VV_03",
        "VV_04",
        "VV_05",
        "VV_06",
        "VV_07",
        "VV_08",
        "VV_09",
        "VV_10",
        "VV_11",
        "VV_12",
        "VH_01_div_VV_01",
        "VH_02_div_VV_02",
        "VH_03_div_VV_03",
        "VH_04_div_VV_04",
        "VH_05_div_VV_05",
        "VH_06_div_VV_06",
        "VH_07_div_VV_07",
        "VH_08_div_VV_08",
        "VH_09_div_VV_09",
        "VH_10_div_VV_10",
        "VH_11_div_VV_11",
        "VH_12_div_VV_12",
        "VH_01_x_VV_01",
        "VH_02_x_VV_02",
        "VH_03_x_VV_03",
        "VH_04_x_VV_04",
        "VH_05_x_VV_05",
        "VH_06_x_VV_06",
        "VH_07_x_VV_07",
        "VH_08_x_VV_08",
        "VH_09_x_VV_09",
        "VH_10_x_VV_10",
        "VH_11_x_VV_11",
        "VH_12_x_VV_12",
    ],
    "model": {
        "family": "lgb",
        "hyperparams": {
            "num_leaves": 16,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "min_child_samples": 10,
            "subsample": 0.8,
            "colsample_bytree": 1.0,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        },
        "num_boost_round": 500,
        "early_stopping": 50,
    },
}

sar_optical = {
    "_variant": "sar_optical_ratios",
    "_rationale": (
        "Cross-modal SAR-to-optical ratio features per month. VH/NIR is near-zero "
        "for open water (NIR absorbs in water; VH is low for calm surfaces), making "
        "it a strong pond detector. VV/NIR and VH/SWIR1 exploit the same microwave-vs-"
        "reflectance contrast. VH/VV (cross-pol ratio) is also included. All 48 ratios "
        "are expressed as generic ratio pairs. No feature_columns restriction — the "
        "model trains on all 144 raw bands plus the 48 new ratio features."
    ),
    "feature_engineering": {
        "ratios": [
            ["VH_01", "nir_01"],
            ["VH_02", "nir_02"],
            ["VH_03", "nir_03"],
            ["VH_04", "nir_04"],
            ["VH_05", "nir_05"],
            ["VH_06", "nir_06"],
            ["VH_07", "nir_07"],
            ["VH_08", "nir_08"],
            ["VH_09", "nir_09"],
            ["VH_10", "nir_10"],
            ["VH_11", "nir_11"],
            ["VH_12", "nir_12"],
            ["VV_01", "nir_01"],
            ["VV_02", "nir_02"],
            ["VV_03", "nir_03"],
            ["VV_04", "nir_04"],
            ["VV_05", "nir_05"],
            ["VV_06", "nir_06"],
            ["VV_07", "nir_07"],
            ["VV_08", "nir_08"],
            ["VV_09", "nir_09"],
            ["VV_10", "nir_10"],
            ["VV_11", "nir_11"],
            ["VV_12", "nir_12"],
            ["VH_01", "swir1_01"],
            ["VH_02", "swir1_02"],
            ["VH_03", "swir1_03"],
            ["VH_04", "swir1_04"],
            ["VH_05", "swir1_05"],
            ["VH_06", "swir1_06"],
            ["VH_07", "swir1_07"],
            ["VH_08", "swir1_08"],
            ["VH_09", "swir1_09"],
            ["VH_10", "swir1_10"],
            ["VH_11", "swir1_11"],
            ["VH_12", "swir1_12"],
            ["VH_01", "VV_01"],
            ["VH_02", "VV_02"],
            ["VH_03", "VV_03"],
            ["VH_04", "VV_04"],
            ["VH_05", "VV_05"],
            ["VH_06", "VV_06"],
            ["VH_07", "VV_07"],
            ["VH_08", "VV_08"],
            ["VH_09", "VV_09"],
            ["VH_10", "VV_10"],
            ["VH_11", "VV_11"],
            ["VH_12", "VV_12"],
        ]
    },
    "model": {
        "family": "lgb",
        "hyperparams": {
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "min_child_samples": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.05,
            "reg_lambda": 0.5,
        },
        "num_boost_round": 500,
        "early_stopping": 50,
    },
}

variants_dir = comp / "variants"
variants_dir.mkdir(parents=True, exist_ok=True)

for name, data in [("sar_radar_only", sar_radar), ("sar_optical_ratios", sar_optical)]:
    p = variants_dir / f"{name}.json"
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    # Round-trip verify
    loaded = json.loads(p.read_text(encoding="utf-8"))
    n_ratios = len(loaded["feature_engineering"]["ratios"])
    fc = len(loaded.get("feature_columns", []))
    print(
        f"[OK] {name}.json — ratios={n_ratios}, feature_columns={fc}, model={loaded['model']['family']}"
    )
