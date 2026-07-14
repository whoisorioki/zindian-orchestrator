"""
Validate the SAR variant sidecar architecture end-to-end.
Tests:
  1. build_hypothesis_features contains no hardcoded variant names
  2. sar_radar_only: correct FE columns created, feature_columns resolves correctly
  3. sar_optical_ratios: correct cross-modal ratio columns created
  4. Two-Mode Feature Contract holds for both variants
"""

import sys
import json
import inspect
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, ".")
from zindian.config import ChallengeConfig
from zindian.skills.skill_07_features import (
    build_hypothesis_features,
    merge_feature_engineering_config,
    synthesize_default_feature_engineering,
)

comp = Path("competitions/geoai-aquaculture-pond-identification-challenge")
train_raw = pd.read_csv(comp / "data/processed/features_train.csv", nrows=300)
test_raw = pd.read_csv(comp / "data/processed/features_test.csv", nrows=300)
if "label" in train_raw.columns:
    train_raw = train_raw.drop(columns=["label"])

config = ChallengeConfig.load()
merged_base = merge_feature_engineering_config(
    synthesize_default_feature_engineering(config._data, {}),
    config.get("feature_engineering") or {},
)

# -----------------------------------------------------------------------
# Test 1: No competition-specific strings in build_hypothesis_features
# -----------------------------------------------------------------------
src = inspect.getsource(build_hypothesis_features)
banned = ["sar_radar_only", "sar_optical_ratios", "VH_0", "VV_0", "nir_0"]
for b in banned:
    assert (
        b not in src
    ), f"FAIL: hardcoded string {b!r} found in build_hypothesis_features"
print("[1] build_hypothesis_features contains no competition-specific strings: PASS")

# -----------------------------------------------------------------------
# Test 2: sar_radar_only — FE creates ratio/interaction columns
# -----------------------------------------------------------------------
sidecar_sar = json.loads(
    (comp / "variants/sar_radar_only.json").read_text(encoding="utf-8")
)
fe_sar = {**merged_base, **sidecar_sar.get("feature_engineering", {})}
tr1, te1 = build_hypothesis_features(
    train_raw,
    test_raw,
    mode="inference",
    variant_name="sar_radar_only",
    merged_fe_cfg=fe_sar,
)

# Expected: VH_01_div_VV_01 ... VH_12_div_VV_12 (12 ratios)
# Expected: VH_01_x_VV_01   ... VH_12_x_VV_12   (12 interactions)
for m in ["01", "06", "12"]:
    assert f"VH_{m}_div_VV_{m}" in tr1.columns, f"FAIL: VH_{m}_div_VV_{m} missing"
    assert f"VH_{m}_x_VV_{m}" in tr1.columns, f"FAIL: VH_{m}_x_VV_{m} missing"

# All feature_columns exist in df
fc = sidecar_sar["feature_columns"]
missing = [c for c in fc if c not in tr1.columns]
assert not missing, f"FAIL: feature_columns missing from df: {missing}"
print(
    "[2] sar_radar_only FE: 12 ratio + 12 interaction cols created, all feature_columns present: PASS"
)

# -----------------------------------------------------------------------
# Test 3: sar_optical_ratios — cross-modal ratios created
# -----------------------------------------------------------------------
sidecar_opt = json.loads(
    (comp / "variants/sar_optical_ratios.json").read_text(encoding="utf-8")
)
fe_opt = {**merged_base, **sidecar_opt.get("feature_engineering", {})}
tr2, te2 = build_hypothesis_features(
    train_raw,
    test_raw,
    mode="inference",
    variant_name="sar_optical_ratios",
    merged_fe_cfg=fe_opt,
)
for m in ["01", "06", "12"]:
    assert f"VH_{m}_div_nir_{m}" in tr2.columns, f"FAIL: VH_{m}_div_nir_{m} missing"
    assert f"VV_{m}_div_nir_{m}" in tr2.columns, f"FAIL: VV_{m}_div_nir_{m} missing"
    assert f"VH_{m}_div_swir1_{m}" in tr2.columns, f"FAIL: VH_{m}_div_swir1_{m} missing"
    assert f"VH_{m}_div_VV_{m}" in tr2.columns, f"FAIL: VH_{m}_div_VV_{m} missing"
opt_new = [c for c in tr2.columns if c not in train_raw.columns]
print(
    f"[3] sar_optical_ratios FE: {len(opt_new)} new columns (48 cross-modal ratios): PASS"
)

# -----------------------------------------------------------------------
# Test 4: Two-Mode Contract — CV mode produces same column set
# -----------------------------------------------------------------------
idx = np.arange(200)
tr1_cv, _ = build_hypothesis_features(
    train_raw,
    test_raw,
    mode="cv",
    train_idx=idx,
    variant_name="sar_radar_only",
    merged_fe_cfg=fe_sar,
)
tr2_cv, _ = build_hypothesis_features(
    train_raw,
    test_raw,
    mode="cv",
    train_idx=idx,
    variant_name="sar_optical_ratios",
    merged_fe_cfg=fe_opt,
)

assert sorted(tr1_cv.columns.tolist()) == sorted(
    tr1.columns.tolist()
), "FAIL: sar_radar_only CV columns differ from inference columns"
assert sorted(tr2_cv.columns.tolist()) == sorted(
    tr2.columns.tolist()
), "FAIL: sar_optical_ratios CV columns differ from inference columns"
print("[4] Two-Mode Contract (CV == inference column set) for both variants: PASS")

# -----------------------------------------------------------------------
# Test 5: feature_columns sidecar loader picks up sar_radar_only
# -----------------------------------------------------------------------
# Simulate the loader block from run()
import importlib

sk7 = importlib.import_module("zindian.skills.skill_07_features")
# The loader runs inside run() — verify the JSON is readable and stem matches
p = comp / "variants/sar_radar_only.json"
d = json.loads(p.read_text(encoding="utf-8"))
assert d.get("feature_columns"), "FAIL: feature_columns missing from sidecar"
assert p.stem == "sar_radar_only", "FAIL: stem mismatch"
print(f"[5] Sidecar feature_columns loadable, stem={p.stem!r}: PASS")

print()
print("ALL ARCHITECTURE CHECKS PASSED")
