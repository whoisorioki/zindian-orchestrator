"""
Dry-run validation for variant-10 and variant-11 feature lists.
Checks A5 compliance, target-leak absence, and test-frame completeness.
Run from repo root:
    python scripts/_validate_variants.py
"""

import os
import sys

sys.path.insert(0, ".")
os.environ["ZINDIAN_COMPETITION_SLUG"] = (
    "june-study-jam-series-transaction-volume-forecasting-challenge"
)

import pandas as pd
from zindian.config import ChallengeConfig
from zindian.paths import resolve_competition_paths

paths = resolve_competition_paths()
config = ChallengeConfig.load()

train_feat = pd.read_csv(paths.data_processed_dir / "features_train.csv")
test_feat = pd.read_csv(paths.data_processed_dir / "features_test.csv")

target_col = config.get("target_col") or config.get("target_column") or "target"
id_col = config.get("id_col") or "ID"
DROP = {id_col, target_col, "Latitude", "Longitude", "ID", "target"}
all_features = [c for c in train_feat.columns if c not in DROP and c != target_col]

_dead = set(config.get("dead_features", []) or [])
_noise = set(config.get("noise_features", []) or [])
_excluded = _dead | _noise
clean_features = [f for f in all_features if f not in _excluded]

_fe_cfg = config.get("feature_engineering", {}) or {}
_interaction_pairs = _fe_cfg.get("interactions", []) or []
interaction_cols = [
    f"{pair[0]}_x_{pair[1]}"
    for pair in _interaction_pairs
    if len(pair) == 2
    and pair[0] in train_feat.columns
    and pair[1] in train_feat.columns
]

v10 = clean_features
v11 = clean_features + interaction_cols

print(f"all_features   : {len(all_features)}")
print(f"dead removed   : {sorted(_dead)}")
print(f"noise removed  : {sorted(_noise)}")
print(f"clean_features : {len(clean_features)}")
print(f"interaction_cols ({len(interaction_cols)}): {interaction_cols}")
print(f"variant-10     : {len(v10)} features")
print(f"variant-11     : {len(v11)} features")
print()

errors = []

# A5: no competition-specific strings in skill code
# (column names come from config, not skill literals — validated by inspection)

# No target leak
if target_col in v10:
    errors.append("TARGET LEAK in variant-10")
if target_col in v11:
    errors.append("TARGET LEAK in variant-11")

# Interaction cols derivable from config (names come from pairs, not literals)
for pair in _interaction_pairs:
    if len(pair) != 2:
        errors.append(f"Malformed interaction pair: {pair}")
    elif not isinstance(pair[0], str) or not isinstance(pair[1], str):
        errors.append(f"Non-string column in pair: {pair}")

# Simulate build_hypothesis_features output on test frame
test_copy = test_feat.copy()
for pair in _interaction_pairs:
    if len(pair) == 2 and pair[0] in test_copy.columns and pair[1] in test_copy.columns:
        out = f"{pair[0]}_x_{pair[1]}"
        test_copy[out] = test_copy[pair[0]].astype(float) * test_copy[pair[1]].astype(
            float
        )

missing_in_test = [f for f in v11 if f not in test_copy.columns]
if missing_in_test:
    errors.append(
        f"Columns missing from test after interaction build: {missing_in_test}"
    )

# Verify no dead/noise features leaked back in
for f in v10 + v11:
    if f in _dead:
        errors.append(f"Dead feature in variant list: {f}")
    if f in _noise:
        errors.append(f"Noise feature in variant list: {f}")

if errors:
    print("VALIDATION FAILED:")
    for e in errors:
        print(f"  ERROR: {e}")
    sys.exit(1)
else:
    print("All checks PASS.")
    print()
    print("variant-10 features:", v10)
    print()
    print("variant-11 extra interaction cols:", interaction_cols)
