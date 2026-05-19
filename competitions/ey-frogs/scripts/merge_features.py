"""
Feature Merger
Combines:
  1. features_train/test.csv       — 52 TC bands (mean/std/min/max)
  2. features_last3mo_train/test.csv — 13 last-3mo means
  3. Derived: range = max - min     — 13 features
  4. Derived: cv = std / mean       — 13 features
Total: 52 + 13 + 13 + 13 = 91 features

Output: features_full_train.csv, features_full_test.csv
These are the files variants use — not the base features_train.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

SLUG  = "ey-frogs"
BASE  = Path(f"competitions/{SLUG}")

TC_VARIABLES = [
    "aet", "def", "pdsi", "pet", "ppt",
    "q", "soil", "srad", "swe",
    "tmax", "tmin", "vap", "vpd",
]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add range and cv for each TC variable."""
    for var in TC_VARIABLES:
        mean_col = f"{var}_mean"
        std_col  = f"{var}_std"
        min_col  = f"{var}_min"
        max_col  = f"{var}_max"
        if all(c in df.columns for c in [mean_col, std_col, min_col, max_col]):
            df[f"{var}_range"] = df[max_col] - df[min_col]
            df[f"{var}_cv"]    = df[std_col] / (df[mean_col].abs() + 1e-9)
    return df


def apply_approved_signatures(df: pd.DataFrame) -> pd.DataFrame:
    """
    Appends literature-validated non-linear transformations to the matrix.
    Includes epsilon guards to eliminate NaN generation risks.
    """
    print("\n[Pipeline] Materializing literature-backed feature signatures...")
    epsilon = 1e-6

    # 1. Structural Aridity Index (PET / PPT)
    if "pet_mean" in df.columns and "ppt_mean" in df.columns:
        df["pet_mean|ppt_mean::ratio"] = df["pet_mean"] / (df["ppt_mean"] + epsilon)
        print("  ✓ Compiled: pet_mean|ppt_mean::ratio")

    # 2. Evapotranspiration Fraction (AET / PET)
    if "aet_mean" in df.columns and "pet_mean" in df.columns:
        df["aet_mean|pet_mean::ratio"] = df["aet_mean"] / (df["pet_mean"] + epsilon)
        print("  ✓ Compiled: aet_mean|pet_mean::ratio")

    # 3. High-Heat Desiccation Risk (TMAX * VPD)
    if "tmax_mean" in df.columns and "vpd_mean" in df.columns:
        df["tmax_mean|vpd_mean::interaction_product"] = df["tmax_mean"] * df["vpd_mean"]
        print("  ✓ Compiled: tmax_mean|vpd_mean::interaction_product")

    return df


def merge(split: str) -> pd.DataFrame:
    base     = pd.read_csv(BASE / f"data/processed/features_{split}.csv")
    last3mo  = pd.read_csv(BASE / f"data/processed/features_last3mo_{split}.csv")

    # Merge on ID
    merged = base.merge(last3mo, on="ID", how="left")

    # Add derived features
    merged = add_derived_features(merged)

    # Add approved literature-backed signatures (ratios / interaction products)
    merged = apply_approved_signatures(merged)

    feature_cols = [c for c in merged.columns
                    if c not in ["ID", "Occurrence Status", "Latitude", "Longitude"]]
    print(f"  {split}: {len(merged)} rows, {len(feature_cols)} features")

    out = BASE / f"data/processed/features_full_{split}.csv"
    merged.to_csv(out, index=False)
    print(f"  ✅ Saved → {out}")
    return merged


def main():
    print("=" * 60)
    print("Feature Merger — TC base + last3mo + range + cv")
    print("=" * 60)

    last3mo_train = BASE / "data/processed/features_last3mo_train.csv"
    last3mo_test  = BASE / "data/processed/features_last3mo_test.csv"

    if not last3mo_train.exists():
        print("❌ Last-3mo features not found — run fetch_terraclimate_last3mo.py first")
        return

    print("\nMerging train features...")
    train = merge("train")

    print("\nMerging test features...")
    test  = merge("test")

    # Feature summary
    base_cols   = [c for c in train.columns if any(
        c == f"{v}_{s}" for v in TC_VARIABLES for s in ["mean","std","min","max"])]
    last3mo_cols = [c for c in train.columns if "last3mo" in c]
    range_cols  = [c for c in train.columns if c.endswith("_range")]
    cv_cols     = [c for c in train.columns if c.endswith("_cv")]

    print(f"\nFeature breakdown:")
    print(f"  TC base (mean/std/min/max) : {len(base_cols)}")
    print(f"  Last-3mo means             : {len(last3mo_cols)}")
    print(f"  Range (max-min)            : {len(range_cols)}")
    print(f"  CV (std/mean)              : {len(cv_cols)}")
    print(f"  Total                      : {len(base_cols)+len(last3mo_cols)+len(range_cols)+len(cv_cols)}")
    print(f"\n✅ Merge complete — use features_full_train/test.csv for variants")


if __name__ == "__main__":
    main()
