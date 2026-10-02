#!/usr/bin/env python3
"""
Phase 5.1 — macro-stress-geofence refit as a residual specialist.
Removes hand-multiplied age_x_spei_6 and age_x_viirs_radiance_lag1m columns.
Trains on anchor-baseline OOF residuals (y_true - anchor_prob).
Final prediction = clip(anchor_prob + residual_pred, 1e-7, 1 - 1e-7)
Evaluated with corrected gate formula: 0.6 * F1@0.5 + 0.4 * AUC
Registers OOF record into SKILL_STATE.json per Zindian OOF contract.
"""
import json
import sys
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

from zindian.metrics import ScoreProvenance, composite_metric
from zindian.state import SkillStateStore, write_oof_record

COMP_DIR = Path("competitions/climate-risk-health-prediction-challenge")
DATA_DIR = COMP_DIR / "data/processed"
SCORES_DIR = COMP_DIR / "scores"
VARIANTS_DIR = COMP_DIR / "variants"

# Ensure output directories exist
SCORES_DIR.mkdir(parents=True, exist_ok=True)
VARIANTS_DIR.mkdir(parents=True, exist_ok=True)

config = json.loads((COMP_DIR / "challenge_config.json").read_text(encoding="utf-8"))

TARGET = config.get("target_col") or "is_climate_sensitive"
SEED = config.get("reproducibility", {}).get("seed", 42)
ID_COL = config.get("id_col", "ID")
REMOVE_COLS = ["age_x_spei_6", "age_x_viirs_radiance_lag1m"]

print(f"Target: {TARGET}, Seed: {SEED}")

# Load anchor OOF
with open(SCORES_DIR / "branch_anchor-baseline_oof.json") as f:
    anchor_oof = np.array(json.load(f), dtype=np.float64)
print(f"Anchor OOF: {len(anchor_oof)} rows, mean={anchor_oof.mean():.4f}")

# Load data
train = pd.read_csv(DATA_DIR / "features_train_macro-stress-geofence.csv")
test = pd.read_csv(DATA_DIR / "features_test_macro-stress-geofence.csv")
print(f"Train: {train.shape}, Test: {test.shape}")

feature_cols = [
    c
    for c in train.columns
    if c not in REMOVE_COLS + [TARGET, ID_COL, "zone"] and c in test.columns
]
print(f"Features ({len(feature_cols)}): {feature_cols}")
print(f"Removed: {[c for c in REMOVE_COLS if c in train.columns]}")

assert len(train) == len(anchor_oof), (
    f"Mismatch: train={len(train)}, anchor_oof={len(anchor_oof)}"
)
y_true = train[TARGET].values.astype(np.int32)
residuals = y_true.astype(np.float64) - anchor_oof
print(f"Residuals: mean={residuals.mean():.4f}, std={residuals.std():.4f}")

# Encode non-numeric columns
for col in feature_cols:
    if not pd.api.types.is_numeric_dtype(train[col]):
        le = LabelEncoder()
        all_vals = pd.concat([train[col], test[col]]).astype(str).unique()
        le.fit(all_vals)
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))
        print(f"  Encoded {col}: {len(all_vals)} unique values")

X = train[feature_cols].values.astype(np.float64)
X_test = test[feature_cols].values.astype(np.float64)

# CV splits
cv_split_path = DATA_DIR / "cv_split_indices.json"
if cv_split_path.exists():
    with open(cv_split_path) as f:
        cv_splits = json.load(f)
    print(f"Using explicit CV splits: {len(cv_splits)} folds")
else:
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_splits = [(t.tolist(), v.tolist()) for t, v in kf.split(X)]
    print(f"Using KFold({len(cv_splits)})")

# Train LightGBM Regressor on residuals
oof_res = np.zeros(len(train), dtype=np.float64)
test_res = np.zeros(len(test), dtype=np.float64)

for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=SEED,
        verbose=-1,
    )
    model.fit(
        X[train_idx],
        residuals[train_idx],
        eval_set=[(X[val_idx], residuals[val_idx])],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    oof_res[val_idx] = model.predict(X[val_idx])
    test_res += model.predict(X_test) / len(cv_splits)
    print(f"  Fold {fold_idx+1}: best_iteration={model.best_iteration_}")

oof_final = np.clip(anchor_oof + oof_res, 1e-7, 1 - 1e-7)

# Load anchor test probs for final combination
anchor_test_path = DATA_DIR / "calib_test_probs_anchor-baseline.csv"
if not anchor_test_path.exists():
    anchor_test_path = DATA_DIR / "test_probs_anchor-baseline.csv"

anchor_test = pd.read_csv(anchor_test_path)
prob_cols = [c for c in anchor_test.columns if c != ID_COL]
anchor_test_probs = anchor_test[prob_cols[0]].values.astype(np.float64)
test_final = np.clip(anchor_test_probs + test_res, 1e-7, 1 - 1e-7)
print(f"Anchor test probs: {len(anchor_test_probs)} rows")

# Evaluate metrics using composite_metric guard
oof_binary = (oof_final >= 0.5).astype(int)
f1 = f1_score(y_true, oof_binary)
auc = roc_auc_score(y_true, oof_final)
composite = composite_metric(f1, auc)

anchor_bin = (anchor_oof >= 0.5).astype(int)
a_f1 = f1_score(y_true, anchor_bin)
a_auc = roc_auc_score(y_true, anchor_oof)
a_comp = composite_metric(a_f1, a_auc)

print(f"\n{'='*50}")
print("macro-stress-geofence-refit (residual specialist)")
print(f"OOF F1@0.5:  {f1:.6f}")
print(f"OOF AUC:     {auc:.6f}")
print(f"Composite:   {composite:.6f}")
print("\nAnchor-baseline:")
print(f"OOF F1@0.5:  {a_f1:.6f}")
print(f"OOF AUC:     {a_auc:.6f}")
print(f"Composite:   {a_comp:.6f}")
print(f"\nDelta vs anchor: {composite - a_comp:+.6f}")

# Save score vector & predictions
variant_name = "macro-stress-geofence-refit"
with open(SCORES_DIR / f"branch_{variant_name}_oof.json", "w") as f:
    json.dump(oof_final.tolist(), f)

test_id_vals = anchor_test[ID_COL].values if ID_COL in anchor_test.columns else (test[ID_COL].values if ID_COL in test.columns else np.arange(len(test_final)))

test_df_out = pd.DataFrame({
    ID_COL: test_id_vals,
    "TargetRAUC": test_final,
    "TargetF1": (test_final >= 0.5).astype(int),
})
test_df_out.to_csv(
    DATA_DIR / f"calib_test_probs_{variant_name}.csv", index=False
)
test_df_out.to_csv(DATA_DIR / f"test_probs_{variant_name}.csv", index=False)

train[feature_cols + [TARGET]].to_csv(
    DATA_DIR / f"features_train_{variant_name}.csv", index=False
)
test[feature_cols].to_csv(
    DATA_DIR / f"features_test_{variant_name}.csv", index=False
)

# Write OOF record into SKILL_STATE.json per Zindian OOF contract
store = SkillStateStore(COMP_DIR / "SKILL_STATE.json")
write_oof_record(
    store=store,
    branch_name=variant_name,
    scores=oof_final.tolist(),
    cv_strategy_id="config:BufferedSpatialCV",
    seed=SEED,
    model_config={
        "feature_count": len(feature_cols),
        "removed_columns": REMOVE_COLS,
        "variant": variant_name,
        "training_method": "residual_specialist",
    },
)

# Record variant metadata
variant_config = {
    "description": (
        "Refit as residual specialist. Trained on anchor OOF residuals. "
        "Hand-multiplied age_x_spei_6 and age_x_viirs_radiance_lag1m removed."
    ),
    "parent_variant": "macro-stress-geofence",
    "training_method": "residual_specialist",
    "removed_columns": REMOVE_COLS,
    "feature_columns": feature_cols,
    "feature_count": len(feature_cols),
    "oof_results": {
        "f1_05": round(f1, 6),
        "auc": round(auc, 6),
        "composite_06_04": round(composite, 6),
        "delta_vs_anchor": round(composite - a_comp, 6),
    },
}
with open(VARIANTS_DIR / f"{variant_name}.json", "w") as f:
    json.dump(variant_config, f, indent=2)

print(f"\nPhase 5.1 complete. Artifacts saved.")
