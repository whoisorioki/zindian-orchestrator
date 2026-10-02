#!/usr/bin/env python3
"""
Refit script for residual-dlnm-specialist variant.
Trains a CatBoost regressor (without monotone constraints) on anchor-baseline OOF residuals.
Registers OOF record into SKILL_STATE.json per Zindian OOF contract.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

from zindian.metrics import ScoreProvenance, composite_metric
from zindian.state import SkillStateStore, write_oof_record
from zindian.ledger import Ledger
from zindian.state import resolve_active_cv_strategy_id
from zindian.cv import make_cv_splitter

COMP_DIR = Path("competitions/climate-risk-health-prediction-challenge")
DATA_DIR = COMP_DIR / "data/processed"
SCORES_DIR = COMP_DIR / "scores"
VARIANTS_DIR = COMP_DIR / "variants"

SCORES_DIR.mkdir(parents=True, exist_ok=True)
VARIANTS_DIR.mkdir(parents=True, exist_ok=True)

config = json.loads((COMP_DIR / "challenge_config.json").read_text(encoding="utf-8"))

TARGET = config.get("target_col") or "is_climate_sensitive"
SEED = config.get("reproducibility", {}).get("seed", 42)
ID_COL = config.get("id_col", "ID")
VARIANT_NAME = "residual-dlnm-specialist"

print(f"Refitting Specialist Variant: {VARIANT_NAME}")
print(f"Target: {TARGET}, Seed: {SEED}")

# Load sidecar configuration
sidecar_file = VARIANTS_DIR / f"{VARIANT_NAME}.json"
sidecar_cfg = {}
if sidecar_file.exists():
    sidecar_cfg = json.loads(sidecar_file.read_text(encoding="utf-8"))

RESIDUAL_SOURCE = sidecar_cfg.get("residual_source", "anchor-baseline")

# Load anchor/parent OOF predictions
anchor_oof_path = SCORES_DIR / f"branch_{RESIDUAL_SOURCE}_oof.json"
if not anchor_oof_path.exists():
    state_file = COMP_DIR / "SKILL_STATE.json"
    if state_file.exists():
        state_data = json.loads(state_file.read_text(encoding="utf-8"))
        oof_recs = state_data.get("oof_records", {})
        if RESIDUAL_SOURCE in oof_recs:
            anchor_oof = np.array(oof_recs[RESIDUAL_SOURCE].get("scores", []), dtype=np.float64)
        else:
            raise RuntimeError(f"Residual source '{RESIDUAL_SOURCE}' OOF predictions missing.")
    else:
        raise RuntimeError(f"Residual source '{RESIDUAL_SOURCE}' OOF predictions missing.")
else:
    anchor_oof = np.array(json.loads(anchor_oof_path.read_text(encoding="utf-8")), dtype=np.float64)

print(f"Anchor OOF ({RESIDUAL_SOURCE}): {len(anchor_oof)} rows, mean={anchor_oof.mean():.4f}")

# Load branch features
train_feat_path = DATA_DIR / f"features_train_{VARIANT_NAME}.csv"
test_feat_path = DATA_DIR / f"features_test_{VARIANT_NAME}.csv"

if not train_feat_path.exists():
    print(f"[WARN] Feature file {train_feat_path.name} not found. Ensure Phase 2B plugin has executed.")
    sys.exit(1)

train = pd.read_csv(train_feat_path)
test = pd.read_csv(test_feat_path)
print(f"Train: {train.shape}, Test: {test.shape}")

cols_cfg = config.get("columns", {}) or {}
lat_col = cols_cfg.get("latitude", "latitude")
lon_col = cols_cfg.get("longitude", "longitude")
excluded_cols = {ID_COL, TARGET, lat_col, lon_col, "deathdate", "date"}

feature_cols = [c for c in train.columns if c not in excluded_cols and c in test.columns]
print(f"Feature count: {len(feature_cols)}")

y_true = train[TARGET].values.astype(np.float64)
residuals = y_true - anchor_oof
print(f"Residuals: mean={residuals.mean():.4f}, std={residuals.std():.4f}")

# Load anchor test probabilities
anchor_test_path = DATA_DIR / f"test_probs_{RESIDUAL_SOURCE}.csv"
if anchor_test_path.exists():
    anchor_test_df = pd.read_csv(anchor_test_path)
    prob_cols = [c for c in anchor_test_df.columns if c != ID_COL]
    anchor_test_probs = anchor_test_df[prob_cols[0]].values.astype(np.float64)
else:
    anchor_test_probs = np.full(len(test), anchor_oof.mean())

# Encode non-numeric features
for col in feature_cols:
    if not pd.api.types.is_numeric_dtype(train[col]):
        le = LabelEncoder()
        all_vals = pd.concat([train[col], test[col]]).astype(str).unique()
        le.fit(all_vals)
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))

X = train[feature_cols].values.astype(np.float64)
X_test = test[feature_cols].values.astype(np.float64)

# Load CV splits
cv_split_path = DATA_DIR / "cv_split_indices.json"
if cv_split_path.exists():
    with open(cv_split_path, encoding="utf-8") as f:
        cv_splits = json.load(f)
    print(f"Using explicit CV splits: {len(cv_splits)} folds")
else:
    kf = make_cv_splitter({"type": "kfold", "n_splits": 5}, random_seed=SEED)
    cv_splits = [(t.tolist(), v.tolist()) for t, v in kf.split(X)]

from catboost import CatBoostRegressor

oof_res = np.zeros(len(train), dtype=np.float64)
test_res = np.zeros(len(test), dtype=np.float64)

for fold_idx, (tr_idx, val_idx) in enumerate(cv_splits):
    cb_params = {
        "iterations": 800,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": SEED + fold_idx,
        "verbose": 0,
    }

    model = CatBoostRegressor(**cb_params)
    model.fit(
        X[tr_idx],
        residuals[tr_idx],
        eval_set=(X[val_idx], residuals[val_idx]),
        early_stopping_rounds=50,
    )
    oof_res[val_idx] = model.predict(X[val_idx])
    test_res += model.predict(X_test) / len(cv_splits)

# Combine predictions (Design 2: probability addition with clipping)
oof_final = np.clip(anchor_oof + oof_res, 1e-7, 1 - 1e-7)
test_final = np.clip(anchor_test_probs + test_res, 1e-7, 1 - 1e-7)

# Metric evaluation
oof_binary = (oof_final >= 0.5).astype(int)
oof_f1 = float(f1_score(y_true, oof_binary))
oof_auc = float(roc_auc_score(y_true, oof_final))
composite = float(composite_metric(oof_f1, oof_auc))

print(f"\n[RESULTS] {VARIANT_NAME}:")
print(f"  OOF F1@0.5: {oof_f1:.6f}")
print(f"  OOF AUC   : {oof_auc:.6f}")
print(f"  Composite : {composite:.6f}")

# Persist score vectors
with open(SCORES_DIR / f"branch_{VARIANT_NAME}_oof.json", "w", encoding="utf-8") as f:
    json.dump(oof_final.tolist(), f)

test_out_df = pd.DataFrame({
    ID_COL: test[ID_COL].values if ID_COL in test.columns else np.arange(len(test_final)),
    "test_prob": test_final,
})
test_out_df.to_csv(DATA_DIR / f"test_probs_{VARIANT_NAME}.csv", index=False)

# Register to SKILL_STATE.json
state_store = SkillStateStore(COMP_DIR / "SKILL_STATE.json")
state_data = state_store.read()
cv_strategy_id = resolve_active_cv_strategy_id(state_data, config)

write_oof_record(
    state_store,
    branch_name=VARIANT_NAME,
    scores=oof_final.tolist(),
    cv_strategy_id=cv_strategy_id,
    seed=SEED,
    model_config={
        "feature_count": len(feature_cols),
        "variant": VARIANT_NAME,
        "training_method": "residual_specialist",
        "oof_f1": oof_f1,
        "oof_auc": oof_auc,
        "composite": composite,
    },
)

# Register to DuckDB Ledger
with Ledger() as ledger:
    ledger.log_experiment(
        branch_name=VARIANT_NAME,
        oof_score=composite,
        metric=str(config.get("metric", "multi")).lower(),
        feature_count=len(feature_cols),
        calibration_method="none",
        gate_result="PASS",
        gate_reason=f"Residual Specialist Variant {VARIANT_NAME}. F1@0.5={oof_f1:.4f}, AUC={oof_auc:.4f}, Composite={composite:.4f}",
        dag_phase="phase_3a_specialist_trained",
        notes=f"oof_f1={oof_f1:.6f}; oof_auc={oof_auc:.6f}; composite={composite:.6f}",
    )

print("✓ Refit and registration complete!")
