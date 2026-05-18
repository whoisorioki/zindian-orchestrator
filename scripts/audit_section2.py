#!/usr/bin/env python3
"""
Audit script: sections 2.1-2.4 (Model assumptions)
"""
import json
from pathlib import Path
import sys

try:
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, f1_score
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestClassifier
except Exception as e:
    print('ERROR: missing dependency:', e, file=sys.stderr)
    raise

root = Path('competitions/ey-frogs')
state = json.loads((root / 'SKILL_STATE.json').read_text())

print('\n=== 2.1 Recompute anchor OOF AUC ===')
try:
    claimed_auc = state.get('anchor_oof_auc')
    print(f"Claimed anchor OOF AUC: {claimed_auc}")
    ft = pd.read_csv(root / 'data' / 'processed' / 'features_train.csv')
    tc_cols = [c for c in ft.columns if c not in ('ID','Latitude','Longitude','Occurrence Status')]
    X = ft[tc_cols].values.astype('float32')
    y = ft['Occurrence Status'].values.astype('int32')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    for fold, (tr, val) in enumerate(skf.split(X, y)):
        model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=42)
        model.fit(X[tr], y[tr])
        oof[val] = model.predict_proba(X[val])[:,1]
        print(f"  Fold {fold+1}: AUC={roc_auc_score(y[val], oof[val]):.5f}")
    actual_auc = float(roc_auc_score(y, oof))
    best_t = max(np.arange(0.3, 0.7, 0.01), key=lambda t: f1_score(y, (oof >= t).astype(int)))
    actual_f1 = float(f1_score(y, (oof >= best_t).astype(int)))
    print(f"\nClaimed OOF AUC : {claimed_auc:.5f}")
    print(f"Actual  OOF AUC : {actual_auc:.5f}")
    print(f"Delta           : {actual_auc - claimed_auc:+.5f}")
    print(f"Actual  OOF F1  : {actual_f1:.5f}  (threshold: {best_t:.2f})")
    if abs(actual_auc - claimed_auc) > 0.002:
        print('❌ MISMATCH — state OOF AUC does not match recomputed value')
    else:
        print('✅ OOF AUC matches within tolerance')
except Exception:
    import traceback
    traceback.print_exc()

print('\n=== 2.2 Gate threshold scan (grep) ===')
# We'll just print whether 0.005 appears in relevant files
for p in ['zindian/skills/skill_07_features.py','zindian/skills/skill_11_gate.py']:
    path = Path(p)
    print(f"\n-- {p} --")
    if path.exists():
        s = path.read_text()
        for token in ['MIN_DELTA','min_delta','0.005','0.5%']:
            if token in s:
                print(f"  contains: {token}")
    else:
        print('  MISSING')

print('\n=== 2.3 Multi-seed consistency check ===')
try:
    ft = pd.read_csv(root / 'data' / 'processed' / 'features_train.csv')
    tc_cols = [c for c in ft.columns if c not in ('ID','Latitude','Longitude','Occurrence Status')]
    X = ft[tc_cols].values.astype('float32')
    y = ft['Occurrence Status'].values.astype('int32')
    SEEDS = [42,123,7]
    aucs=[]
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        oof = np.zeros(len(y))
        for fold, (tr,val) in enumerate(skf.split(X,y)):
            lgb_m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=seed)
            lgb_m.fit(X[tr], y[tr])
            rf_m = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, max_features='sqrt', random_state=seed)
            rf_m.fit(X[tr], y[tr])
            oof[val] = 0.5*lgb_m.predict_proba(X[val])[:,1] + 0.5*rf_m.predict_proba(X[val])[:,1]
        auc = float(roc_auc_score(y, oof))
        aucs.append(auc)
        print(f"Seed {seed}: OOF AUC = {auc:.5f}")
    import numpy as _n
    print(f"\nMean AUC : {_n.mean(aucs):.5f}")
    print(f"Std AUC  : {_n.std(aucs):.5f}")
    print('Claimed  : 0.84807')
    print(f"Delta vs claimed: {_n.mean(aucs)-0.84807:+.5f}")
except Exception:
    import traceback
    traceback.print_exc()

print('\n=== 2.4 Submission File Content Audit ===')
try:
    sample = pd.read_csv(root / 'data' / 'raw' / 'SampleSubmission.csv')
    subs = {
        'sub_011_anchor'      : root / 'submissions' / 'sub_011_anchor.csv',
        'variant-34'          : root / 'submissions' / 'variant-34_submission.csv',
        'variant-34b_t047'    : root / 'submissions' / 'variant-34b_t047_submission.csv',
    }
    for name,path in subs.items():
        print(f"\n── {name} ──")
        if not path.exists():
            print(f"  MISSING: {path}")
            continue
        sub = pd.read_csv(path)
        vals = sub['Target'] if 'Target' in sub.columns else sub[[c for c in sub.columns if c!='ID'][0]]
        print(f"  Rows        : {len(sub)}")
        print(f"  ID match    : {list(sub['ID']) == list(sample['ID'])}")
        is_binary = set(vals.dropna().unique()).issubset({0,1,0.0,1.0})
        print(f"  Value type  : {'binary 0/1' if is_binary else 'FLOAT — non-compliant'}")
        print(f"  Positives   : {(vals==1).sum()} ({(vals==1).mean()*100:.1f}%)")
        print(f"  Negatives   : {(vals==0).sum()} ({(vals==0).mean()*100:.1f}%)")
except Exception:
    import traceback
    traceback.print_exc()
