#!/usr/bin/env python3
"""
Audit script: sections 1.1-1.5 (Data integrity)
"""
import hashlib
import json
import sys
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
except Exception as e:
    print('ERROR: missing dependency:', e, file=sys.stderr)
    raise

root = Path('competitions/ey-frogs')
state_path = root / 'SKILL_STATE.json'

print('\n=== 1.1 MD5 Hash Verification ===')
try:
    state = json.loads(state_path.read_text())
    files = {
        'md5_train_file':      root / 'data' / 'raw' / 'Training_Data.csv',
        'md5_test_file':       root / 'data' / 'raw' / 'Test.csv',
        'md5_sample_sub_file': root / 'data' / 'raw' / 'SampleSubmission.csv',
        'md5_target_hash':     None,
    }
    for key, path in files.items():
        if path is None:
            continue
        locked = state.get(key)
        try:
            actual = hashlib.md5(path.read_bytes()).hexdigest()
        except Exception as e:
            actual = f'ERROR: {e}'
        match = '✅' if locked == actual else '❌ MISMATCH'
        print(f"{match} {key}")
        print(f"   locked : {locked}")
        print(f"   actual : {actual}")
except Exception:
    import traceback
    traceback.print_exc()

print('\n=== 1.2 Target Column Hash Verification ===')
try:
    state  = json.loads(state_path.read_text())
    locked = state.get('md5_target_hash')
    train  = pd.read_csv(root / 'data' / 'raw' / 'Training_Data.csv')
    target_cols = [c for c in train.columns if c not in ('ID', 'Latitude', 'Longitude')]
    print(f"Candidate target columns: {target_cols}")
    for col in target_cols:
        actual = hashlib.md5(train[col].astype(str).str.cat(sep=",").encode()).hexdigest()
        match = '✅' if actual == locked else '❌'
        print(f"{match} {col}: {actual}")
    print(f"Locked hash: {locked}")
except Exception:
    import traceback
    traceback.print_exc()

print('\n=== 1.3 Training Data Shape and Class Balance ===')
try:
    train  = pd.read_csv(root / 'data' / 'raw' / 'Training_Data.csv')
    test   = pd.read_csv(root / 'data' / 'raw' / 'Test.csv')
    sample = pd.read_csv(root / 'data' / 'raw' / 'SampleSubmission.csv')
    print(f"Train shape   : {train.shape}")
    print(f"Test shape    : {test.shape}")
    print(f"Sample shape  : {sample.shape}")
    print(f"\nTrain columns : {list(train.columns)}")
    print(f"Test columns  : {list(test.columns)}")
    print(f"Sample columns: {list(sample.columns)}")
    target = 'Occurrence Status'
    if target in train.columns:
        vc = train[target].value_counts()
        print(f"\nClass balance:")
        print(f"  Class 0 (absent) : {vc.get(0,0)} ({vc.get(0,0)/len(train)*100:.1f}%)")
        print(f"  Class 1 (present): {vc.get(1,0)} ({vc.get(1,0)/len(train)*100:.1f}%)")
        try:
            ratio = vc.get(1,0)/vc.get(0,1)
        except Exception:
            ratio = 'N/A'
        print(f"  Imbalance ratio  : {ratio}")
    print(f"\nMissing values in train:")
    print(train.isnull().sum()[train.isnull().sum() > 0])
    print(f"\nMissing values in test:")
    print(test.isnull().sum()[test.isnull().sum() > 0])
except Exception:
    import traceback
    traceback.print_exc()

print('\n=== 1.4 TerraClimate Feature File Audit ===')
try:
    ft  = pd.read_csv(root / 'data' / 'processed' / 'features_train.csv')
    ftt = pd.read_csv(root / 'data' / 'processed' / 'features_test.csv')
    print(f"features_train shape: {ft.shape}")
    print(f"features_test shape : {ftt.shape}")
    print(f"\nLat/Lon present in train: {'Latitude' in ft.columns}")
    print(f"Lat/Lon present in test : {'Latitude' in ftt.columns}")
    tc_cols = [c for c in ft.columns if c not in ('ID', 'Latitude', 'Longitude', 'Occurrence Status')]
    print(f"\nTC feature count: {len(tc_cols)}")
    print(f"TC features: {tc_cols}")
    nulls = ft[tc_cols].isnull().sum()
    if nulls.sum() > 0:
        print(f"\n❌ Nulls in TC features:")
        print(nulls[nulls > 0])
    else:
        print(f"\n✅ No nulls in TC features")
    print(f"\nTC feature value ranges:")
    for col in tc_cols[:5]:
        print(f"  {col}: [{ft[col].min():.3f}, {ft[col].max():.3f}]  mean={ft[col].mean():.3f}")
    train_ids = set(ft['ID'])
    test_ids  = set(ftt['ID'])
    overlap   = train_ids & test_ids
    print(f"\nID overlap between train and test: {len(overlap)}")
    if overlap:
        print(f"❌ CRITICAL: {len(overlap)} IDs appear in both train and test")
    else:
        print(f"✅ No ID overlap — train and test are separate")
except Exception:
    import traceback
    traceback.print_exc()

print('\n=== 1.5 TerraClimate Extraction Verification ===')
try:
    ft = pd.read_csv(root / 'data' / 'processed' / 'features_train.csv')
    tc_cols = [c for c in ft.columns if c not in ('ID', 'Latitude', 'Longitude', 'Occurrence Status')]
    print('Checking for constant or near-constant TC columns:')
    suspicious = []
    for col in tc_cols:
        std = ft[col].std()
        unique = ft[col].nunique()
        if std < 0.001 or unique < 10:
            suspicious.append((col, std, unique))
            print(f"  ⚠️  {col}: std={std:.6f}, unique={unique}")
    if not suspicious:
        print('  ✅ All TC columns have meaningful variance')
    print('\nChecking for all-zero columns:')
    zeros = [c for c in tc_cols if (ft[c] == 0).all()]
    if zeros:
        print(f"  ❌ All-zero columns: {zeros}")
    else:
        print('  ✅ No all-zero columns')
except Exception:
    import traceback
    traceback.print_exc()
