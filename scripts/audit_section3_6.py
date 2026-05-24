#!/usr/bin/env python3
"""
Audit script: sections 3.1-6.2
Text-only / state-only checks, no ML training.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

root = Path('competitions/ey-frogs')
state = json.loads((root / 'SKILL_STATE.json').read_text())
log = (root / 'reports' / 'submission_log.md').read_text()

print('=== 3.1 Lat/Lon scan in skill files ===')
patterns = [r'feature_cols.*Latitude', r'feature_cols.*Longitude', r'Latitude.*model', r'Longitude.*model', r'X.*Latitude', r'X.*Longitude']
for file_path in [Path('zindian/skills/skill_07_features.py'), Path('zindian/skills/skill_08_anchor.py'), Path('zindian/skills/skill_05_cv.py')]:
    print(f'-- {file_path} --')
    text = file_path.read_text()
    hits = []
    for i, line in enumerate(text.splitlines(), 1):
        if any(re.search(pattern, line) for pattern in patterns):
            hits.append((i, line.strip()))
    if hits:
        for i, line in hits:
            print(f'  {i}: {line}')
    else:
        print('  ZERO RESULTS')

print('\n=== 3.2 VARIANTS dict / variant lines ===')
text_lines = Path('zindian/skills/skill_07_features.py').read_text().splitlines()
for i, line in enumerate(text_lines, 1):
    if 'variant-' in line and ':' in line:
        flag = '❌ BANNED FEATURE' if ('Latitude' in line or 'Longitude' in line) else '✅'
        print(f'  {flag} Line {i}: {line.strip()}')

print('\n=== 3.3 Seed injection scan ===')
for i, line in enumerate(text_lines, 1):
    if re.search(r'SEED|random_state|random_seed|np\.random\.seed', line) and not line.strip().startswith('#'):
        print(f'  {i}: {line.rstrip()}')

print('\n=== 3.4 Hardcoded path scan ===')
for file_path in [Path('zindian/skills/skill_07_features.py'), Path('zindian/skills/skill_08_anchor.py'), Path('zindian/skills/skill_16_submit.py')]:
    for i, line in enumerate(file_path.read_text().splitlines(), 1):
        if re.search(r'ey-frogs|/home/adrian|C:\\\\Users|Training_Data\.csv|Test\.csv', line):
            print(f'  {file_path.name} {i}: {line.rstrip()}')

print('\n=== 3.5 Submit gate presence in skill_16 ===')
for i, line in enumerate(Path('zindian/skills/skill_16_submit.py').read_text().splitlines(), 1):
    if re.search(r'validate|gate|human|YES|remaining', line):
        print(f'  {i}: {line.rstrip()}')

print('\n=== 3.6 Prob vs labels in skill_08 ===')
for i, line in enumerate(Path('zindian/skills/skill_08_anchor.py').read_text().splitlines(), 1):
    if 'submissions_dir' in line or 'sub_path' in line or 'threshold' in line or 'astype(int)' in line:
        print(f'  {i}: {line.rstrip()}')

print('\n=== 4.1 SKILL_STATE field audit ===')
required_fields = {
    'competition': str,
    'md5_target_hash': str,
    'current_git_branch': str,
    'anchor_git_branch': str,
    'anchor_oof_rmse': float,
    'anchor_lb_score': float,
    'feature_round': int,
    'variants_tested': int,
    'variants_passed': int,
    'dag_phase': str,
    'human_gate_1_approved': bool,
    'human_gate_2_by_branch': dict,
    'human_gate_3_approved': bool,
    'human_gate_4_approved': bool,
    'human_gate_5_selection': list,
    'selected_submissions': list,
    'submissions_used_today': int,
    'submissions_used_total': int,
    'remaining_submissions': int,
    'anchor_oof_auc': float,
    'anchor_oof_f1': float,
    'anchor_rank': int,
    'cv_strategy': str,
    'legality_status': str,
    'anchor_compliant': bool,
}
for field, expected_type in required_fields.items():
    val = state.get(field)
    if val is None:
        print(f'  ⚠️  {field}: NULL')
    elif not isinstance(val, expected_type):
        print(f'  ❌ {field}: type={type(val).__name__}, expected={expected_type.__name__}, value={val}')
    else:
        print(f'  ✅ {field}: {val}')

print('\n=== Human gate key presence ===')
for key in ['human_gate_1_approved', 'human_gate_3_approved', 'human_gate_4_approved', 'human_gate_5_selection']:
    val = state.get(key, None)
    if val is None:
        print(f'  ⚠️  {key}: NULL or missing')
    else:
        print(f'  ✅ {key}: present (type={type(val).__name__})')

# Check branch map
bmap = state.get('human_gate_2_by_branch')
if not isinstance(bmap, dict):
    print('  ❌ human_gate_2_by_branch: missing or not a dict')
else:
    print(f'  ✅ human_gate_2_by_branch entries: {len(bmap)}')

print('\n=== 4.2 selected_submissions audit ===')
selected = state.get('selected_submissions', [])
print(f'selected_submissions count: {len(selected)} (must be 2)')
print(f'selected IDs: {selected}')
if len(selected) != 2:
    print('❌ CRITICAL: Must select exactly 2 before May 19')
ids = re.findall(r'Submission ID[:\s]+([A-Za-z0-9]+)', log)
scores = re.findall(r'"rank":\s*(\d+)', log)
print(f'  Submission IDs found in log: {ids}')
print(f'  Ranks found in log: {scores}')

print('\n=== 4.3 Git state vs SKILL_STATE consistency ===')
print('Git current branch : anchor-v2')
print(f'State current_branch: {state.get("current_git_branch")}')
print(f'State anchor_branch : {state.get("anchor_git_branch")}')

print('\n=== 4.4 Submission budget consistency ===')
actual_subs = len(re.findall(r'^## Submission \[', log, re.MULTILINE))
state_total = state.get('submissions_used_total', 0)
print(f'Submissions in log   : {actual_subs}')
print(f'State submissions    : {state_total}')
if actual_subs != state_total:
    print(f'❌ MISMATCH: log={actual_subs} vs state={state_total}')
else:
    print('✅ Budget count consistent')

print('\n=== 5.1 Score progression ===')
blocks = re.split(r'## Submission \[', log)[1:]
print(f'Total submission blocks: {len(blocks)}')
print()
for block in blocks:
    lines = block.strip().splitlines()
    date = lines[0].rstrip(']') if lines else 'unknown'
    file_m = re.search(r'\*\*File\*\*: (.+)', block)
    rank_m = re.search(r'"rank":\s*(\d+)', block)
    score_m = re.search(r'"score":\s*([\d.]+)', block)
    comment = re.search(r'\*\*Comment\*\*: (.+)', block)
    print(f'Date    : {date}')
    print(f'File    : {file_m.group(1) if file_m else "unknown"}')
    print(f'Rank    : {rank_m.group(1) if rank_m else "not recorded"}')
    print(f'Score   : {score_m.group(1) if score_m else "not recorded"}')
    if comment:
        print(f'Comment : {comment.group(1)[:80]}')
    print()

print('=== 5.2 Best score verification ===')
print('State claims best LB:')
print(f'  Score      : {state.get("anchor_lb_score")}')
print(f'  Sub ID     : {state.get("anchor_lb_submission")}')
print(f'  File       : {state.get("anchor_lb_file")}')
print()
claimed_file = state.get('anchor_lb_file')
sub_path = root / 'submissions' / claimed_file
if sub_path.exists():
    sub = pd.read_csv(sub_path)
    print(f'  File exists: ✅ ({len(sub)} rows)')
    print(f'  Positives  : {(sub["Target"]==1).sum()}')
else:
    print(f'  File exists: ❌ MISSING — {sub_path}')

print('\n=== 6.1 Submission compliance ===')
sample = pd.read_csv(root / 'data' / 'raw' / 'SampleSubmission.csv')
sub_files = sorted((root / 'submissions').glob('*.csv'))
print(f'Checking {len(sub_files)} submission files:\n')
for sub_path in sub_files:
    sub = pd.read_csv(sub_path)
    pred_col = [c for c in sub.columns if c != 'ID'][0]
    vals = sub[pred_col]
    issues = []
    if list(sub.columns) != list(sample.columns):
        issues.append('column mismatch')
    if len(sub) != len(sample):
        issues.append(f'row count {len(sub)} ≠ {len(sample)}')
    if not set(vals.unique()).issubset({0, 1, 0.0, 1.0}):
        issues.append('non-binary values (probabilities)')
    if vals.isnull().any():
        issues.append('nulls found')
    if list(sub['ID'].astype(str)) != list(sample['ID'].astype(str)):
        issues.append('ID order mismatch')
    if issues:
        print(f'❌ {sub_path.name}')
        for issue in issues:
            print(f'   → {issue}')
    else:
        print(f'✅ {sub_path.name}')

print('\n=== 6.2 Lat/Lon clean in submission process ===')
for file_path in [Path('zindian/skills/skill_07_features.py'), Path('zindian/skills/skill_08_anchor.py'), Path('zindian/skills/skill_16_submit.py')]:
    print(f'-- {file_path} --')
    hits = []
    for i, line in enumerate(file_path.read_text().splitlines(), 1):
        if 'Latitude' in line or 'Longitude' in line:
            hits.append((i, line.rstrip()))
    if hits:
        for i, line in hits:
            print(f'  {i}: {line}')
    else:
        print('  ZERO RESULTS')
