#!/usr/bin/env python3
"""Write per-OOF metadata JSON files alongside existing OOF CSVs for provenance.

For each competition under `competitions/`, find files matching:
 - data/processed/oof_*.csv
 - reports/oof_*.csv
and create a sibling file `*.meta.json` with fields:
 - generated_by: best-effort skill name
 - cv_strategy_id: from SKILL_STATE keys if present (last_oof_cv_strategy_id, anchor_cv_strategy_id, etc.)
 - file_md5: md5 checksum of the CSV
 - rows: number of rows
 - created_at: isoformat now
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path.cwd()


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


for comp in (ROOT / 'competitions').iterdir():
    if not comp.is_dir():
        continue
    state_path = comp / 'SKILL_STATE.json'
    if not state_path.exists():
        print(f'[WARN] Missing SKILL_STATE.json for {comp.name} — skipping')
        continue
    state = json.loads(state_path.read_text())
    proc_dir = comp / 'data' / 'processed'
    reports_dir = comp / 'reports'
    candidates = []
    if proc_dir.exists():
        candidates.extend(sorted(proc_dir.glob('oof_*.csv')))
    if reports_dir.exists():
        candidates.extend(sorted(reports_dir.glob('oof_*.csv')))

    if not candidates:
        print(f'[INFO] No OOF files found for {comp.name}')
        continue

    for f in candidates:
        meta_path = f.with_suffix(f.suffix + '.meta.json')
        if meta_path.exists():
            print(f'[SKIP] Meta exists: {meta_path.name}')
            continue
        # best-effort cv id
        cv_id = state.get('last_oof_cv_strategy_id') or state.get('anchor_cv_strategy_id')
        # rows
        try:
            import pandas as pd
            rows = int(pd.read_csv(f).shape[0])
        except Exception:
            rows = None
        payload = {
            'file': str(f.name),
            'generated_by': 'unknown',
            'cv_strategy_id': cv_id,
            'file_md5': md5(f),
            'rows': rows,
            'created_at': datetime.now(timezone.utc).isoformat(),
        }
        meta_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        print(f'[WROTE] {meta_path.relative_to(ROOT)}')

print('DONE')
