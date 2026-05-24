#!/usr/bin/env python3
"""Preflight ENFORCE check for reproducibility and environment lock.

Checks:
- `requirements.in` exists
- `requirements.txt` exists
- `challenge_config.json` contains `reproducibility.seed` and `cv_strategy` block
- `SKILL_STATE.json` contains the SoT v1.7 human gate keys

Usage:
  python scripts/preflight_enforce.py --competition competitions/ey-frogs

Exits 0 on success, 1 on failure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import argparse


def fail(msg: str):
    print(f"ERROR: {msg}")
    sys.exit(1)


def ok(msg: str):
    print(f"OK: {msg}")


parser = argparse.ArgumentParser()
parser.add_argument("--competition", default=None, help="Path to competition folder (e.g., competitions/ey-frogs)")
args = parser.parse_args()

root = Path.cwd()
# Check requirements files
req_in = root / "requirements.in"
req_txt = root / "requirements.txt"
if not req_in.exists():
    fail("requirements.in missing at project root")
else:
    ok("requirements.in present")

if not req_txt.exists():
    fail("requirements.txt missing — run: pip-compile requirements.in --output-file requirements.txt")
else:
    ok("requirements.txt present")

# Check competition config
if args.competition:
    comp_path = Path(args.competition)
else:
    # try to find single competition with non-empty folder
    comps = [p for p in (root / 'competitions').iterdir() if p.is_dir()]
    if not comps:
        fail("No competitions/ folder or empty")
    comp_path = comps[0]

config_path = comp_path / 'challenge_config.json'
if not config_path.exists():
    fail(f"challenge_config.json not found in {comp_path}")

cfg = json.loads(config_path.read_text())
# reproducibility
repro = cfg.get('reproducibility')
if not repro or not isinstance(repro, dict):
    fail('challenge_config.json missing `reproducibility` block')
ok('reproducibility block present')

seed = repro.get('seed')
if seed is None:
    fail('reproducibility.seed is missing')
ok(f'reproducibility.seed: {seed}')

cv = cfg.get('cv_strategy')
if not cv or not isinstance(cv, dict):
    fail('challenge_config.json missing `cv_strategy` block')
ok(f'cv_strategy present: {cv.get("type", "<unknown>")}')

# Check SKILL_STATE.json
state_path = comp_path / 'SKILL_STATE.json'
if not state_path.exists():
    fail(f'SKILL_STATE.json not found in {comp_path}')
state = json.loads(state_path.read_text())

# Validate SoT v1.7 human gate keys
required_gate_keys = {
    'human_gate_1_approved': bool,
    'human_gate_2_by_branch': dict,
    'human_gate_3_approved': bool,
    'human_gate_4_approved': bool,
    'human_gate_5_selection': list,
}
missing_gates = []
for k, t in required_gate_keys.items():
    v = state.get(k)
    if v is None:
        missing_gates.append(k)
    elif not isinstance(v, t):
        fail(f'SKILL_STATE.{k} has wrong type: {type(v).__name__}, expected {t.__name__}')

if missing_gates:
    fail(f'Missing required human gate keys: {missing_gates}')
ok('SKILL_STATE.json contains required human gate keys')

print('\nPRELIGHT ENFORCE: ALL CHECKS PASSED')
sys.exit(0)
