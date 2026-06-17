# Zindian Orchestrator — Workspace Rules & Conventions

> **Purpose:** This document captures every structural convention, naming rule, workflow pattern, and architectural invariant that an AI agent must follow to maintain structural and logical integrity when working in this repository.
>
> **Audience:** AI coding agents, human developers, code reviewers.
>
> **Paired with:** `AGENTS.md` (agent system prompt), `docs/source_of_truth.md` (authoritative architecture spec)

---

## Table of Contents

1. Repository Topography
2. File & Module Naming Conventions
3. Import & Dependency Rules
4. Function Entry-Point Contracts
5. Config Access Patterns
6. State Access Patterns (Safe Access)
7. Phase Architecture & Pipeline Order
8. Skill Implementation Checklist
9. OOF Contract
10. SHAP Computation Rules
11. Seed Discipline
12. Two-Mode Feature Contract
13. Config Temporal Lock
14. Test Conventions
15. Dependency Management
16. CI/CD Workflow
17. Script Pipeline
18. Human Gates
19. Preflight Validation
20. Repository Hygiene Rules

---

## 1. Repository Topography

### Top-Level Layout

```
zindian_orchestrator/
├── zindian/                  # Main package (installable)
│   ├── __init__.py
│   ├── config.py             # ChallengeConfig reader & get_seed()
│   ├── constants.py          # Named constants (uppercase)
│   ├── cv.py                 # CV strategy factory
│   ├── ledger.py             # Audit ledger
│   ├── orchestrator.py       # Phase/skill runner + registry
│   ├── paths.py              # Path resolution
│   ├── schemas.py            # JSON schema validation
│   ├── state.py              # State file I/O
│   ├── zindi_client.py       # Zindi API client
│   ├── clients/              # External API clients
│   │   └── semantic_scholar.py
│   └── skills/               # All skill modules (22 skills)
│       ├── __init__.py       # Single line: """Skill modules (competition-aware)."""
│       ├── _lightgbm_shared.py  # Private shared helper (leading underscore)
│       ├── skill_00_zindi_monitor.py
│       ├── skill_01_integrity.py
│       ├── ... (skill_NN_name.py)
│       ├── skill_22_reproducibility_audit.py
│       └── skill_NN_reference.md  # Reference docs (paired with some skills)
├── tests/                    # All tests (flat, no subdirectories)
│   ├── conftest.py
│   ├── test_skillNN_name.py       # Skill-specific tests
│   ├── test_feature_or_policy.py  # Cross-cutting tests
│   └── ...
├── scripts/                  # CLI utility scripts
│   ├── bootstrap_competition.py
│   ├── compile_requirements.sh
│   ├── init_ledger.py
│   ├── inspect_zindi.py
│   ├── preflight_enforce.py
│   ├── test_phase_1.py
│   ├── verify_competition_state.py
│   ├── verify_phase_b.py
│   ├── write_oof_meta.py
│   └── zindian_audit.sh
├── docs/                           # Documentation
│   ├── source_of_truth.md        # THE authoritative spec (2521 lines)
│   ├── architecture_matrix.md
│   ├── orchestrator_current_state.md
│   ├── refactor_reports/
|   └── session_logs/
├── templates/                # JSON templates
│   ├── challenge_config_template.json
│   └── SKILL_STATE_template.json
├── specs/                    # Design specs
│   ├── design.md
│   ├── requirements.md
│   └── tasks.md
├── plugins/                  # Optional plugins
│   ├── __init__.py
│   └── terraclimate_extractor.py
├── tabula/                   # CLI entry-point package
│   ├── __init__.py
│   ├── __main__.py
│   └── init.py
├── .github/workflows/        # CI (GitHub Actions)
│   └── ci.yml
├── reports/                  # Runtime outputs (git-ignored)
├── data/                     # Competition data (git-ignored)
├── submissions/              # Submission files (git-ignored)
├── .gitignore
├── AGENTS.md                 # System prompt for coding agents
├── CLAUDE.md                 # Operator guide (session prompts)
├── README.md
├── setup.py
├── requirements.in           # Unpinned deps (source of truth)
├── requirements.txt          # Pinned deps (generated via pip-compile)
└── pyrightconfig.json
```

### Ignored Paths (from `.gitignore`)

The following are **never committed** — they are competition-specific or generated artifacts:

- `.env` — credentials
- `data/raw/`, `data/processed/` — competition datasets
- `submissions/` — submission CSV files
- `SKILL_STATE.json` — execution state
- `challenge_config.json` — competition config (generated per competition)
- `competitions/*/data/`, `competitions/*/submissions/`, `competitions/*/reports/`, `competitions/*/SKILL_STATE.json`, `competitions/*/challenge_config.json`
- `.venv/`, `__pycache__/`, `*.pyc`, `*.egg-info/`
- `.vscode/`, `.idea/`
- `.env` is on disk but **must never be tracked** — `.gitignore` protects it

---

## 2. File & Module Naming Conventions

### Skill Files

```
Format:  skill_{NN}_{name}.py
Example: skill_04_eda.py, skill_11_gate.py
Rules:
  - NN is a zero-padded two-digit number: 00, 01, 02, ..., 22
  - name is lowercase snake_case
  - No gaps in numbering (currently 00–22, all contiguous)
```

### Private Modules

```
Format:  _{name}.py
Example: _lightgbm_shared.py
Rules:
  - Leading underscore signals "internal to skills package"
  - No skill number prefix
  - Contains shared logic (model training, CV loop) consumed by multiple skills
```

### Reference Docs

```
Format:  skill_{NN}_reference.md
Example: skill_01_reference.md
Rules:
  - Paired with skill modules (not all skills have them)
  - Contains design notes, literature references, or implementation guidance
```

### Test Files

```
Format:  test_skill{NN}_{name}.py  OR  test_{feature_or_policy}.py
Examples:
  - test_skill04_eda.py          (skill-specific test)
  - test_oof_schema.py           (cross-cutting feature test)
  - test_cv_policy.py            (policy/contract test)
  - test_deep_research_scaffolds.py  (integration test)
Rules:
  - Flat directory — no subdirectories in tests/
  - One `conftest.py` at tests/ root (no nested conftest files)
```

### Script Files

```
Examples: bootstrap_competition.py, init_ledger.py, zindian_audit.sh
Rules:
  - Lowercase snake_case
  - `.py` for Python, `.sh` for shell
  - No numeric prefix
```

### Package Module Files

```
Examples:
  - zindian/config.py          (snake_case, no prefix)
  - zindian/zindi_client.py    (snake_case)
  - zindian/clients/semantic_scholar.py  (sub-package)
  - plugins/terraclimate_extractor.py    (plugin module)
Rules:
  - Lowercase snake_case throughout
  - No abbreviated names unless domain-standard (cv, eda, shap, oof)
```

### Constant Names

```python
# In zindian/constants.py:
# UPPER_SNAKE_CASE for module-level constants
COMPETITION_DIRNAME = "competitions"
SKILL_STATE_FILENAME = "SKILL_STATE.json"
CHALLENGE_CONFIG_FILENAME = "challenge_config.json"
```

---

## 3. Import & Dependency Rules

### Hard Blocks

| Rule | Description | Enforced By |
|------|-------------|-------------|
| **No cross-skill imports** | No skill module may import from another skill module directly. `from .skill_04 import X` is forbidden. | Static scan + `test_cross_skill_policy.py` |
| **No AutoML** | No AutoML library imports in any skill body: `auto-sklearn`, `TPOT`, `H2O`, `AutoGluon`, etc. Includes "just for feature selection." | Static scan + preflight |
| **No hardcoded competition strings** | No string literals for column names, target names, metric names, coordinate names, dataset names, competition identifiers. Always read from config. | Static scan + preflight |
| **No unlisted packages** | Every import must resolve to a package in `requirements.txt`. If a package is missing, raise the issue — do not add it without confirmation. | CI verification |
| **No private packages** | No custom or unlisted packages. | Code review |

### Allowed Import Patterns

```python
# ✅ Correct — imports from framework, not from other skills
from zindian.config import ChallengeConfig, get_seed
from zindian.paths import resolve_competition_paths
from zindian.state import SKILL_STATE  # via state module
from zindian.cv import get_cv_splits
from zindian.constants import COMPETITION_DIRNAME
from zindian.skills._lightgbm_shared import train_lightgbm_cv  # private shared helper

# ✅ Correct — standard library
import json
from pathlib import Path
from typing import Any, Dict

# ✅ Correct — third-party (must be in requirements.txt)
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
import shap

# ❌ WRONG — cross-skill import
from zindian.skills.skill_04_eda import compute_target_std

# ❌ WRONG — AutoML
from autosklearn.classification import AutoSklearnClassifier
```

### Shared Helpers Pattern

The only shared code that skills may import is `_lightgbm_shared.py`:

```python
# Any OOF-generating skill may import:
from zindian.skills._lightgbm_shared import train_lightgbm_cv

# This module contains the canonical CV training loop with:
# - Seed discipline enforcement
# - OOF schema compliance
# - Fold score collection
# - CV strategy override awareness
```

---

## 4. Function Entry-Point Contracts

### Skill Entry Point

Every skill module exposes **exactly one** public entry-point function:

```python
def run(config: dict, state: dict) -> dict:
    """
    One-line description of what the skill does.

    Reads: state["parent_key"]["field"]
    Writes: state["output_key"]
    """
    # Implementation
    state["output_key"] = { ... }
    return state
```

**Rules:**
- Signature is always `run(config: dict, state: dict) -> dict`
- Always returns the updated state dict (never returns None or raises for normal flow)
- No skill holds internal state between calls (stateless design)
- Docstring must document what it reads and writes (SoT contract)
- The `config` dict comes from `challenge_config.json`
- The `state` dict comes from `SKILL_STATE.json`

### Orchestrator Entry Points

```python
# Run a single skill by name
run_skill(skill_name: str, **kwargs) -> Dict[str, Any]

# Run all skills for a given phase
run_phase(phase: int, **kwargs) -> Dict[str, Any]

# Run research pipeline (skills 18, 19, 20)
run_deep_research(domain: str = "geospatial", dry_run: bool = False, **kwargs) -> Dict[str, Any]
```

### CLI Entry Point

```python
# setup.py defines:
entry_points = {
    "console_scripts": [
        "tabula=tabula.__main__:main",
    ]
}
```

---

## 5. Config Access Patterns

### ChallengeConfig Class

All config access goes through `zindian.config.ChallengeConfig`:

```python
from zindian.config import ChallengeConfig, get_seed

cfg = ChallengeConfig.load()

# Safe read with default
value = cfg.get("key", default_value)

# Required read (raises ConfigNotPopulated if null)
value = cfg.get_required("key")

# Property access for well-known fields
metric = cfg.metric              # cfg.get_required("metric")
direction = cfg.metric_direction # cfg.get_required("metric_direction")
use_probs = cfg.use_probabilities
slug = cfg.slug

# Dict-like access (KEY ERROR if missing)
value = cfg["key"]

# Seed from config
seed = get_seed(default=42)  # reads config["reproducibility"]["seed"]
```

### Nested Config Access Pattern

```python
# CV strategy
cv_strategy = config.get("cv_strategy", {})
cv_type = cv_strategy.get("type", "KFold")

# Reproducibility
seed = config.get("reproducibility", {}).get("seed", 42)

# Spatial signal
spatial = config.get("spatial_signal", {})
lat_col = spatial.get("lat_col")
lon_col = spatial.get("lon_col")

# Submission budget
budget = config.get("submission_budget", {})
total = budget.get("total", 0)
daily = budget.get("daily", 0)
used = budget.get("used", 0)
```

---

## 6. State Access Patterns (Safe Access)

### Core Rule

**Never use direct bracket access on optional state keys — always use `.get()` with defaults.**

```python
# ✅ CORRECT — safe access
override_active = SKILL_STATE.get(
    "cv_strategy_override", {}
).get("active", False)

# ❌ WRONG — crashes on first run before keys are populated
override_active = SKILL_STATE["cv_strategy_override"]["active"]
```

### Mandatory Safe Access Patterns

```python
# CV strategy override (all OOF-generating skills)
override_active = SKILL_STATE.get(
    "cv_strategy_override", {}
).get("active", False)

# Pseudo-label retraining check (skill_11 gate condition 3)
retraining_active = SKILL_STATE.get(
    "pseudo_label_result", {}
).get("retraining_required", False)

# Anchor challenge check (skill_11 gate condition 3)
challenge_active = SKILL_STATE.get(
    "anchor_challenge", {}
).get("active", False)

# Drift threshold (skill_00)
drift_threshold = SKILL_STATE.get(
    "drift_threshold",
    config.get("drift_threshold", 0.05)
)

# Sidecar recommendations (all consuming skills)
sidecar_recommendations = SKILL_STATE.get(
    "sidecar_recommendations", []
)

# Sidecar consumption guard
if not sidecar_recommendations:
    log("No sidecar recommendations — proceeding from fingerprint")
else:
    log(f"Sidecar recommendations consumed: {len(sidecar_recommendations)} items")
```

### State Write Location

All execution state is written to `SKILL_STATE.json`. No skill holds internal state between runs.

---

## 7. Phase Architecture & Pipeline Order

### Five-Phase Sequence

```
Phase 1:  Competition Fingerprint + Config Lock
Phase 2A: Data Cleaning (policy_gate → skill_06)
Phase 2B: Feature Engineering + Anchor (skill_07 → skill_08)
Phase 3:  Branch Training + SHAP + Calibration (skill_09, skill_10)
Phase 4:  Gate + Submit (skill_11 → skill_16)
Phase 5:  Fusion + Final Submit (skill_13 → skill_14 → skill_17)
```

### Orchestrator Phase-Skill Mapping (Hardcoded Default)

```python
PHASE_1_SKILLS = ["skill_01", "skill_02", "skill_15"]   # + skill_03, skill_04, skill_05
PHASE_2_SKILLS = ["skill_03", "skill_08"]                # incomplete — also skill_06, skill_07
PHASE_3_SKILLS = ["skill_04", "skill_05", "skill_09", "skill_10"]
PHASE_4_SKILLS = ["skill_11", "skill_16"]
PHASE_5_SKILLS = ["skill_13", "skill_14", "skill_17"]
```

**Note:** The hardcoded lists in `orchestrator.py` do not fully match the SoT Phase Architecture. The SoT defines:

```
Phase 1:  skill_01 → skill_02 → skill_03 → skill_04 → skill_05 → skill_15
Phase 2A: policy_gate() → skill_06
Phase 2B: skill_07 → skill_08
Phase 3A: skill_09 → skill_10
Phase 3B: skill_11 → skill_12 → skill_22
Phase 4:  skill_11 (branch gate) → skill_16 (submit)
Phase 5:  skill_13 → skill_14 → skill_17
```

The `challenge_config.json` `phase_skill_map` overrides the hardcoded lists when present.

### Execution Order Within Phases

Phases execute in cascading sequence. Within a phase, skills run in the order listed. The orchestrator enforces:

- Complete resolution of Phase 1 before Phase 2A
- Phase 2A before Phase 2B
- Phase 3A before Phase 3B
- This is a **structural system dependency**, not a sequential preference

### Variant Branch Lifecycle

```
skill_07 generates features
    ↓
skill_08 trains anchor model (single branch: "anchor")
    ↓
Human Gate 1 approves
    ↓
skill_09 trains variant branches (branches: "anchor", "branch_1", "branch_2", ...)
    ↓
skill_10 computes SHAP per branch
    ↓
skill_11 gates each branch against anchor
    ↓
Promoted branches → Human Gate 2 per branch
    ↓
skill_16 submits promoted branches
    ↓
skill_13 fuses selected branches (after Human Gate 3)
    ↓
skill_14 formats inference (after Human Gate 4)
    ↓
Human Gate 5 selects 2 submissions for private LB
```

---

## 8. Skill Implementation Checklist

Every skill must satisfy its corresponding DoD checklist in SoT Section 8. The checklist items common to ALL skills:

- [ ] `def run(config: dict, state: dict) -> dict` signature is correct
- [ ] Docstring documents reads/writes per SoT contract
- [ ] No cross-skill imports
- [ ] No hardcoded competition strings
- [ ] No AutoML imports
- [ ] Safe state access patterns (`.get()` with defaults) used throughout
- [ ] Config accessed via `ChallengeConfig` or config dict accessor
- [ ] Seed read from config (never hardcoded)
- [ ] OOF schema compliance (if OOF-generating)
- [ ] Config temporal lock respected (no writes to challenge_config.json after Phase 1)
- [ ] Returns updated state dict

---

## 9. OOF Contract

### Universal Rules

1. Every skill that generates OOF scores must tag them with `cv_strategy_id`
2. Every skill that reads OOF scores must validate that tag
3. No skill defines its own CV object — all use the strategy from `challenge_config.json` or `cv_strategy_override` in `SKILL_STATE.json`
4. A contract violation is a hard halt — not a warning

### OOF Output Schema

```python
SKILL_STATE[f"branch_{branch_name}_oof"] = {
    "scores": oof_array.tolist(),          # List[float]
    "cv_strategy_id": config["cv_strategy"]["type"],  # Matches CV strategy
    "seed": config["reproducibility"]["seed"],
    "branch_name": branch_name,
    "model_config": model_config_dict,     # Hyperparameters used
}
```

### Augmented OOF (Pseudo-Label Loop)

```python
SKILL_STATE[f"branch_{branch_name}_oof_augmented"] = {
    "scores": oof_array.tolist(),
    "cv_strategy_id": config["cv_strategy"]["type"],
    "seed": config["reproducibility"]["seed"],
    "branch_name": branch_name,
    "model_config": model_config_dict,
}
```

### Overwrite Protection

```python
key = f"branch_{branch_name}_oof"
if key in SKILL_STATE and retraining_active:
    raise RuntimeError(
        f"Retraining loop attempted to overwrite original OOF key: "
        f"{key}. Write to '{key}_augmented' instead."
    )
```

---

## 10. SHAP Computation Rules

### Correct Pattern

SHAP must be computed **per-fold on validation fold predictions only**:

```python
shap_arrays = []
for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
    model.fit(X[train_idx], y[train_idx])
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X[val_idx])
    shap_arrays.append(np.abs(shap_values).mean(axis=0))

mean_shap = np.mean(shap_arrays, axis=0)
```

**Full-train SHAP is prohibited.**

### Single-Feature Fallback

If `X.shape[1] < 2`, skip the ratio audit and write:

```python
SKILL_STATE["shap_audit_skipped_reason"] = "single_feature"
```

Then proceed to `skill_11` gating without a `leaked_features` entry. The branch is not promoted automatically — all other gate conditions still apply.

### Leak Audit

```python
leaked = []
for i, feature in enumerate(feature_names):
    ratio = max_shap_values[i] / mean_shap[i]
    if ratio > config.get("shap_leak_threshold", 3.0):
        leaked.append(feature)

if leaked:
    SKILL_STATE["leaked_features"] = leaked
```

---

## 11. Seed Discipline

Every model training call sets three seeds, all from config:

```python
seed = config["reproducibility"]["seed"]
random.seed(seed)
np.random.seed(seed)
model = LGBMClassifier(random_state=seed, ...)
```

**Rules:**
- Seed is never overridden locally — always read from config
- `random.seed(seed)` for Python's random module
- `np.random.seed(seed)` for NumPy
- `model(random_state=seed)` for ML model
- Verify with `zindian.config.get_seed()`

---

## 12. Two-Mode Feature Contract

Target-dependent features must implement two computation modes:

```python
def compute_spatial_lag(X, y, train_idx=None, mode="cv"):
    """
    mode="cv"        — use train_idx targets only (fold-restricted)
    mode="inference" — use all y targets (final model fit)
    """
    if mode == "cv":
        assert train_idx is not None
        target_values = y[train_idx]
        spatial_subset = X[train_idx]
    else:
        target_values = y
        spatial_subset = X
    # ... compute lag using target_values and spatial_subset
```

Structural features (Haversine distance, nearest-neighbour arrays, non-target group counts) do NOT require two-mode treatment and may be computed on the full dataset at any time.

---

## 13. Config Temporal Lock

### Phase 1 Mutable Window

Only these skills may write to `challenge_config.json` during Phase 1:

| Skill | Writes |
|-------|--------|
| `skill_01` | `file_hashes` |
| `skill_02` | fingerprint fields, seed, submission_budget, target_domain_bounds |
| `skill_03` | `policy_filters` (via `policy_writer()`) |
| `skill_05` | `cv_strategy` block |
| `skill_00` | `community_signals` (at ANY time — sole exception) |

### Post-Phase 1 Lock

- `challenge_config.json` becomes **strictly read-only** after Phase 1 gate passes
- No core skill may write to it after this point
- Any attempted write by a non-permitted skill is a hard error — written to SKILL_STATE.json and pipeline halts
- **Sole exception:** `skill_00` may write to `community_signals` at any time

### Policy Enforcement

```python
# Allowed writers (Phase 1 only):
ALLOWED_CONFIG_WRITERS = frozenset([
    "skill_00",  # community_signals only — any phase
    "skill_01",  # Phase 1 — file_hashes
    "skill_02",  # Phase 1 — fingerprint fields
    "skill_03",  # Phase 1 — policy_filters
    "skill_05",  # Phase 1 — cv_strategy
])
```

---

## 14. Test Conventions

### Test File Layout

```
tests/
  conftest.py                        # Shared fixtures (no nested conftest)
  test_skill04_eda.py                # Test specific skill
  test_skill05_cv_architect.py       # Test skill CV logic
  test_skill05_spatial_fallback.py   # Test spatial edge case
  test_oof_schema.py                 # Test OOF output contract
  test_cv_policy.py                  # Test CV strategy policy
  test_cross_skill_policy.py         # Test no cross-skill imports
  test_challenge_config_write_policy.py  # Test config temporal lock
  test_skill_state_safe_access.py    # Test .get() patterns
  test_fetch_guard.py                # Test network guard
  test_seed_discipline.py            # Test seed reproducibility
  test_shap_per_fold.py              # Test per-fold SHAP
  test_shap_audit_unit.py            # Test SHAP leak audit
  test_phase1_guard_and_integrity.py # Test Phase 1 gate
  test_phase2_anchor_oof.py          # Test Phase 2 OOF
  test_phase4_integration.py         # Test Phase 4 pipeline
  test_phase5_fusion_and_submit.py   # Test Phase 5 pipeline
  test_calibration_flow.py           # Test calibration
  test_calibration_writes_oof.py     # Test OOF write contract
  test_skill_coverage.py             # Test all skills exist
  test_deep_research_scaffolds.py    # Test research pipeline
  test_config_write_policy.py        # Test config write permissions
  test_features_contracts.py         # Test feature contracts
  test_lightgbm_runresult_schema.py  # Test LightGBM schema
  test_skill00_monitor.py            # Test skill_00
  test_skill00_monitor_edgecases.py  # Test skill_00 edge cases
  test_skill01_integrity_target_override.py  # Test target override
  test_skill02_intake.py             # Test skill_02
  test_skill02_intake_edgecases.py   # Test skill_02 edge cases
  test_skill03_legality.py           # Test skill_03
  test_skill03_matching.py           # Test skill_03 matching
  test_skill10_shap_schema.py        # Test SHAP schema
  test_skill11_gate.py               # Test gate logic
  test_train_variant_monkeypatch.py  # Test variant training
```

### Test Naming Convention

```python
# File name: test_skillNN_name.py
# Class name: TestSkillNNName
# Function name: test_[what_it_tests]

# Example:
class TestSkill04Eda:
    def test_target_std_computation(self):
        ...
    
    def test_mnar_detection(self):
        ...
```

### Conftest Scope

- Single `conftest.py` at `tests/` root
- No nested conftest files
- Shared fixtures: sample config, sample state, sample CV splits, sample data

---

## 15. Dependency Management

### Workflow

```
requirements.in  (unpinned — source of truth)
       ↓
pip-compile requirements.in --output-file requirements.txt
       ↓
requirements.txt  (pinned — committed to repo)
       ↓
pip install -r requirements.txt  (in fresh venv)
```

### Rules

- `requirements.in` is the human-maintained source (unpinned dependencies)
- `requirements.txt` is generated via `pip-compile` from `pip-tools`
- Both files are committed to the repository
- Before importing any package, verify it appears in `requirements.txt`
- CI verifies `requirements.txt` is not older than `requirements.in`
- CI verifies `pip-compile` output matches committed `requirements.txt`
- No AutoML libraries in any dependency list

### Current Core Dependencies

```
# From requirements.in:
google-genai
requests
numpy
pandas
scikit-learn
lightgbm
xgboost
shap
matplotlib
seaborn
pip-tools
```

---

## 16. CI/CD Workflow

### GitHub Actions (`.github/workflows/ci.yml`)

**Trigger:** Push to `main` or `refactor/sot`, PR targeting `main`

**Environment:** `ZINDIAN_DISABLE_NETWORK=1` (prevents network calls)

**Jobs:**

1. **`policy` job:** Runs policy contract tests:
   - `test_challenge_config_write_policy.py`
   - `test_cv_policy.py`
   - `test_cross_skill_policy.py`
   - `test_skill_state_safe_access.py`
   - `test_oof_schema.py`
   - `test_fetch_guard.py`
   - Verifies `requirements.txt` is up-to-date

2. **`test` job:** Runs full test suite via `pytest -q`

### Enforcing Rules Locally

```bash
# Run policy tests
pytest -q tests/test_challenge_config_write_policy.py \
          tests/test_cv_policy.py \
          tests/test_cross_skill_policy.py \
          tests/test_skill_state_safe_access.py \
          tests/test_oof_schema.py \
          tests/test_fetch_guard.py

# Run all tests
pytest -q

# Verify requirements freshness
pip-compile --output-file /tmp/expected_requirements.txt requirements.in
diff requirements.txt /tmp/expected_requirements.txt
```

---

## 17. Script Pipeline

### Order of Operations

```bash
# 0. Setup
scripts/compile_requirements.sh          # Generate requirements.txt from requirements.in

# 1. Bootstrap a new competition
scripts/bootstrap_competition.py         # Initialize competition directory + config

# 2. Initialize ledger
scripts/init_ledger.py                   # Create audit ledger

# 3. Inspect Zindi competition
scripts/inspect_zindi.py                 # Fetch competition metadata from Zindi API

# 4. Run preflight
scripts/preflight_enforce.py             # Validate environment before pipeline start

# 5. Run Phase 1 test
scripts/test_phase_1.py                  # Test Phase 1 skills sequence

# 6. Verify state
scripts/verify_competition_state.py      # Validate SKILL_STATE.json completeness

# 7. Write OOF metadata (if needed)
scripts/write_oof_meta.py                # Write OOF metadata to state

# 8. Audit
scripts/zindian_audit.sh                 # Full repository audit

# 9. Verify Phase B
scripts/verify_phase_b.py                # Phase B-specific verification
```

### Script Intent Contracts

| Script | Purpose | Requires |
|--------|---------|----------|
| `bootstrap_competition.py` | Create competition directory, populate config template | Empty competition directory |
| `compile_requirements.sh` | Run pip-compile to generate requirements.txt | `pip-tools` installed |
| `init_ledger.py` | Initialize audit ledger file | Existing config |
| `inspect_zindi.py` | Fetch competition metadata from Zindi API | API credentials in `.env` |
| `preflight_enforce.py` | Run full preflight validation | Populated config + state |
| `test_phase_1.py` | Execute Phase 1 skill sequence | Bootstrap complete |
| `verify_competition_state.py` | Validate state file completeness | Phase 1 complete |
| `write_oof_meta.py` | Write OOF metadata for reproducibility | OOF scores generated |
| `zindian_audit.sh` | Full repo audit (branch, state, coverage) | Git repository |
| `verify_phase_b.py` | Phase B-specific validation | Phase A complete |

---

## 18. Human Gates

Five human gates exist in the pipeline. Each requires an explicit approval key in `SKILL_STATE.json` before proceeding:

```python
# Gate keys (written ONLY by human operator, never by skills or orchestration)
human_gate_1_approved         # After anchor evaluation, before variant generation
human_gate_2_{branch}_approved  # Per promoted branch, before candidate pool entry
human_gate_3_approved         # Before skill_13 oracle fusion runs
human_gate_4_approved         # Before skill_14 inference formatting runs
human_gate_5_selection        # Final private LB submission pair confirmed
```

**Rules:**
- Gate keys are never written by any skill or by the orchestrator
- When any gate key is absent, the orchestrator halts, surfaces a human-readable prompt, and waits
- No timeout, retry, or bypass under any condition

---

## 19. Preflight Validation

### Two Modes

| Mode | Trigger | Purpose |
|------|---------|---------|
| **INIT** | `challenge_config.json` does not exist | Allow Phase 1 to populate config from scratch |
| **ENFORCE** | `challenge_config.json` exists and populated | Full validation before any skill runs |

### INIT Mode Checks

- Competition workspace directory exists and is writable
- Raw data files present in expected location
- No conflicting SKILL_STATE.json from a prior run
- Environment lock file present (requirements.txt)
- No AutoML library imports in any skill body (static scan)
- No cross-skill imports present (static scan)

### ENFORCE Mode Checks

- Config completeness (all required fields present)
- CV strategy block present
- SHAP/variance/gate thresholds set
- Submission budget present
- Seed present and set
- State integrity (valid JSON, file hashes match)
- OOF contract (all tags carry cv_strategy_id)
- Architecture integrity (no cross-skill imports, no hardcoded strings)
- Zindi compliance (no AutoML, probability format, seed set, budget > 0)
- Human gate status (all five reported)
- Config temporal lock active

### Output

- `reports/preflight_INIT_{timestamp}.json` or `reports/preflight_ENFORCE_{timestamp}.json`
- On pass: `SKILL_STATE.json["preflight_confirmed"] = true`

---

## 20. Repository Hygiene Rules

### What Must Be Committed

- All source code (`zindian/`, `tests/`, `scripts/`, `docs/`, `templates/`, `specs/`, `plugins/`, `tabula/`)
- `requirements.in` and `requirements.txt`
- `setup.py`, `.gitignore`, `pyrightconfig.json`
- `README.md`, `AGENTS.md`, `CLAUDE.md`
- `.github/workflows/` (CI config)

### What Must NEVER Be Committed

- `.env` (contains API credentials)
- `data/raw/`, `data/processed/` (competition data)
- `submissions/` (submission files)
- `SKILL_STATE.json`, `challenge_config.json` (per-competition generated files)
- `competitions/*/data/`, `competitions/*/submissions/`, `competitions/*/reports/`
- `.venv/`, `__pycache__/`, `*.pyc`, `*.pyo`
- `.pytest_cache/`, `.ipynb_checkpoints/`
- `*.egg-info/`
- `*.jsonl` (debug artifacts)

### Branch Hygiene

- Primary branch: `main` (GitHub default) or `master` (current local default)
- Experiment branches should be deleted after merging or when stale
- Experiment branches found: `anchor-baseline`, `anchor-v2`, `anchor-v3`, `anchor-v5`, `anchor-v6`, `exp-feature-aridity`, `exp-feature-desiccation`, `exp-feature-evap-fraction`, `exp-feature-srad`
- Current active work: `refactor/sot`
- No remote configured — remotes must be added before GitHub publish

### Code Review Gates

Every PR must satisfy:
- [ ] All DoD checklist items in SoT Section 8 pass for affected skills
- [ ] No cross-skill imports introduced (enforced by CI)
- [ ] No AutoML imports introduced (enforced by CI + static scan)
- [ ] Config temporal lock respected (enforced by CI)
- [ ] Safe state access patterns used (enforced by CI)
- [ ] OOF schema compliance (enforced by CI)
- [ ] All existing tests pass
- [ ] New tests added for any new functionality
- [ ] `requirements.txt` is up-to-date with `requirements.in`
- [ ] No hardcoded competition strings in skill bodies

### File Size Limits

- No file should exceed 10MB (larger files are likely data artifacts)
- `.venv/` binaries are excluded by `.gitignore`
- Data files (CSV, PKL, H5, zip, parquet, db, sqlite, jsonl) are excluded by `.gitignore`

---

*Generated by repository audit — captures all conventions observed across the codebase.*
*Paired with: `AGENTS.md`, `docs/source_of_truth.md`*
*Maintains structural and logical integrity for AI agent sessions.*