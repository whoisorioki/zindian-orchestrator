# Zindian Orchestrator

**Version:** 2.3  
**Status:** Production Ready  
**Last Updated:** June 2026
**License:** Apache 2.0

An **autonomous ML competition agent framework** for Zindi Africa competitions.

> **Framework, not specific competition** — Works for any Zindi competition by reading competition rules dynamically.

This framework manages end-to-end competition pipelines autonomously with human oversight at 5 critical checkpoints.

## What Is This?

Think of Zindian Orchestrator as an **intelligent assistant for data science competitions**. It systematically builds, tests, and submits machine learning models while following strict rules and requiring human approval at key decision points.

**Key Features:**
- **5 Human Gates** - Stops for approval at critical points
- **100% Reproducible** - Same data = identical results
- **Zero Hardcoding** - Reads competition rules dynamically
- **Carbon Tracking** - Measures environmental impact (R5)
- **Multi-Target Support** - Handles multiple prediction targets
- **Zindi Compliant** - No AutoML, full audit trail

## Architecture Overview

### Core Principles

1. **Three-Lens Decision Philosophy** — Every decision evaluated through General, Specific, and Generalization lenses
2. **Competition Agnosticism** — Zero hardcoded competition-specific values (Assumption A5)
3. **Atomic State Management** — Tempfile + os.replace prevents corruption
4. **Immutable Config** — Locked after Phase 1, read-only thereafter
5. **Human-in-the-Loop** — 5 mandatory approval gates
6. **Reproducibility Contract** — R1-R5 requirements enforced

### The 4 Main Phases

```
Phase 1: Competition Fingerprint (~5 min)        [COMPLETE]
  └─ Understand rules, lock config, select CV strategy

Phase 2: Anchor + Feature Search (~30-60 min)    [COMPLETE]
  └─ Build baseline, generate variants
  └─ [HUMAN GATE 1]: Review anchor

Phase 3: Generalization Audit (~60-120 min)      [COMPLETE]
  └─ SHAP leak detection, calibration, gating
  └─ [HUMAN GATE 2]: Approve variants (per branch)
  └─ [HUMAN GATE 3]: Approve fusion

Phase 4: Governance (~10 min)                    [COMPLETE]
  └─ Format, submit, audit reproducibility
  └─ [HUMAN GATE 4]: Approve inference
  └─ [HUMAN GATE 5]: Select final 2 submissions
```

**Total Runtime:** 2-3 hours per competition

---

## Project Structure

```
zindian-orchestrator/
├── competitions/                     ← per-competition workspace
│   └── <slug>/
│       ├── challenge_config.json
│       ├── SKILL_STATE.json
│       ├── data/
│       ├── notebooks/
│       └── reports/
├── docs/                            Documentation
│   ├── ORCHESTRATOR_OVERVIEW.md     Complete guide (start here)
│   ├── source_of_truth.md           Official specification v2.3
│   ├── sot_audit_report.md          Known gaps & issues
│   └── PROGRESS_TRACKER.md          v2.3 refactor status
├── specs/                           Technical specifications
│   ├── requirements.md              Functional requirements
│   ├── design.md                    Architecture & data flow
│   └── tasks.md                     Phase checklist
├── zindian/                         Python package
│   ├── state.py                     SKILL_STATE.json reader/writer
│   ├── config.py                    challenge_config.json reader
│   ├── ledger.py                    DuckDB wrapper
│   ├── cv.py                        CV splits generator
│   ├── orchestrator.py              Skill orchestration
│   └── skills/                      All 22 implemented skills
├── tabula/                          Competition bootstrapper CLI
├── scripts/                         Utility scripts
└── tests/                           Test suite (160+ tests)
```

---

## Quick Start

### 1. Understand the Architecture (5 min)

```bash
# Read the complete guide
cat docs/ORCHESTRATOR_OVERVIEW.md

# Read the official specification
cat docs/source_of_truth.md

# Read the CLI reference
cat docs/cli_integration_guide.md
```

### 2. Install & Verify (5 min)

```bash
# Activate venv (Unix)
source .venv/bin/activate

# Activate venv (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install dependencies from pinned file
pip install -r requirements.txt
```

### Manage Python dependencies (pinned)

This repository uses `requirements.in` plus `pip-compile` (from `pip-tools`) to produce a pinned `requirements.txt`.

```bash
# Install the compiler
pip install --upgrade pip-tools

# Generate pinned requirements.txt
pip-compile requirements.in --output-file requirements.txt

# Install the pinned environment
pip install -r requirements.txt
```

### 3. Initialize DuckDB Ledger (2 min)

```bash
python scripts/init_ledger.py
```

### 4. Run Automated Test Suite (5 min)

```bash
# Run tests inside virtualenv
.venv\Scripts\pytest  # Windows
.venv/bin/pytest      # Unix
```

### 5. Use the CLI

```bash
# Console entry point installed by setup.py
zindian-cli --help

# Equivalent source checkout form
python -m zindian.cli --help
```

### 6. Run Phase 1 Tests (5 min)

```bash
python scripts/test_phase_1.py
```

---

## Documentation

### Start Here
| Document | Purpose | Audience |
|----------|---------|----------|
| **[ORCHESTRATOR_OVERVIEW.md](docs/ORCHESTRATOR_OVERVIEW.md)** | **Complete guide (non-technical + technical)** | **Everyone** |
| [source_of_truth.md](docs/source_of_truth.md) | Official specification (v2.3) | Developers, architects |
| [cli_integration_guide.md](docs/cli_integration_guide.md) | CLI command reference | Users, operators |
| [QUICK_START.md](docs/QUICK_START.md) | Refactor onboarding and command flow | Contributors |

### Technical Documentation
| Document | Purpose | Audience |
|----------|---------|----------|
| [specs/requirements.md](specs/requirements.md) | Functional requirements | Architects, reviewers |
| [specs/design.md](specs/design.md) | Architecture & data flow | Developers, code reviewers |
| [specs/tasks.md](specs/tasks.md) | Phase checklist | Project managers |
| [docs/architecture_matrix.md](docs/architecture_matrix.md) | Stable control-flow map | Contributors |
| [docs/orchestrator_current_state.md](docs/orchestrator_current_state.md) | Current implementation snapshot | Maintainers |
| [sot_audit_report.md](docs/sot_audit_report.md) | Known gaps & issues | Developers, QA |
| [PROGRESS_TRACKER.md](docs/PROGRESS_TRACKER.md) | v2.3 refactor status | Project managers |
| [documentation_audit_report.md](docs/documentation_audit_report.md) | Doc freshness audit | Maintainers |

### Operational Documentation
| Document | Purpose | Audience |
|----------|---------|----------|
| [troubleshooting_guide.md](docs/troubleshooting_guide.md) | Common failure modes and fixes | Developers, operators |
| [workspace_rules.md](docs/workspace_rules.md) | Repository conventions and guardrails | Contributors |
| [session_log.md](docs/session_logs/session_log.md) | Historical implementation notes | Maintainers |
| [competition_data_lifecycle.md](docs/session_logs/competition_data_lifecycle.md) | Data lifecycle and repo workflow notes | Maintainers |

---

## Security & Compliance

### Data Integrity
- MD5 hash lock on all raw data files
- Atomic state updates (tempfile + os.replace)
- Config immutability after Phase 1
- No hardcoded competition-specific values (A5)

### Zindi Compliance
- No AutoML libraries (preflight scan)
- Fixed seed for reproducibility
- Raw probabilities preserved (classification)
- Submission budget guard
- Complete audit trail for code review

### Credentials
- `.env` file (not committed)
- Environment variable fallbacks
- No credentials in code or config

---

## Testing

### Automated Tests (pytest)
- Comprehensive pytest-based test framework
- Run using: `.venv\Scripts\pytest` (Windows) or `.venv/bin/pytest` (Unix)
- Unit tests for: `state.py`, `config.py`, `ledger.py`, `cv.py`, `paths.py`
- Skill verification tests for: anchor training, gating logic, SHAP audit, pseudo-labeling

---

## Key Features (v2.3)

### Human Gates (5 Checkpoints)
System stops and waits for approval at:
1. **Gate 1:** After anchor model - "Does baseline look reasonable?"
2. **Gate 2:** Before promoting variants - "Keep this model?" (per variant)
3. **Gate 3:** Before fusion - "Ready to blend models?"
4. **Gate 4:** Before inference - "Generate predictions?"
5. **Gate 5:** Before close - "Which 2 submissions?"

### Reproducibility Contract (R1-R5)
- **R1:** Seed always set (fixed at Phase 1)
- **R2:** Rerun = identical output (bit-identical)
- **R3:** No custom packages (all in requirements.txt)
- **R4:** Submission reproducible from config + state
- **R5:** Carbon tracking (CO2 estimation per skill)

### Carbon Tracking (R5 - New in v2.3)
Measures environmental impact:
```json
"telemetry.skill_08_anchor": {
  "duration_sec": 123.45,
  "peak_memory_mb": 2048,
  "carbon_kg_estimate": 0.0012,
  "tracker_method": "mlco2_formula",
  "hardware_type": "cpu",
  "region": "us-east-1"
}
```

### Multi-Target Support
Handles competitions with multiple prediction targets:
- Trains separate model per target
- Computes weighted composite score
- Normalizes by target standard deviation
- Single gate decision for all targets

### Pseudo-Labeling (Skill 21)
For classification: expands training set with confident predictions
- 6 guard conditions (all must pass)
- Retraining loop with augmented dataset
- Rollback if zero variants pass gate
- Classification-only (guard condition 1)

### Scale-Invariant Gating
Dynamic threshold normalization:
- **RMSE/MAE:** Scales by target_std
- **RMSLE:** No scaling (dimensionless)
- **Classification:** No scaling (bounded metrics)

---

## Development Status

### v2.3 Complete
- All skill modules implemented and verified
- 5 human gates operational
- Carbon tracking (R5) instrumented
- Multi-target pipeline functional
- Pseudo-labeling with rollback
- Scale-invariant gate normalization
- Comprehensive documentation suite under docs/

### Known Limitations
- **C1:** Bootstrap phase string mismatch (workaround documented)
- **GAP-3:** SHAP interaction features (deferred to v3.0)

See [sot_audit_report.md](docs/sot_audit_report.md) for details.

---

## Contributing

The framework is **specification-driven**. To add a new skill:

1. **Design** — Add to `specs/tasks.md`
2. **Implement** — Create `zindian/skills/skill_XX_*.py`
3. **Test** — Add to `tests/test_skill_XX.py`
4. **Document** — Update `docs/source_of_truth.md`

### Skill Template

```python
"""Skill XX — Description"""
from zindian.config import ChallengeConfig
from zindian.state import SkillStateStore
from pathlib import Path

def run(
    *,
    state_path: str = "SKILL_STATE.json",
    config_path: str = "challenge_config.json",
    **kwargs
):
    """
    Run Skill XX.
    
    Returns:
        Dict: {"status": "GO|ERROR", "result": ..., "message": "..."}
    """
    try:
        config = ChallengeConfig.load(config_path)
        state_store = SkillStateStore(Path(state_path))
        
        # YOUR LOGIC HERE
        
        state_store.update(dag_phase="phase_X_done")
        return {"status": "GO", "result": ...}
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}

if __name__ == "__main__":
    print(run())
```

---

## Known Issues

| Issue | Status | Reference |
|-------|--------|-----------|
| Bootstrap phase string mismatch | DOCUMENTED | [sot_audit_report.md](docs/sot_audit_report.md) C1 |
| SHAP interaction features | DEFERRED | v3.0 roadmap |

---

## Support

- **Getting Started** → Read [ORCHESTRATOR_OVERVIEW.md](docs/ORCHESTRATOR_OVERVIEW.md)
- **Architecture Questions** → Read [source_of_truth.md](docs/source_of_truth.md)
- **Implementation Details** → Check `specs/` directory
- **Bug Reports** → See [sot_audit_report.md](docs/sot_audit_report.md)

---

## License

Apache 2.0. See [LICENSE](LICENSE).

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| **2.3** | June 2026 | Carbon tracking (R5), multi-target support, pseudo-labeling, scale-invariant gating |
| 2.2.1 | May 2026 | Multi-target pipeline, regression support |
| 2.2 | April 2026 | Core skill modules, 5 human gates |
| 2.0 | March 2026 | Phase 0-5 complete |

---

**Last Updated:** June 2026  
**Status:** v2.3 Production Ready
