# Zindian Orchestrator

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

## Project Structure


```
zindian-orchestrator/
├── competitions/                     ← Per-competition workspace folders
│   └── <slug>/
│       ├── challenge_config.json     ← Competition config contract
│       ├── SKILL_STATE.json          ← Execution state (memory)
│       ├── data/
│       ├── notebooks/
│       └── reports/
├── docs/                             ← Standardized documentation
│   ├── source_of_truth.md            ← Authoritative specification v2.3
│   ├── orchestrator_overview.md      ← System architecture & design overview
│   ├── quick_start.md                ← Local run onboarding & execution flow
│   ├── cli_integration_guide.md      ← CLI command usage reference
│   ├── ledger_architecture.md         ← DuckDB audit ledger schema definition
│   └── troubleshooting_guide.md      ← Common errors, recoveries & guardrails
├── zindian/                          ← Core Python package
│   ├── state.py                      ← Atomic state I/O operations
│   ├── config.py                     ← Safe challenge_config reader
│   ├── ledger.py                     ← DuckDB ledger wrapper
│   ├── cv.py                         ← CV strategy factory
│   ├── orchestrator.py               ← Phase & skill manager
│   └── skills/                       ← Implemented skill modules (skills 00-22)
├── tabula/                           ← CLI bootstrapping tool
├── scripts/                          ← Utility scripts
└── tests/                            ← Automated test suite (160+ tests)
```

---

## Quick Start

### 1. Read the Architecture Docs
Open the canonical documentation files under the `docs/` directory. We recommend reading them in this order:
1. [docs/orchestrator_overview.md](docs/orchestrator_overview.md) — Architectural philosophy and phase flow.
2. [docs/source_of_truth.md](docs/source_of_truth.md) — authoritative specifications and rules.
3. [docs/cli_integration_guide.md](docs/cli_integration_guide.md) — command references.

### 2. Install & Verify
First, activate your virtual environment:

* **Unix/macOS:**
  ```bash
  source .venv/bin/activate
  ```
* **Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```

Then install the pinned dependencies:
```bash
python -m pip install -r requirements.txt
```

#### Managing Dependencies (Optional)
To add or update top-level packages:
```bash
# Install package compiler
python -m pip install --upgrade pip-tools

# Recompile requirements.txt from requirements.in
pip-compile requirements.in --output-file requirements.txt

# Install compiled environment
python -m pip install -r requirements.txt
```

### 3. Initialize DuckDB Ledger
Set up the SQLite-compatible DuckDB experiments ledger:
```bash
python -m zindian.cli init-ledger

```

### 4. Run Automated Test Suite
Verify your environment by running the test suite:
```bash
python -m pytest
```

### 5. Use the CLI
Interact with the orchestrator using Python's module syntax:
```bash
# Run command bootstrapper CLI
python -m zindian.cli --help

# Alternative console entrypoint (if setup.py was installed)
zindian-cli --help
```

### 6. Run Phase 1 Simulation Demo
Run the mock simulation check to verify your setup:
```bash
python scripts/test_phase_1.py
```



---

## Documentation

### Core Guides
| Document | Purpose | Audience |
|----------|---------|----------|
| **[docs/orchestrator_overview.md](docs/orchestrator_overview.md)** | **Complete system guide (non-technical + technical)** | **Everyone** |
| [docs/source_of_truth.md](docs/source_of_truth.md) | Official architectural spec (v2.3) | Developers, reviewers |
| [docs/quick_start.md](docs/quick_start.md) | Guide for setting up local runs | Developers, users |
| [docs/cli_integration_guide.md](docs/cli_integration_guide.md) | CLI command syntax reference | Operators, users |
| [docs/ledger_architecture.md](docs/ledger_architecture.md) | Experiment ledger schema specifications | DAAD Reviewers, DBAs |
| [docs/troubleshooting_guide.md](docs/troubleshooting_guide.md) | Common errors and resolutions | Developers, operators |

---

## Security & Compliance

### Data Integrity
- MD5 hash locking on raw input files (calculated at intake).
- Atomic state updates (written via tempfile + os.replace).
- Immutability checks on challenge configuration after Phase 1.
- Zero hardcoded competition-specific strings (columns, targets, metrics).

### Zindi Compliance
- AutoML library scanner checks (fails preflight if unauthorized imports found).
- Fixed random seeds for reproducible folds and predictions.
- Raw class probability values preserved for calibration checks.
- Daily submission budget limit safety guards.

---

## Testing

### Automated Tests (pytest)
- Comprehensive test coverage of core modules: `state.py`, `config.py`, `ledger.py`, `cv.py`.
- Run tests using standard python syntax:
  ```bash
  python -m pytest
  ```
- Specific tests verify: anchor baseline training, threshold calibration, SHAP ratio leakage checks, and pseudo-labeling retraining.

---

## Key Features (v2.3)

### Human Gates (5 Checkpoints)
The orchestrator pauses execution and requests human Operator validation at:
1. **Gate 1:** After anchor model training completes.
2. **Gate 2:** Before promoting feature variants to state (evaluated per-branch).
3. **Gate 3:** Before triggering model fusion (blending).
4. **Gate 5:** Before close, to select the final 2 submissions.

### Reproducibility Contract (R1-R5)
- **R1:** Config-pinned reproducibility seed.
- **R2:** End-to-end runs yield bit-identical predictions.
- **R3:** Pinning of runtime dependencies in `requirements.txt`.
- **R4:** Submissions are fully regenerable from the competition folder config + state.
- **R5:** Carbon telemetry (real-time CPU/GPU memory & CO2 estimates computed per-skill).

---

## Development Status

### v2.3 Complete
- All skill modules implemented and verified.
- 5 human gates operational.
- Carbon tracking (R5) telemetry instrumented.
- Multi-target composite scoring pipeline functional.
- Pseudo-labeling with rollback.
- Scale-invariant gate normalization.

---

## Contributing

The framework is specification-driven. To add or update a skill:

1. **Design** — Outline the behavior in the master spec.
2. **Implement** — Create the module under `zindian/skills/skill_XX_*.py`.
3. **Test** — Add assertions to `tests/test_skill_XX.py`.
4. **Document** — Update [docs/source_of_truth.md](docs/source_of_truth.md).

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
```

---

## Support

- **Getting Started** → Read [docs/orchestrator_overview.md](docs/orchestrator_overview.md)
- **Architecture & Specifications** → Read [docs/source_of_truth.md](docs/source_of_truth.md)
- **Troubleshooting** → Check [docs/troubleshooting_guide.md](docs/troubleshooting_guide.md)
- **Licenses & Legal** → Check [LICENSE](LICENSE)

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
