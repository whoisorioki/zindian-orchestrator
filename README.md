# Zindian Orchestrator

An **autonomous multi-phase ML competition framework** for tabular supervised learning competitions on the Zindi Africa platform. Reads competition schemas, target definitions, and evaluation rules dynamically from a locked configuration contract — zero hardcoded competition literals.

Zindian Orchestrator is a deterministic, phase-gated pipeline that converts raw tabular data and a `challenge_config.json` contract into submission-ready predictions. The system is governed by the **Source of Truth** (`docs/source_of_truth.md` v2.5). When code, documentation, and `AGENTS.md` disagree, resolution order is: runtime behavior > SoT > AGENTS.md.

**System properties:**
- **Phase-gated execution** — 4 phases with 5 human gates; state transitions recorded atomically in `SKILL_STATE.json`
- **Deterministic CV factory** — single `make_cv_splitter()` entry point; no skill instantiates its own splitter
- **OOF contract enforcement** — every out-of-fold generator calls `write_oof_record()` with `cv_strategy_id` tagging; augmented runs write to `_augmented`-suffixed keys only
- **Two-mode feature contract** — target-dependent features enforce fold restriction in CV mode; structural features computed on full data
- **Zero competition literals in skill bodies** — column names, targets, metrics, coordinates resolved from config at runtime (A5)
- **No AutoML** — static preflight scan rejects `auto-sklearn`, `flaml`, `tpot`, `h2o`, `pycaret`, `optuna.integration`

---

## Architecture Overview

### Core Principles

1. Three-Lens Decision Philosophy
2. Competition Agnosticism
3. Atomic State Management
4. Immutable Config
5. Human-in-the-Loop
6. Reproducibility Contract

### The 4 Main Phases

```
Phase 1 (Fingerprint) → Phase 2 (Anchor + Variants) → Phase 3 (Audit + Promotion) → Phase 4 (Governance)
Five human gates: Gate 1 (anchor), Gate 2 (per variant), Gate 3 (fusion), Gate 4 (inference), Gate 5 (final select)
```

See [docs/source_of_truth.md](docs/source_of_truth.md) v2.5 for the full v2.4 feature specifications (S1–S10, R1–R6).

---

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
│           ├── audits/          ← policy, legality, SHAP, governance, reproducibility
│           ├── diagnostics/     ← EDA, hypotheses (+ predictions/ for pseudo-label CSVs)
│           ├── summaries/       ← phase summary Markdown/JSON
│           └── sessions/        ← session-scoped startup event logs
├── docs/                             ← Standardized documentation
│   ├── source_of_truth.md            ← Authoritative specification v2.5
│   ├── orchestrator_overview.md      ← System architecture & design overview
│   ├── quick_start.md                ← Local run onboarding & execution flow
│   ├── cli_integration_guide.md      ← CLI command usage reference
│   ├── ledger_architecture.md        ← DuckDB audit ledger schema definition
│   ├── troubleshooting_guide.md      ← Common errors, recoveries & guardrails
│   └── document_map.md               ← Document structure, overlaps & ownership matrix
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
Start with [docs/orchestrator_overview.md](docs/orchestrator_overview.md) for a complete system walkthrough.

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
```bash
python -m zindian.cli init-ledger
```

### 4. Run Automated Test Suite
```bash
python -m pytest
```

### 5. Use the CLI
```bash
# Run command bootstrapper CLI
python -m zindian.cli --help

# Alternative console entrypoint (if setup.py was installed)
zindian-cli --help
```

### 6. Run Phase 1 Simulation Demo
```bash
python scripts/test_phase_1.py
```

---

## Documentation

| Document | Purpose |
|----------|---------|
| **[docs/orchestrator_overview.md](docs/orchestrator_overview.md)** | Complete system guide (non-technical + technical) |
| [docs/source_of_truth.md](docs/source_of_truth.md) | Authoritative architectural spec (v2.5) |
| [docs/quick_start.md](docs/quick_start.md) | Local run setup walkthrough |
| [docs/cli_integration_guide.md](docs/cli_integration_guide.md) | All CLI commands reference |
| [docs/ledger_architecture.md](docs/ledger_architecture.md) | Experiment ledger schema |
| [docs/troubleshooting_guide.md](docs/troubleshooting_guide.md) | Common errors and resolutions |
| [docs/document_map.md](docs/document_map.md) | Consolidated structure maps, cross-document overlaps, and ownership matrix |

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
| **2.5** | August 2026 | Lean documentation restructure; no architecture changes from v2.4 |
| **2.4** | August 2026 | Nadeau-Bengio corrected variance, 1-SE promotion margins, Kuncheva residual diversity, MI audits, spatial buffer CV, adaptive pseudo-labeling, 3-tier FP tolerance |
| 2.3 | June 2026 | Carbon tracking (R5), multi-target support, pseudo-labeling, scale-invariant gating |
| 2.2.1 | May 2026 | Multi-target pipeline, regression support |
| 2.2 | April 2026 | Core skill modules, 5 human gates |
| 2.0 | March 2026 | Phase 0-5 complete |

---

**Last Updated:** August 2026
**Status:** v2.5 Production Ready
