# Zindian Orchestrator — Quick Start Guide

This guide walks you through setting up Zindian Orchestrator and running your first competition pipeline.

---

## 1. Environment Setup

Activate your virtual environment and install the dependencies:

*   **Unix/macOS:**
    ```bash
    source .venv/bin/activate
    ```
*   **Windows (PowerShell):**
    ```powershell
    .venv\Scripts\Activate.ps1
    ```

Install the pinned dependencies:
```bash
python -m pip install -r requirements.txt
```

Verify that the environment works by running the automated test suite:
```bash
python -m pytest
```

Alternatively, run the mock Phase 1 simulation script to verify skill module imports and executions on dummy data:
```bash
python scripts/test_phase_1.py
```

---

## 2. Initialize the Experiments Database

Initialize the DuckDB audit ledger, which records all training experiments and Zindi submissions:
```bash
python -m zindian.cli init-ledger
```
This creates the DuckDB ledger file located at `reports/experiments.db`.

---

## 3. Bootstrap a New Competition

Use the CLI tool to bootstrap a new competition workspace. This creates the required folders and generates default configurations:
```bash
python -m zindian.cli bootstrap my-tabular-challenge
```
This creates a new folder under `competitions/my-tabular-challenge/` with:
- `challenge_config.json` (competition contract template)
- `SKILL_STATE.json` (agent memory template)
- Empty directories for data, notebooks, and reports.

---

## 4. Run the Pipeline Phases

The orchestrator executes standard data science workflows in 5 sequential phases:

### Phase 1: Competition Fingerprint
Reads competition rules, examines raw data, and locks the configuration:
```bash
python -m zindian.cli phase 1
```

### Phase 2A: Data Cleaning
Cleans raw tables and prepares base features:
```bash
python -m zindian.cli phase 2A
```

### Phase 2B: Baseline Anchor Model
Trains the initial baseline model and triggers **Human Gate 1** (to approve the baseline score):
```bash
python -m zindian.cli phase 2B
```

### Phase 3A: Generalization Audit
Runs SHAP feature-leak checks, probability calibration, and variant gating:
```bash
python -m zindian.cli phase 3A
```

### Phase 3B: Model Fusion
Applies Oracle Fusion to blend predictions and triggers **Human Gate 3**:
```bash
python -m zindian.cli phase 3B
```

### Phase 4: Governance & Inference
Formats test predictions to the submission schema, runs reproducibility checks, and selects the final submissions:
```bash
python -m zindian.cli phase 4
```

---

## 5. Query the Experiments Ledger

To review all training runs, variant promotions, and scores logged in DuckDB, use the ledger query command:
```bash
# Show all experiments logged
python -m zindian.cli ledger experiments

# Show the best performing experiment
python -m zindian.cli ledger best

# Show passed model variants
python -m zindian.cli ledger passed
```
**Last Updated:** July 2026
