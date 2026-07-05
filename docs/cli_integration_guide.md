# Zindian CLI Integration Guide

**Purpose:** Complete reference for using and extending the Zindian CLI, consolidating all modules and utility scripts.

---

## Table of Contents
1. [Quick Start](#quick-start)
2. [Competition Context Resolution](#competition-context-resolution)
3. [Unified Console Commands (21 Commands)](#unified-console-commands-21-commands)
   - [Group A: Intake & Initialization](#group-a-intake--initialization)
   - [Group B: Phase Execution & Monitoring](#group-b-phase-execution--monitoring)
   - [Group C: Reproducibility & Validation](#group-c-reproducibility--validation)
   - [Group D: Submissions & Leaderboards](#group-d-submissions--leaderboards)
4. [Command Development Pattern](#command-development-pattern)
5. [Testing Commands](#testing-commands)
6. [Common CLI Patterns](#common-cli-patterns)

---

## Quick Start

### Setup Environment
```bash
cd /home/sagemaker-user/shared/zindian-orchestrator

# Option 1: Set competition via environment variable
export ZINDIAN_COMPETITION="world-cup-2026-goal-prediction-challenge"

# Option 2: Run from competition directory (auto-detects)
cd competitions/world-cup-2026-goal-prediction-challenge

# Execute CLI
zindian-cli <command>
```

If the console script is not available in a source checkout, `python -m zindian.cli <command>` is the equivalent fallback.

---

## Competition Context Resolution

The system resolves competition context in this order:
1. **Current Working Directory (CWD)**: Auto-detects if running inside `competitions/<slug>/`.
2. **Environment Variable**: `ZINDIAN_COMPETITION` or `COMPETITION_SLUG` or `ZINDIAN_COMPETITION_SLUG`.
3. **`.env` file**: Parses `.env` from repository root for `ZINDIAN_COMPETITION=<slug>` or `COMPETITION_SLUG=<slug>`.
4. **Auto-detect Fallback**: Fallback to resolving the active folder if only one competition folder with a state file is present.
5. **Fallback error** if none are found.

---

## Unified Console Commands (21 Commands)

The Zindian CLI exposes exactly 21 commands, grouped logically by their role in the Competition Data Lifecycle.

### Group A: Intake & Initialization

#### 1. `bootstrap` - Setup Competition Folder
```bash
python -m zindian.cli bootstrap <slug> [--move-files] [--yes]
```
*   **Description:** Creates the directory tree, writes templates for `challenge_config.json` and `SKILL_STATE.json` if missing, and optionally moves known root datasets.

#### 2. `init-ledger` - Initialize experiments database
```bash
python -m zindian.cli init-ledger
```
*   **Description:** Initializes the DuckDB experiments database (`reports/experiments.db`) with `experiments` and `submissions` tables.

#### 3. `preflight` - Run compliance checks
```bash
python -m zindian.cli preflight [--competition <path>]
```
*   **Description:** Validates project state, files, gate keys, and schemas under `INIT` or `ENFORCE` modes.

#### 4. `preflight-sim` - Run preflight simulations
```bash
python -m zindian.cli preflight-sim
```
*   **Description:** Wraps `scripts/run_preflight_sim.sh` to run preflight simulation checks on `tmpcomp` and `ey-frogs`.

#### 5. `sync` - Update State from Git + Zindi
```bash
python -m zindian.cli sync
```
*   **Description:** Synchronizes active Git branch, submissions history, leaderboard rank, and budget metrics from Zindi API.

---

### Group B: Phase Execution & Monitoring

#### 6. `phase` - Execute Pipeline Phase
```bash
python -m zindian.cli phase <1|2A|2B|3A|3B|4> [--verbose] [--variant <name>]
```
*   **Description:** Executes a complete pipeline phase with all corresponding skills. Shows the execution status of each skill.

#### 7. `status` - Show current state
```bash
python -m zindian.cli status
```
*   **Description:** Displays the active competition name, current DAG phase, remaining budget, public LB score, and git branch as structured JSON.

#### 8. `monitor` - Check Zindi updates
```bash
python -m zindian.cli monitor
```
*   **Description:** Scans the Zindi discussion board, data patches, and rule changes. Logs community signals to `SKILL_STATE["community_signals"]`.

#### 9. `report` - Generate summary report
```bash
python -m zindian.cli report
```
*   **Description:** Generates `reports/phase_<N>_summary.json` containing configuration, state, and ledger stats.

---

### Group C: Reproducibility & Validation

#### 10. `verify-state` - Verify competition state files
```bash
python -m zindian.cli verify-state
```
*   **Description:** Performs ground-truth checks of state, configuration, raw data shapes, features, and submission formatting.

#### 11. `verify-phase-b` - Verify Phase B package assertions
```bash
python -m zindian.cli verify-phase-b
```
*   **Description:** Validates package hardening assertions including `FrozenDict` usage, `SkillStateStore` operations, and API fallbacks.

#### 12. `write-oof-meta` - Write metadata JSON files
```bash
python -m zindian.cli write-oof-meta
```
*   **Description:** Scans processed data and reports for OOF CSVs, writing a sibling `.meta.json` file detailing row count, MD5 checksum, and CV strategy mapping.

#### 13. `compile-requirements` - Compile pinned requirements
```bash
python -m zindian.cli compile-requirements
```
*   **Description:** Wraps `scripts/compile_requirements.sh` to generate pinned `requirements.txt` from specifications `requirements.in`.

#### 14. `audit` - Run reproducibility audit
```bash
python -m zindian.cli audit [--slug <slug>]
```
*   **Description:** Run a full static audit for a competition directory checking for imports, config properties, and schemas.

#### 15. `audit-framework` - Audit codebase framework
```bash
python -m zindian.cli audit-framework
```
*   **Description:** Wraps `scripts/zindian_audit.sh` to run a comprehensive codebase validation (files, open code skills, python package, and venv environment).

#### 16. `check-deployment` - Check deployment storage
```bash
python -m zindian.cli check-deployment
```
*   **Description:** Wraps `scripts/check_skill_state_deployment.sh` to assert storage optimization and deployment structures.

---

### Group D: Submissions & Leaderboards

#### 17. `submit` - Upload to Zindi
```bash
python -m zindian.cli submit <path-to-csv>
```
*   **Description:** Validates submission formatting, checks budget availability, triggers human gate confirmation, uploads to Zindi, and records to ledger DB.

#### 18. `submissions` - Show submission board
```bash
python -m zindian.cli submissions
```
*   **Description:** Renders a formatted board of all Zindi submissions.

#### 19. `leaderboard` - Show rankings
```bash
python -m zindian.cli leaderboard [--per-page N]
```
*   **Description:** Pulls and displays the current leaderboard standings.

#### 20. `archive` - Archive completed competition
```bash
python -m zindian.cli archive <slug>
```
*   **Description:** Wraps `scripts/archive_competition.sh` to compress the competition folder while excluding raw dataset files to conserve space.

#### 21. `ledger` - Query experiments database
```bash
python -m zindian.cli ledger <experiments|submissions|best|passed|failed>
```
*   **Description:** Queries experiments and submissions DuckDB database. Supports subcommands:
    - `experiments`: Show all logged experiments.
    - `submissions`: Show all logged submissions.
    - `best`: Show the best experiment.
    - `passed`: Show passed experiments.
    - `failed`: Show failed experiments.

---

## Command Development Pattern

### Minimal Command Template
```python
# 1. Add subparser in main() inside zindian/cli.py
my_parser = subparsers.add_parser("mycommand", help="Description")
my_parser.add_argument("--option", help="Optional argument")

# 2. Add handler in main()
elif args.command == "mycommand":
    from zindian.module import function
    
    result = function(option=args.option if hasattr(args, "option") else None)
    
    # Output as JSON or formatted text
    if isinstance(result, dict):
        print(json.dumps(result, indent=2, default=str))
    else:
        print(result)
    
    sys.exit(0 if result.get("success") else 1)
```

### Command Checklist
- [ ] Add subparser with clear help text.
- [ ] Add all required and optional arguments.
- [ ] Import module/skill lazily inside the handler to prevent slow CLI boots.
- [ ] Output structured data (JSON preferred).
- [ ] Return appropriate exit code (0=success, 1=error).

---

## Testing Commands

### Unit & Policy Tests
```bash
# Test CLI argument parsing
pytest tests/test_cli_edge_cases.py -v

# Test phase execution
pytest tests/test_orchestrator_refactor.py -v

# Test submission flow
pytest tests/test_submission_board_leaderboard_integration.py -v
```

### Integration Test Pattern
```python
import subprocess
import json

def test_phase_1_execution():
    """Test Phase 1 execution via CLI."""
    result = subprocess.run(
        ["python", "-m", "zindian.cli", "phase", "1"],
        cwd="/path/to/competitions/test-comp",
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0
    assert "PHASE 1 RESULTS" in result.stdout
    assert "skill_01: GO" in result.stdout

def test_status_command():
    """Test status command returns valid JSON."""
    result = subprocess.run(
        ["python", "-m", "zindian.cli", "status"],
        cwd="/path/to/competitions/test-comp",
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert "competition" in data
    assert "dag_phase" in data
```

---

## Common CLI Patterns

### Read-Only Query
```python
elif args.command == "query":
    from zindian.state import SkillStateStore
    from zindian.paths import resolve_competition_paths
    
    paths = resolve_competition_paths()
    state = SkillStateStore(paths.state_path).read()
    print(json.dumps(state.get("key"), indent=2))
```

### State Mutation
```python
elif args.command == "update":
    from zindian.state import SkillStateStore
    from zindian.paths import resolve_competition_paths
    
    paths = resolve_competition_paths()
    store = SkillStateStore(paths.state_path)
    store.update(key="value")
    print("✅ State updated")
```
