# Zindian CLI Integration Guide

**Purpose:** Complete reference for using and extending the Zindian CLI, including phase execution patterns.

---

## Table of Contents
1. [Quick Start](#quick-start)
2. [Existing Commands](#existing-commands)
3. [Adding Phase Execution](#adding-phase-execution)
4. [Command Development Pattern](#command-development-pattern)
5. [Competition Context Resolution](#competition-context-resolution)
6. [Testing Commands](#testing-commands)

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
python -m zindian.cli <command>
```

### Competition Context Priority
The system resolves competition context in this order:
1. **Current working directory** (if inside `competitions/<slug>/`)
2. **Environment variable** `ZINDIAN_COMPETITION`
3. **`.env` file** `ZINDIAN_COMPETITION=<slug>`
4. **Fallback error** if none found

---

## Existing Commands

### 1. Status - Current Competition State
```bash
python -m zindian.cli status
```
**Returns:**
```json
{
  "competition": "world-cup-2026-goal-prediction-challenge",
  "dag_phase": "phase_1_complete",
  "submissions_used_today": 0,
  "remaining_submissions": 10,
  "anchor_oof_score": null,
  "anchor_lb_score": null,
  "current_git_branch": "main"
}
```

### 2. Sync - Update State from Git + Zindi
```bash
python -m zindian.cli sync
```
**Updates:** Git branch, submissions, leaderboard rank, remaining budget

### 3. Submissions - View History
```bash
python -m zindian.cli submissions
```
**Output:** Formatted table of all submissions

### 4. Leaderboard - Competition Rankings
```bash
python -m zindian.cli leaderboard [--per-page 20]
```

### 5. Submit - Upload to Zindi
```bash
python -m zindian.cli submit submissions/my_submission.csv
```
**Workflow:** Validate → Budget check → Human gate → Submit → Record

### 6. Ledger - Experiments Database
```bash
# All experiments
python -m zindian.cli ledger experiments

# All submissions
python -m zindian.cli ledger submissions

# Best experiment (by metric)
python -m zindian.cli ledger best

# Filter by gate result
python -m zindian.cli ledger passed
python -m zindian.cli ledger failed
```

### 7. Monitor - Competition Updates
```bash
python -m zindian.cli monitor
```
**Checks:** Discussion board, data patches, rule changes

### 8. Report - Phase Summary
```bash
python -m zindian.cli report
```
**Creates:** `reports/phase_<N>_summary.json`

### 10. Phase - Execute Pipeline Phase
```bash
python -m zindian.cli phase <1|2A|2B|3A|3B|4> [--verbose]
```
**Executes:** Complete phase with all skills, returns status for each

---

## Phase Execution

### Execute Pipeline Phases
```bash
python -m zindian.cli phase <1|2A|2B|3A|3B|4> [--verbose]
```

**Usage Examples:**

```bash
# Execute Phase 1 (Intake & EDA)
python -m zindian.cli phase 1

# Execute Phase 2A with verbose output
python -m zindian.cli phase 2A --verbose

# Execute Phase 2B
python -m zindian.cli phase 2B

# Check status after phase
python -m zindian.cli status
```

**Competition Override:**

```bash
# Execute phase for specific competition
ZINDIAN_COMPETITION="world-cup-2026-goal-prediction-challenge" \
  python -m zindian.cli phase 1

# Or from competition directory
cd competitions/world-cup-2026-goal-prediction-challenge
python -m zindian.cli phase 1
```

---

## Command Development Pattern

### Minimal Command Template

```python
# 1. Add subparser in main()
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
    
    # Exit with appropriate code
    sys.exit(0 if result.get("success") else 1)
```

### Command Checklist
- [ ] Add subparser with clear help text
- [ ] Add all required/optional arguments
- [ ] Import skill/module in handler (not at top)
- [ ] Handle missing arguments gracefully
- [ ] Output structured data (JSON preferred)
- [ ] Return appropriate exit code (0=success, 1=error)
- [ ] Update this guide with usage examples

---

## Competition Context Resolution

### How Paths Are Resolved

The `resolve_competition_paths()` function checks:

```python
from zindian.paths import resolve_competition_paths

# Auto-detects competition from:
# 1. Current directory (if in competitions/<slug>/)
# 2. ZINDIAN_COMPETITION env var
# 3. .env file
paths = resolve_competition_paths()

# Access paths
paths.competition_dir  # /path/to/competitions/<slug>
paths.data_dir         # /path/to/competitions/<slug>/data
paths.config_path      # /path/to/competitions/<slug>/challenge_config.json
paths.state_path       # /path/to/competitions/<slug>/SKILL_STATE.json
```

### Setting Competition Context

**Method 1: Environment Variable (Temporary)**
```bash
export ZINDIAN_COMPETITION="world-cup-2026-goal-prediction-challenge"
python -m zindian.cli phase 1
```

**Method 2: .env File (Persistent)**
```bash
# Edit .env file
echo "ZINDIAN_COMPETITION=world-cup-2026-goal-prediction-challenge" >> .env

# All commands now use this competition
python -m zindian.cli status
python -m zindian.cli phase 1
```

**Method 3: Working Directory (Auto-detect)**
```bash
cd competitions/world-cup-2026-goal-prediction-challenge
python -m zindian.cli phase 1  # Auto-detects from pwd
```

### Multi-Competition Workflow

```bash
# Work on Competition A
cd competitions/competition-a
python -m zindian.cli phase 1
python -m zindian.cli status

# Switch to Competition B
cd ../competition-b
python -m zindian.cli phase 1
python -m zindian.cli status

# Or use explicit override
ZINDIAN_COMPETITION="competition-c" python -m zindian.cli status
```

---

## Testing Commands

### Unit Tests
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
# tests/test_cli_phase_execution.py
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

### Manual Testing Checklist

```bash
# 1. Test phase execution
python -m zindian.cli phase 1
python -m zindian.cli phase 2A

# 2. Test status after phase
python -m zindian.cli status

# 3. Test error handling
python -m zindian.cli phase 2B  # Should fail if 2A not complete

# 4. Test verbose mode
python -m zindian.cli phase 1 --verbose

# 5. Test competition context
cd competitions/world-cup-2026-goal-prediction-challenge
python -m zindian.cli status  # Should show correct competition

# 6. Test environment override
ZINDIAN_COMPETITION="other-comp" python -m zindian.cli status
```

---

## Common Patterns

### Pattern 1: Read-Only Query
```python
elif args.command == "query":
    from zindian.state import SkillStateStore
    from zindian.paths import resolve_competition_paths
    
    paths = resolve_competition_paths()
    state = SkillStateStore(paths.state_path).read()
    
    print(json.dumps(state.get("key"), indent=2))
```

### Pattern 2: State Mutation
```python
elif args.command == "update":
    from zindian.state import SkillStateStore
    from zindian.paths import resolve_competition_paths
    
    paths = resolve_competition_paths()
    store = SkillStateStore(paths.state_path)
    store.update(key="value")
    
    print("✅ State updated")
```

### Pattern 3: Skill Execution
```python
elif args.command == "run_skill":
    from zindian.orchestrator import run_skill
    
    result = run_skill(args.skill_name)
    
    if result.get("status") == "GO":
        print(f"✅ {args.skill_name} completed")
    else:
        print(f"❌ {args.skill_name} failed: {result.get('message')}")
        sys.exit(1)
```

### Pattern 4: Human Gate
```python
elif args.command == "gate":
    from zindian.skills.skill_11_gate import run
    
    result = run()
    
    if result.get("gate_result") == "PASS":
        print("✅ Gate passed - promoting to production")
    else:
        print(f"❌ Gate failed: {result.get('reason')}")
        sys.exit(1)
```

---

## Quick Reference Card

```bash
# Phase Execution (NEW)
python -m zindian.cli phase <1|2A|2B|3A|3B|4> [--verbose]

# State Management
python -m zindian.cli status              # View current state
python -m zindian.cli sync                # Update from git/Zindi

# Submissions
python -m zindian.cli submit <file>       # Submit to Zindi
python -m zindian.cli submissions         # View history
python -m zindian.cli leaderboard         # View rankings

# Experiments
python -m zindian.cli ledger experiments  # All experiments
python -m zindian.cli ledger best         # Best by metric
python -m zindian.cli ledger passed       # Passed gate

# Monitoring
python -m zindian.cli monitor             # Check updates
python -m zindian.cli report              # Generate summary
python -m zindian.cli audit               # Reproducibility check

# Competition Context
export ZINDIAN_COMPETITION="<slug>"       # Set competition
cd competitions/<slug>                    # Auto-detect from pwd
```

---

## Next Steps

1. **Use phase command** for automated execution
2. **Test with World Cup 2026** validation workflow
3. **Add to CI/CD** for automated phase execution
4. **Create shell aliases** for common workflows:
   ```bash
   alias zp1='python -m zindian.cli phase 1'
   alias zp2a='python -m zindian.cli phase 2A'
   alias zstatus='python -m zindian.cli status'
   ```

---

## Troubleshooting

### "No competition found"
- Check `ZINDIAN_COMPETITION` env var
- Verify `.env` file has correct slug
- Ensure running from competition directory

### "Phase X blocked: Phase Y must complete first"
- Check `python -m zindian.cli status` for current phase
- Verify `SKILL_STATE.json` has `phase_<X>_complete: true`
- Run prerequisite phases first

### "Skill failed: No module named 'zindi'"
- Zindi client is intentionally disabled (network isolation)
- Use `--dry-run` mode or mock for testing
- Check `zindi_stub_backup_DISABLED/` for reference

---

**Last Updated:** 2026-06-17  
**Maintainer:** Zindian Core Team  
**Related Docs:** 
- [CLI Quick Reference](session_logs/cli_quick_reference.md)
- [Orchestrator Architecture](orchestrator_current_state.md)
- [Source of Truth](source_of_truth.md)
