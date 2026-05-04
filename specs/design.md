# Zindian Orchestrator — Design

## Architecture Overview

The Zindian Orchestrator is a **pure Python system** that can be invoked by any IDE or terminal. State flows through two JSON files acting as the agent's memory. The IDE is just the window — the agent lives in the files.

```
User or IDE (any tool)
         ↓
    invokes Python package: zindian_orchestrator/
         ↓
    reads: SKILL_STATE.json, challenge_config.json
         ↓
    executes skill (Skill 01–17 based on dag_phase)
         ↓
    writes: SKILL_STATE.json, experiments.db (DuckDB ledger)
         ↓
    returns status: {status: "GO|PRUNE|ERROR", ...}
         ↓
    IDE displays result to user
```

---

## Core Modules

### `zindian_orchestrator/state.py`

**Purpose**: Atomic reader/writer for `SKILL_STATE.json`.

**Responsibility**:
- Read entire SKILL_STATE into memory
- Update specific fields (e.g., `md5_target_hash`, `dag_phase`, `submissions_used_today`)
- Write back atomically (write-then-rename to prevent corruption)

**API**:
```python
state = State()  # reads SKILL_STATE.json

# Read
phase = state.get("dag_phase")
hash = state.get("md5_target_hash")

# Update
state.set("dag_phase", "phase_1_integrity")
state.set("md5_target_hash", "abc123def456...")
state.write()  # atomic write back to disk

# Increment submissions
state.increment("submissions_used_today", 1)
state.write()
```

**Key guard**: Must raise `StateFileNotFound` if SKILL_STATE.json doesn't exist.

---

### `zindian_orchestrator/config.py`

**Purpose**: Read `challenge_config.json` with validation and null guarding.

**Responsibility**:
- Load competition configuration
- Raise `ConfigNotPopulated` if required fields are null
- Provide type-safe access via `.get(key)` method

**API**:
```python
config = Config(path="challenge_config.json")

# Read with null guard
metric = config.get("metric")  # raises ConfigNotPopulated if null
domain = config.get("domain", default="generic")  # safe default

# Check competition-specific rules
if config.get("use_probabilities"):
    predictions = model.predict_proba(X)
else:
    predictions = model.predict(X)

if config.get("domain") == "solar":
    apply_nighttime_constraint(predictions)
```

**Required fields**:
- `metric`
- `metric_direction` (minimize or maximize)
- `automl_permitted`
- `use_probabilities`
- `daily_limit`

**Optional fields** (may have defaults):
- `domain`
- `allowed_external_data`

---

### `zindian_orchestrator/ledger.py`

**Purpose**: DuckDB wrapper for tracking experiments and submissions.

**Responsibility**:
- Maintain `reports/experiments.db` with two tables: `experiments` and `submissions`
- Log every branch result automatically (OOF RMSE, feature count, calibration method)
- Log every Zindi submission (branch name, score, my_rank, timestamp)

**Schema**:

**Table: experiments**
| Column | Type | Notes |
|--------|------|-------|
| `experiment_id` | INT (PK) | Auto-increment |
| `branch_name` | VARCHAR | e.g., "feature_v2" |
| `oof_rmse` | FLOAT | Out-of-fold RMSE |
| `feature_count` | INT | Number of features used |
| `calibration_method` | VARCHAR | e.g., "isotonic", "none" |
| `gate_result` | VARCHAR | "PASS" or "FAIL" |
| `gate_reason` | VARCHAR | e.g., "OOF RMSE beats anchor by 0.8%" or "OOF RMSE too high" |
| `timestamp` | TIMESTAMP | When experiment was logged |

**Table: submissions**
| Column | Type | Notes |
|--------|------|-------|
| `submission_id` | INT (PK) | Auto-increment |
| `experiment_id` | INT (FK) | Link to experiments table |
| `branch_name` | VARCHAR | Branch that was submitted |
| `submission_rank` | INT | Submission number on Zindi (e.g., 1, 2, 3) |
| `public_score` | FLOAT | Public leaderboard score |
| `private_score` | FLOAT | Private score (if available) |
| `my_rank` | INT | User's rank on leaderboard after submit |
| `selected_for_final` | BOOL | True if selected for final 2 submissions |
| `final_selection_rationale` | VARCHAR | Why this submission was selected |
| `timestamp` | TIMESTAMP | When submission was made |

**API**:
```python
ledger = Ledger(path="reports/experiments.db")

# Log experiment
ledger.log_experiment(
    branch_name="feature_v2",
    oof_rmse=0.248,
    feature_count=42,
    calibration_method="isotonic",
    gate_result="PASS",
    gate_reason="OOF RMSE beats anchor by 0.8%"
)

# Log submission
ledger.log_submission(
    experiment_id=5,
    branch_name="feature_v2",
    submission_rank=1,
    public_score=0.251,
    my_rank=47
)

# Query
experiments = ledger.query("SELECT * FROM experiments WHERE gate_result = 'PASS'")
best_experiment = ledger.query("SELECT * FROM experiments ORDER BY oof_rmse LIMIT 1")
```

---

### `zindian_orchestrator/zindi_client.py`

**Purpose**: Thin wrapper around Zindi CLI with budget guard.

**Responsibility**:
- Wrap Zindi API calls (submit, get_rank, list_competitions)
- **Enforce budget guard**: check `remaining_submissions` before every submit
- Structure submission comments automatically from run metadata
- Poll leaderboard after submit to retrieve `my_rank`

**API**:
```python
client = ZindiClient()

# Before submit, check budget
if not client.check_remaining_submissions():
    raise BudgetExhausted("No submissions remaining today")

# Submit with structured comment
result = client.submit(
    submission_file="submissions/sub_001_anchor.csv",
    branch_name="anchor",
    oof_rmse=0.252,
    feature_count=8,
    calibration_method="none"
)
# Automatically writes comment as: "branch:anchor|oof_rmse:0.252|features:8|calib:none"

# Result includes my_rank
print(result)  # {"submission_id": 12345, "my_rank": 48, "public_score": 0.251}
```

**Guard implementation**:
```python
def check_remaining_submissions(self):
    """Raises BudgetExhausted if budget is zero"""
    remaining = self.get_remaining_submissions()
    if remaining <= 0:
        raise BudgetExhausted(f"Remaining submissions: {remaining}")
    return True
```

---

## Data Flow Per Session

### Session Initialization
1. Agent reads `SKILL_STATE.json` → knows current `dag_phase`
2. Agent reads `challenge_config.json` → knows competition rules
3. Agent checks: is this a **new competition** or **continuing**?
   - If new: dag_phase = "phase_0_foundation"
   - If continuing: dag_phase = current phase

### Phase Execution
4. Agent determines **next skill** based on dag_phase and checklist
5. Skill executes:
   - Reads from `config` and `state`
   - Does work (train, EDA, calibration, etc.)
   - Writes results to `SKILL_STATE.json` and DuckDB `ledger`
   - Returns status dict: `{status: "GO" | "PRUNE" | "ERROR", message: "...", ...}`
6. Agent updates `SKILL_STATE.json`:
   ```json
   {
     "dag_phase": "phase_1_integrity",
     "md5_target_hash": "abc123def456...",
     "last_updated": "2026-05-04T10:30:00+03:00"
   }
   ```

### Submission Flow (Phases 2–5)
7. Branch is trained and OOF RMSE calculated
8. Gate check: does OOF RMSE beat `anchor_oof_rmse` by 0.5%?
   - **YES** → proceed to submit
   - **NO** → log to ledger with `gate_result="FAIL"`, prune branch, next
9. Before submit, check `remaining_submissions`
   - **YES** → call `zindi_client.submit()`
   - **NO** → raise `BudgetExhausted`, halt
10. Submit populates DuckDB `submissions` table
11. Zindi API polls `my_rank` and logs to `submissions.my_rank`

### Session End
12. Agent writes final `SKILL_STATE.json`
13. User reviews DuckDB `submissions` table for selections and writes rationale
14. At Phase 5 end: select exactly 2 submissions, log rationale to `reports/`

---

## Phase Map & Skill Allocation

| Phase | Skills | Deliverable | Phase Gate |
|-------|--------|-------------|-----------|
| **0 — Foundation** | Init, auth, ledger | SKILL_STATE.json + challenge_config.json ready | Manual: "ledger initialized" |
| **1 — Integrity + Intake** | 01 (MD5 lock), 02 (rules intake), 15 (reporter) | MD5 hash locked, config verified | Automatic: MD5 present + config populated |
| **2 — Anchor Baseline** | 03 (EDA), 08 (anchor LightGBM) | sub_001_anchor.csv submitted | Automatic: submission successful + public score available |
| **3 — Features + Calibration** | 04 (features), 09 (calibration), 10 (SHAP) | SHAP top-20 + calibrated predictions | Automatic: SHAP analysis written + calibration coefficients persisted |
| **4 — Branch + Gate** | 05 (branches), 11 (gate), 16 (critique) | Multiple branches tested, gate applied | Automatic: ≥2 branches pass gate, rationale logged |
| **5 — Fusion + Final Submit** | 13 (fusion), 14 (inference guard), 17 (submission governance) | Exactly 2 submissions selected for final | Manual: rationale for 2 selections documented |

---

## Entry Points

### Terminal Invocation
```bash
cd /path/to/zindian_orchestrator
python -m zindian_orchestrator.cli run-next-skill
```

### Cursor/VS Code/Any IDE
1. Opens project folder
2. Reads `.cursor/rules/zindian.md` or `.github/instructions/zindian.md` (or AGENTS.md)
3. Agent pastes session-start prompt to user
4. User responds with current SKILL_STATE
5. IDE agent invokes Python via `run_in_terminal` or `mcp_pylance_mcp_s_pylanceRunCodeSnippet`
6. Same data flow as terminal invocation

### Key Insight: Same Python code, different UI layer (IDE vs. terminal)

---

## State Machine Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                       SKILL_STATE.json                          │
│  {dag_phase, md5_target_hash, anchor_oof_rmse, submissions}   │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ↓
        ┌────────────────────────┐
        │  Read challenge_config │
        │     (validate nulls)    │
        └────────────┬───────────┘
                     │
        ┌────────────▼───────────┐
        │   dag_phase = Phase N  │
        │                        │
        │  Select next skill(s)  │
        └────────────┬───────────┘
                     │
        ┌────────────▼───────────┐       ┌─────────────────────┐
        │  Execute Skill K       │───────→  DuckDB Ledger      │
        │  (train, EDA, etc.)    │       │  (experiments.db)   │
        └────────────┬───────────┘       └─────────────────────┘
                     │
        ┌────────────▼───────────┐
        │   Gate Check (Branch)  │
        │   OOF RMSE vs. Anchor  │
        └────────┬───────────────┘
                 │
          ┌──────┴──────┐
          │             │
       PASS          FAIL
          │             │
          ↓             ↓
    ┌─────────┐   ┌──────────┐
    │ Submit  │   │  Prune   │
    │ (if $)  │   │ (log)    │
    └────┬────┘   └──────────┘
         │
         ↓
    ┌────────────────────────┐
    │ Zindi API: submit CSV  │
    │ Get: my_rank, score    │
    └────┬───────────────────┘
         │
         ↓
    ┌────────────────────────┐
    │   Update SKILL_STATE   │
    │   Update DuckDB        │
    └────┬───────────────────┘
         │
         ↓
    ┌────────────────────────┐
    │ Next Phase or Halt     │
    │ (wait for user GO)     │
    └────────────────────────┘
```

---

## Module Dependencies

```
CLI / IDE
  ↓
main.py (orchestrator logic)
  ├→ state.py (read/write SKILL_STATE.json)
  ├→ config.py (read challenge_config.json)
  ├→ ledger.py (DuckDB)
  │   └→ sqlite3 / duckdb
  ├→ zindi_client.py (Zindi API wrapper)
  │   └→ requests / CLI subprocess
  └→ skills/skill_*.py (individual skills)
      ├→ state.py
      ├→ config.py
      ├→ ledger.py
      ├→ lightgbm / pandas / numpy / sklearn
      └→ zindi_client.py (for submissions)
```

---

## Error Handling & Recovery

**Fatal errors** (halt immediately, log to reports/):
- `ConfigNotPopulated`: A required field in challenge_config.json is null
- `StateFileNotFound`: SKILL_STATE.json missing (cannot resume)
- `BudgetExhausted`: No submissions remaining today
- `MD5MismatchError`: Target column hash changed (integrity breach)

**Non-fatal errors** (log, prune branch, continue):
- `SkillExecutionError`: A skill failed (e.g., model training crashed)
- `GateFail`: Branch OOF RMSE didn't beat anchor
- `SubmissionFormatError`: CSV doesn't match submission format

All errors are logged to `reports/` with timestamp and remediation steps.

