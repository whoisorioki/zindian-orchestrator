# Ledger Durability & Path Resolution Fix - Session 2026-06-17

## Root Causes (Two Separate Bugs)

### Bug 1: Mock DuckDB Module Shadowing (Primary)
A test fixture at `zindian-orchestrator/duckdb/__init__.py` was shadowing the real DuckDB library:
- Always used in-memory SQLite (`:memory:`) regardless of path provided
- Never wrote any data to disk
- Destroyed all data when connection closed
- Made every `Ledger()` instantiation start with fresh, empty database

**Impact**: Every experiment "recorded" across every competition was silently lost the moment each CLI process exited.

### Bug 2: Path Resolution Using `cwd()` (Secondary)
`resolve_competition_paths()` used `Path.cwd()` which depends on execution directory:
- Worked when called from repo root (cwd == repo root by coincidence)
- Failed when called from any other directory (`/tmp`, subprocesses, etc.)
- Caused auto-detect fallback to pick wrong competition based on `last_updated` timestamp
- Would have caused ledger to write to wrong location even after mock was fixed

**Impact**: Even with working DuckDB, database would be created in wrong directory.

## The Complete Fix

### 1. Disabled Mock DuckDB
```bash
mv zindian-orchestrator/duckdb zindian-orchestrator/duckdb_mock_DISABLED
```

### 2. Fixed Path Resolution
**Before:**
```python
root = Path.cwd()  # Unreliable - depends on where script runs from
```

**After:**
```python
root = Path(__file__).resolve().parent.parent  # Reliable - uses module location
```

### 3. Added Explicit Error on Bad Slug
Prevents silent fallthrough to auto-detect when user explicitly requests non-existent competition:
```python
if selected_slug:
    candidate = root / "competitions" / selected_slug
    if candidate.exists():
        comp_dir = candidate
    else:
        raise FileNotFoundError(...)  # Fail loudly instead of silent fallthrough
```

### 4. Added Context Manager Support
```python
def __enter__(self) -> "Ledger":
    return self

def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    self.close()
```

### 5. Enhanced close() Method
```python
def close(self) -> None:
    try:
        self.conn.execute("CHECKPOINT")  # Force WAL flush
    except Exception:
        pass
    self.conn.close()
```

### 6. Fixed Schema Creation Order
Moved sequence creation before table creation to avoid catalog errors.

### 7. Updated All Call Sites
Converted all `Ledger()` instantiations to context manager pattern:
- `zindian/skills/skill_16_submit.py`
- `zindian/skills/skill_08_anchor.py`
- `zindian/skills/skill_15_reporter.py`
- `zindian/cli.py`
- `scripts/backfill_ledger.py`

## Verification (Two-Process Test)

### Process 1: Write
```
Wrote experiment_id: 2
DB resolved to: .../june-study-jam.../reports/experiments.db
```

### Process 2: Read (separate invocation)
```
DB resolved to: .../june-study-jam.../reports/experiments.db
Row count: 2
  - experiment_id=1, branch=path-fix-test, oof_rmse=0.123
  - experiment_id=2, branch=verify-test, oof_rmse=0.5
```

✅ **Verified**: Data persists across process boundaries in correct competition directory

## Systemic Issue: Package Shadowing Pattern

This is the **second** package shadowing bug found in this session:
1. `lightgbm/` stub → SHAP InvalidModelError
2. `duckdb/` mock → Complete ledger data loss

Both were test fixtures at repo root shadowing real packages. See `PACKAGE_SHADOWING_AUDIT.md` for full analysis.

## Impact

- **Before**: 0% of experiments persisted (all lost on process exit)
- **After**: 100% of experiments persist durably to correct location
- All historical data prior to this fix is unrecoverable
