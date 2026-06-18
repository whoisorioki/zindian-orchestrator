# Package Shadowing Audit - Session 2026-06-17 (COMPLETE)

## All Shadowing Issues Resolved

### 1. `duckdb/` ✅ RESOLVED
- **Status**: Mock disabled, real package installed
- **Impact**: Caused complete ledger data loss - all experiments silently lost
- **Root cause**: Mock test fixture at repo root shadowing real DuckDB
- **Resolution**: 
  - `mv duckdb/ duckdb_mock_DISABLED/`
  - `pip install duckdb` → v1.5.3 installed
  - Verified: Two-process ledger write/read test successful
- **Verification**: Database persists at correct path across process boundaries

### 2. `lightgbm/` ✅ RESOLVED
- **Status**: Previously identified and resolved
- **Impact**: SHAP InvalidModelError during diagnostics
- **Root cause**: Empty stub shadowing real LightGBM package
- **Resolution**: Stub removed/disabled in prior session

### 3. `google/` ✅ RESOLVED
- **Status**: Moved to test fixtures, real packages installed
- **Impact**: Would have broken Google GenAI/Auth functionality
- **Root cause**: CI testing stub at repo root
- **Resolution**:
  - `mv google/ tests/fixtures/google/`
  - `pip install google-auth` → v2.53.0 installed
  - `pip install google-genai` → v2.8.0 installed
- **Verification**: No import conflicts, stubs isolated to tests

### 4. `zindi/` ✅ RESOLVED
- **Status**: Stub disabled, real package installed
- **Impact**: Submission board returned fabricated dummy data
- **Root cause**: Local stub shadowing missing real `zindi==0.0.4` package
- **Evidence**: `[zindi stub] select_a_challenge` message in output
- **Resolution**:
  - `mv zindi/ zindi_local_DISABLED/`
  - `mv zindi_stub_backup/ zindi_stub_backup_DISABLED/`
  - `pip install zindi==0.0.4` → v0.0.4 installed
- **Verification**: Real submission board retrieved:
  ```
  6QGgj8hi  2026-06-17  0.552117936      sub_010_anchor.csv
  8wmrgAp4  2026-06-16  0.552117936  YES sub_009_anchor.csv
  GbjnsDnP  2026-06-16  0.552117936  YES sub_009_anchor.csv
  ```
  Real IDs, real dates, real scores confirmed

## Critical Finding: Gate 5 Status

**Discovered during verification**: Both `YES` selection markers point to the same file:
- `8wmrgAp4` → `sub_009_anchor.csv` ✓ Selected
- `GbjnsDnP` → `sub_009_anchor.csv` ✓ Selected  
- `6QGgj8hi` → `sub_010_anchor.csv` ✗ Not selected

**Impact**: Gate 5 currently holds **one distinct submission in two slots**, not two diverse submissions as intended. No hedge against public/private LB divergence.

**Action needed**: Replace one `sub_009` selection with `sub_010` or next variant before competition close.

## Pattern Analysis

**Root Cause**: Test fixtures and stubs placed at repository root where Python's import resolution finds them before site-packages.

**Systemic Risk**: Any directory at repo root with the same name as a PyPI dependency will shadow that dependency for ALL code in the repo - production, tests, and CLI.

**Impact Severity**:
- `duckdb/` - CRITICAL (complete data loss)
- `lightgbm/` - HIGH (runtime errors in production features)
- `google/` - HIGH (breaks Google AI functionality)
- `zindi/` - MEDIUM (intentional but undocumented, maintenance risk)

## Recommendations

1. **Immediate**: Resolve `google/` stub (move to tests/ or delete)
2. **Immediate**: Document `zindi/` override or rename to avoid confusion
3. **Short-term**: Move ALL test fixtures to `tests/fixtures/`
4. **Short-term**: Use `conftest.py` with proper `sys.path` scoping for test-only mocks
5. **Long-term**: Add pre-commit hook to detect new directories shadowing dependencies
6. **Long-term**: Add CI check comparing `ls -d */` against `requirements.txt` package names
