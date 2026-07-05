# Zindian Orchestrator — Consolidated Troubleshooting Guide

This guide consolidates troubleshooting steps and fixes for common errors, runtime issues, and database bugs encountered across different sessions in the Zindian Orchestrator.

---

## 1. Package Shadowing Issues
*   **Problem:** Stubs or mock modules (such as `duckdb/`, `lightgbm/`, `google/`, or `zindi/`) located at the repository root folder shadowed the real installed site-packages. This is caused by Python prioritizing the current execution root (`sys.path[0]`) over `site-packages`.
*   **Symptoms:**
    *   *DuckDB Shadowing:* Database queries executing successfully but all data is silently lost on CLI process exit because the mock module uses an in-memory SQLite wrapper.
    *   *LightGBM Shadowing:* SHAP validation crashes with `InvalidModelError` because the mock LightGBM class is loaded.
    *   *Google Shadowing:* Failures when attempting to load Google GenAI / Auth APIs.
*   **Resolution:**
    *   Move all test stubs/fixtures from the repository root to `tests/fixtures/`.
    *   If a mock folder must remain, suffix the folder name with `_DISABLED` (e.g. `duckdb_mock_DISABLED/`, `zindi_local_DISABLED/`).
    *   Verify the real packages are installed in the virtual environment.

---

## 2. Ledger Database Durability & Lock Issues
*   **Problem:** Process context boundaries or wrong path resolution locations when running scripts from other directories than the repository root.
*   **Symptoms:**
    *   Database files generated under `/tmp/` or wrong paths.
    *   `sqlite3.OperationalError: database is locked` during parallel execution or type check runs (e.g., Mypy cache locked).
*   **Resolution:**
    *   *Path Resolution:* Never use `Path.cwd()` to resolve repo-level resources. Always resolve paths using `Path(__file__).resolve().parent.parent` (repo root) inside modules.
    *   *Database Durability:* Ensure the `Ledger` connections are instantiated using Python's context manager pattern (`with Ledger() as ledger:`), which handles flushing commits via checkpoints and closes connections safely on exit.
    *   *Locked Cache:* Clear locking issues in the type checker cache by running:
        *   **Unix/macOS:** `rm -rf .mypy_cache/`
        *   **Windows PowerShell:** `Remove-Item -Recurse -Force .mypy_cache`
        *   **Windows CMD:** `rmdir /s /q .mypy_cache`

---

## 3. Submission & Budget Tracking Failures (`skill_16_submit`)
*   **Problem:** Discrepancies between the live Zindi submission counts and the `SKILL_STATE.json` ledger.
*   **Symptoms:**
    *   The submission budget is decremented/incremented but no file is uploaded.
    *   Comments uploaded to Zindi carry stale OOF metric values from the global state instead of the branch-specific OOF.
*   **Resolution:**
    *   *Budget Drift:* Never increment the state budget count *before* making the Zindi API submission request. Only mutate the state after the Zindi client returns a successful API payload response.
    *   *Comment Traceability:* Do not read the generic global metric state key (e.g., `anchor_oof_score`). Instead, extract the branch-specific scores using the active git branch namespace (e.g., `state.get(f"branch_{branch}_oof")["scores"]`).

---

## 4. Competition Context Resolution Failures
*   **Problem:** Running CLI commands (like `status` or `phase`) yields "No competition found" errors.
*   **Resolution:**
    The orchestrator resolves the active competition slug using the following priority:
    1.  **CWD Context:** Run the command from inside the target competition subfolder under `competitions/<slug>/`.
    2.  **Environment Variables:** Set the environment variables:
        *   **Unix/macOS:**
            ```bash
            export ZINDIAN_COMPETITION="world-cup-2026-goal-prediction-challenge"
            ```
        *   **Windows PowerShell:**
            ```powershell
            $env:ZINDIAN_COMPETITION="world-cup-2026-goal-prediction-challenge"
            ```
        *   **Windows CMD:**
            ```cmd
            set ZINDIAN_COMPETITION=world-cup-2026-goal-prediction-challenge
            ```
    3.  **Local Environment:** Add the following line to your `.env` file at repo root:
        ```env
        ZINDIAN_COMPETITION=world-cup-2026-goal-prediction-challenge
        ```

    4.  **Auto-Select:** If only one competition directory is present under `competitions/`, the system will auto-select it.

---

## 5. Network Isolation and Zindi API Failures
*   **Problem:** Scripts attempting to fetch live Zindi pages or submit files fail with `ConnectionError` or `ImportError`.
*   **Symptoms:**
    *   `Skill failed: No module named 'zindi'` or network timeouts.
*   **Resolution:**
    *   During local testing, network isolation is enforced by the CI/CD pipeline environment variables (`ZINDIAN_DISABLE_NETWORK=1`).
    *   Ensure the mock API wrappers (in `zindi_stub_backup_DISABLED/`) are reviewed if testing isolated stub logic.
    *   Verify credentials exist in your `.env` file before executing live sync or submit operations.
