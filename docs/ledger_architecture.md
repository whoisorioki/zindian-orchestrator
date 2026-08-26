# Ledger Architecture

**Version:** 2.8
**Last Updated:** August 2026
**Authority:** Binding for all experiment tracking

---

## 1. Purpose

Define the DuckDB-based experiment tracking system, persistence guarantees, and multi-process safety contracts.

---

## 2. Schema Definition

### 2.1 Experiments Table

```sql
CREATE SEQUENCE IF NOT EXISTS experiments_id_seq START 1;

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id       INTEGER PRIMARY KEY DEFAULT nextval('experiments_id_seq'),
    branch_name         VARCHAR NOT NULL,
    oof_score           FLOAT,
    metric              VARCHAR,
    oof_rmse            FLOAT,
    feature_count       INTEGER,
    calibration_method  VARCHAR,
    gate_result         VARCHAR,
    gate_reason         VARCHAR,
    md5_target_hash     VARCHAR,
    dag_phase           VARCHAR,
    notes               VARCHAR,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 2.2 Submissions Table

```sql
CREATE SEQUENCE IF NOT EXISTS submissions_id_seq START 1;

CREATE TABLE IF NOT EXISTS submissions (
    submission_id       INTEGER PRIMARY KEY DEFAULT nextval('submissions_id_seq'),
    experiment_id       INTEGER,
    branch_name         VARCHAR NOT NULL,
    submission_rank     INTEGER,
    public_score        FLOAT,
    private_score       FLOAT,
    my_rank             INTEGER,
    selected_for_final  BOOLEAN DEFAULT FALSE,
    selection_rationale VARCHAR,
    comment             VARCHAR,
    submitted_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);
```

---

## 3. Context Manager Lifecycle

### 3.1 Correct Usage Pattern

```python
from zindian.ledger import Ledger

with Ledger() as ledger:
    experiment_id = ledger.log_experiment(
        branch_name="variant-10",
        oof_score=0.5523,
        metric="rmse",
        feature_count=28,
        gate_result="PASS"
    )
    # Connection auto-closed on __exit__
```

### 3.2 Forbidden Patterns

**WRONG:**
```python
ledger = Ledger()
ledger.log_experiment(...)
# Connection never closed → file lock persists
```

**Rationale:** DuckDB uses file-level locking. Unclosed connections block concurrent access and cause "database is locked" errors.

---

## 4. Persistence Guarantees

### 4.1 Checkpoint Operations

All write operations MUST call `CHECKPOINT` to flush WAL to disk:

```python
def log_experiment(self, ...):
    cursor.execute("INSERT INTO experiments ...")
    self.conn.execute("CHECKPOINT")  # Force WAL flush
    return exp_id
```

### 4.2 Verification Test

```python
def test_ledger_persistence():
    """Verify writes survive process termination."""
    with Ledger() as ledger:
        exp_id = ledger.log_experiment(branch_name="test")

    # Simulate process restart
    with Ledger() as ledger:
        result = ledger.get_experiment(exp_id)
        assert result is not None
```

---

## 5. Multi-Process Safety

### 5.1 Read-Only Queries

Safe for concurrent access:
```python
with Ledger() as ledger:
    experiments = ledger.get_all_experiments()
```

### 5.2 Write Operations

In-process write serialization across threads (e.g. main execution thread and deep-research daemon threads) is guaranteed via a class-level `threading.Lock` (`Ledger._write_lock`).

For multi-process access across distinct CLI invocations, serialize via an atomic directory lock (cross-platform, zero dependencies) or external packages like portalocker:
```python
lock_dir = Path("reports/experiments.db.lockdir")
try:
    # mkdir is atomic on both Windows and POSIX systems
    lock_dir.mkdir(parents=True, exist_ok=False)
    with Ledger() as ledger:
        ledger.log_experiment(...)
finally:
    try:
        lock_dir.rmdir()
    except OSError:
        pass
```

---

## 6. Error Handling

### 6.1 Database Locked

```python
try:
    with Ledger() as ledger:
        ledger.log_experiment(...)
except Exception as e:
    if "database is locked" in str(e).lower():
        # Retry with exponential backoff
        time.sleep(0.1)
        retry_log_experiment()
```

### 6.2 Corruption Recovery

Make a backup of the experiments database file before attempting repair:

```bash
# Rebuild from WAL
duckdb reports/experiments.db "CHECKPOINT; VACUUM;"
```


---

## 7. Migration Protocol

When schema changes are required:

1. **Version** the schema in `ledger.py`:
   ```python
   SCHEMA_VERSION = 2
   ```

2. **Create migration** in `scripts/migrate_ledger.py`

3. **Test** on fixture database before production

4. **Document** in `CHANGELOG.md`

---

## 8. Performance Benchmarks

| Operation | Latency (p50) | Latency (p99) |
|-----------|---------------|---------------|
| log_experiment | 2.3ms | 8.1ms |
| get_experiment | 0.8ms | 2.4ms |
| get_all_experiments | 12.5ms | 45.2ms |

**Measured on:** ml.t3.medium, 100 experiments

---

## 9. References

- DuckDB Docs: https://duckdb.org/docs/connect/concurrency
- Implementation: `zindian/ledger.py`
