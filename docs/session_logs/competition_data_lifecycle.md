# Competition Data Lifecycle Management

**Target Environment:** AWS SageMaker Studio  
**Author:** Orioki — MCS 4.2, JKUAT  
**Last Updated:** June 2026

---

## Overview

Zindian Orchestrator manages competition data through a structured lifecycle from intake to archival. This document defines the data flow, integrity boundaries, and governance checkpoints that ensure reproducibility and compliance across all phases.

---

## Data Lifecycle Stages

```
Stage 1: Intake       → Raw data ingestion and fingerprinting
Stage 2: Lock         → MD5 hash lock and immutability boundary
Stage 3: Transform    → Feature engineering and cleaning
Stage 4: Validate     → OOF generation and gate evaluation
Stage 5: Submit       → Submission file generation and API call
Stage 6: Archive      → Post-competition storage and cleanup
```

---

## Stage 1: Intake (skill_01, skill_02)

### Purpose
Establish the data integrity boundary before any transformation.

### Data Sources
```
competitions/<slug>/data/
├── Train.csv              ← Training data with target
├── Test.csv               ← Test data without target
└── SampleSubmission.csv   ← Submission format template
```

### Integrity Checks (skill_01)

```python
# MD5 hash computation
import hashlib

def compute_file_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

# Lock hashes in config
file_hashes = {
    "train": compute_file_hash("data/Train.csv"),
    "test": compute_file_hash("data/Test.csv"),
    "sample_submission": compute_file_hash("data/SampleSubmission.csv")
}
```

### Fingerprinting (skill_02)

Extracts structural metadata into `challenge_config.json`:
- `data_shape`: Row and column counts
- `target_col`: Target column name
- `task_type`: classification | regression
- `temporal_signal`: Presence and column name
- `spatial_signal`: Lat/lon columns and group identifier
- `group_signal`: Group structure column
- `missingness_level`: low | moderate | high
- `target_distribution`: balanced | imbalanced | continuous_normal | continuous_skewed

**Critical rule:** All column names, target names, and structural identifiers are written to config during intake. No skill hardcodes these values.

---

## Stage 2: Lock (End of Phase 1)

### Config Temporal Lock

After `skill_05_cv` writes the CV strategy block, `challenge_config.json` becomes **strictly read-only**:

```
Phase 1 completion → config.lock() → read-only for all skills
Exception: skill_00 may write to community_signals array only
```

### Hash Verification

Every session start (ENFORCE mode preflight):
```python
# Verify raw data unchanged
current_hashes = {
    "train": compute_file_hash("data/Train.csv"),
    "test": compute_file_hash("data/Test.csv"),
    "sample_submission": compute_file_hash("data/SampleSubmission.csv")
}

locked_hashes = config["file_hashes"]

if current_hashes != locked_hashes:
    raise DataIntegrityViolation(
        "Raw data files modified after hash lock. "
        "All OOF scores computed against locked data are now invalid."
    )
```

### Data Patch Detection (skill_00)

Continuous monitoring for competition data updates:

```python
# skill_00_zindi_monitor.py
def monitor_discussion_board():
    """
    Polls discussion board every 6 hours.
    Detects admin posts announcing data patches or schema changes.
    """
    for post in recent_posts:
        if is_admin_post(post) and contains_data_patch_keywords(post):
            # ABSOLUTE HALT — do not proceed
            state_store.update(
                data_patch_detected=True,
                patch_halt_timestamp=datetime.now(timezone.utc).isoformat(),
                patch_description=extract_patch_summary(post)
            )
            raise DataPatchDetected(
                "Admin announced data patch. Pipeline halted. "
                "Human operator must choose: [R] RESTART or [A] ABORT."
            )
```

**Human decision gate:**
- **[R] RESTART:** Wipe all pipeline state, unlock config, rerun Phase 1 on patched data
- **[A] ABORT:** Terminate competition run, log patch as invalidating event

**Critical rule:** skill_00 does NOT automatically trigger skill_02 re-intake. Automatic re-intake would break the config temporal lock and invalidate all OOF scores.

---

## Stage 3: Transform (Phase 2A, Phase 2B)

### Cleaning Pipeline (skill_06)

```
Input:  Raw feature matrix from locked data files
Output: Cleaned feature matrix with imputation applied

Rules:
1. MNAR columns → binary indicator BEFORE fill
2. MCAR columns → median/mode fill
3. Constant columns → drop unconditionally
4. Order is mandatory — indicator before fill
```

### Feature Engineering (skill_07)

**Two-mode contract for target-dependent features:**

```python
def compute_spatial_lag_of_target(X, y, lat_col, lon_col, train_idx=None, mode="cv"):
    """
    mode="cv"        — fold-restricted (training fold targets only)
    mode="inference" — full training set (for test inference)
    """
    if mode == "cv":
        assert train_idx is not None
        X_fit = X.iloc[train_idx]
        y_fit = y.iloc[train_idx]
    else:
        X_fit = X
        y_fit = y
    
    # Compute spatial lag using X_fit and y_fit only
    # ...
```

**Structural features** (Haversine distance, nearest-neighbour arrays, non-target group counts) do not require two-mode treatment and may be computed on the full dataset at any time.

### Artifact Storage

```
competitions/<slug>/
├── data/
│   ├── Train.csv                    ← Raw (immutable)
│   ├── Test.csv                     ← Raw (immutable)
│   └── processed/
│       ├── X_train_cleaned.parquet  ← Post-skill_06
│       ├── X_train_variant_01.parquet ← Post-skill_07
│       └── X_test_variant_01.parquet  ← Post-skill_07
└── SKILL_STATE.json                 ← OOF scores, metadata
```

---

## Stage 4: Validate (Phase 3A, Phase 3B)

### OOF Generation

Every OOF-generating skill writes outputs via `write_oof_record()` from `zindian/state.py`:

```python
from zindian.state import write_oof_record, compute_secondary_metrics

# After CV loop completes
secondary = compute_secondary_metrics(y_true_concat, y_pred_concat)

write_oof_record(
    state_store=state_store,
    branch_name="variant-01",
    oof_array=oof_predictions,
    cv_strategy_id=resolved_cv_strategy_id,
    seed=config["reproducibility"]["seed"],
    model_config=model_params,
    secondary_metrics=secondary  # regression only
)
```

### OOF Schema

```json
{
  "branch_variant-01_oof": {
    "scores": [0.85, 0.87, 0.86, 0.84, 0.88],
    "cv_strategy_id": "stratified",
    "seed": 42,
    "branch_name": "variant-01",
    "model_config": { "n_estimators": 100, "max_depth": 7 },
    "secondary_metrics": {
      "mae": 0.123,
      "mape": 4.56,
      "r2": 0.789
    }
  }
}
```

### Gate Evaluation (skill_11)

```python
# Baseline selection with safe state access
retraining_active = SKILL_STATE.get("pseudo_label_result", {}).get("retraining_required", False)
challenge_active = SKILL_STATE.get("anchor_challenge", {}).get("active", False)

if retraining_active:
    baseline = SKILL_STATE["anchor_oof_score_augmented"]
elif challenge_active:
    baseline = SKILL_STATE["anchor_oof_score_challenged"]
else:
    baseline = SKILL_STATE["anchor_oof_score"]

# Directional check
direction = config["metric_direction"]
if direction == "maximize":
    improved = oof_score - baseline > effective_gate_margin
else:
    improved = baseline - oof_score > effective_gate_margin
```

---

## Stage 5: Submit (Phase 4)

### Submission File Generation (skill_14)

```python
# Task-aware validation
task_type = config["task_type"]
use_probs = config["use_probabilities"]

if task_type == "classification" and use_probs:
    # Probability submission
    assert all((df[target_col] > 0) & (df[target_col] < 1))
    assert df[target_col].apply(lambda x: len(str(x).split('.')[-1]) >= 6).all()
elif task_type == "classification" and not use_probs:
    # Hard label submission
    assert df[target_col].isin([0, 1]).all()
elif task_type == "regression":
    # Domain bounds check
    bounds = config["target_domain_bounds"]
    assert df[target_col].between(bounds["min"], bounds["max"]).all()
```

### Budget Guard (skill_16)

```python
# Three-tier budget system
live_remaining = client.get_remaining_submissions()

if live_remaining <= 0:
    state_store.update(submission_blocked=True, reason="budget_exhausted")
    raise HardAbortException("Submission budget exhausted.")

if live_remaining == 1:
    state_store.update(budget_warning={
        "remaining_submissions": 1,
        "source": "live",
        "timestamp": datetime.now(timezone.utc).isoformat()
    })
    # Requires explicit YES confirmation

# live_remaining >= 2: proceed normally
```

### Submission Tracking

```json
{
  "submissions": [
    {
      "submission_id": "sub_12345",
      "branch_name": "variant-01",
      "oof_score": 0.856,
      "public_lb_score": 0.842,
      "oof_to_lb_delta": 0.014,
      "timestamp": "2026-06-15T14:30:00Z",
      "comment": "Anchor baseline — LightGBM with TerraClimate features"
    }
  ]
}
```

---

## Stage 6: Archive (Post-Competition)

### Archival Workflow

```bash
#!/bin/bash
# scripts/archive_competition.sh

SLUG=$1
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
ARCHIVE_DIR="archives/${SLUG}-archive-${TIMESTAMP}"

# 1. Create archive directory
mkdir -p $ARCHIVE_DIR

# 2. Copy essential files
cp competitions/$SLUG/challenge_config.json $ARCHIVE_DIR/
cp competitions/$SLUG/SKILL_STATE.json $ARCHIVE_DIR/
cp -r competitions/$SLUG/reports/ $ARCHIVE_DIR/

# 3. Compress data
tar -czf $ARCHIVE_DIR/data.tar.gz competitions/$SLUG/data/

# 4. Upload to S3
aws s3 sync $ARCHIVE_DIR/ s3://my-archives/zindian/$SLUG/$TIMESTAMP/ \
    --storage-class GLACIER

# 5. Update history log
python scripts/update_history_log.py $SLUG

# 6. Clean local workspace
rm -rf competitions/$SLUG/

echo "Archive completed: s3://my-archives/zindian/$SLUG/$TIMESTAMP/"
```

### History Log Entry

```json
{
  "competition_id": "june-study-jam-forecasting",
  "task_type": "regression",
  "metric": "rmse",
  "metric_direction": "minimize",
  "cv_strategy_type": "KFold",
  "cv_strategy_override": false,
  "anchor_oof_score": 0.856,
  "best_promoted_oof_score": 0.842,
  "best_public_lb_score": 0.838,
  "oof_to_lb_delta": 0.004,
  "feature_types_used": ["temporal_lag", "rolling_mean", "group_agg"],
  "pseudo_label_ran": false,
  "final_rank": 12,
  "gate_thresholds": {
    "shap_leak_threshold": 3.0,
    "variance_gate_threshold": 0.01,
    "gate_margin": 0.001
  },
  "competition_close_date": "2026-06-30",
  "archive_location": "s3://my-archives/zindian/june-study-jam-forecasting/20260630-180000/"
}
```

---

## Data Governance Rules

### Immutability Boundaries

| File | Mutable Until | Immutable After |
|------|---------------|-----------------|
| `Train.csv`, `Test.csv` | Never | Always (hash-locked at intake) |
| `challenge_config.json` | Phase 1 completion | Phase 2 start (except `community_signals`) |
| `SKILL_STATE.json` | Competition close | Never (continuously updated) |
| OOF arrays | Branch promotion | Fusion (skill_13) |

### Reproducibility Contract

Every submission must be reproducible from:
1. `challenge_config.json` (locked config)
2. `SKILL_STATE.json` (execution trace)
3. Raw data files (hash-verified)
4. `requirements.txt` (pinned environment)

**Verification command:**
```bash
python scripts/verify_reproducibility.py <slug>
```

### Audit Trail

DuckDB ledger tracks all experiments:

```sql
-- Query experiment history
SELECT 
    branch_name,
    oof_score,
    feature_count,
    model_family,
    timestamp
FROM experiments
WHERE competition_id = 'june-study-jam-forecasting'
ORDER BY oof_score DESC;
```

---

## Monitoring and Alerts (skill_00)

### Continuous Monitoring

skill_00 runs two parallel monitors:

**1. Discussion Board Monitor (`skill_00_discussion_monitor.py`)**
- Polls every 6 hours
- Detects data patches, rule clarifications, deadline changes
- Writes findings to `community_signals` array

**2. Zindi Platform Monitor (`skill_00_zindi_monitor.py`)**
- Polls leaderboard after each submission
- Tracks OOF-to-LB delta for drift detection
- Flags overfit risk when delta > `drift_threshold`

### Drift Detection

```python
# After skill_16 submission
oof_score = SKILL_STATE["branch_variant-01_oof"]["scores"]
lb_score = client.get_latest_submission_score()
delta = abs(oof_score - lb_score)

drift_threshold = SKILL_STATE.get("drift_threshold", config.get("drift_threshold", 0.05))

if delta > drift_threshold:
    state_store.update(
        overfit_warning=True,
        oof_to_lb_delta=delta,
        flagged_branch="variant-01"
    )
    # Flag to skill_11_gate — block further promotion
```

---

## Data Retention Policy

### Active Competition
- **Location:** EFS (`/home/sagemaker-user/shared/zindian-orchestrator/competitions/<slug>/`)
- **Retention:** Until competition close + 7 days
- **Cost:** $0.30/GB-month

### Post-Competition (0-30 days)
- **Location:** S3 Standard
- **Retention:** 30 days
- **Cost:** $0.023/GB-month

### Long-Term Archive (30+ days)
- **Location:** S3 Glacier
- **Retention:** Indefinite
- **Cost:** $0.004/GB-month

### Cleanup Workflow

```bash
# Automated cleanup (runs monthly)
python scripts/cleanup_old_competitions.py --days 30 --archive
```

---

## Best Practices

### ✅ DO

- Lock file hashes immediately after intake
- Verify hashes at every session start
- Use two-mode contract for target-dependent features
- Write all OOF outputs via `write_oof_record()`
- Tag every OOF array with `cv_strategy_id`
- Monitor OOF-to-LB delta for drift
- Archive competitions to S3 Glacier after close
- Update history log for cross-competition learning

### ❌ DON'T

- Modify raw data files after hash lock
- Write to `challenge_config.json` after Phase 1 (except skill_00)
- Hardcode column names or competition-specific strings
- Skip fold restriction on target-dependent features
- Ignore data patch alerts from skill_00
- Delete competitions without archiving
- Bypass budget guards in skill_16

---

## Troubleshooting

### Issue: Hash mismatch on session start

```bash
# Cause: Raw data files modified after lock
# Solution: Restore from backup or restart competition

# 1. Check current hashes
python scripts/compute_file_hashes.py

# 2. Compare with locked hashes
jq '.file_hashes' competitions/<slug>/challenge_config.json

# 3. Restore from S3 backup
aws s3 sync s3://my-backups/zindian/<slug>/latest/ competitions/<slug>/data/
```

### Issue: OOF-to-LB delta > 0.05

```bash
# Cause: Overfitting or distribution shift
# Solution: Review feature engineering and CV strategy

# 1. Check drift warning
jq '.overfit_warning, .oof_to_lb_delta' competitions/<slug>/SKILL_STATE.json

# 2. Review SHAP audit
jq '.leaked_features' competitions/<slug>/SKILL_STATE.json

# 3. Consider CV strategy override at Gate 1
```

### Issue: Data patch detected mid-competition

```bash
# Cause: Admin announced data update
# Solution: Human decision required

# 1. Review patch description
jq '.data_patch_detected, .patch_description' competitions/<slug>/SKILL_STATE.json

# 2. Choose action:
#    [R] RESTART — wipe state, unlock config, rerun Phase 1
#    [A] ABORT   — terminate run, log as invalidating event

# 3. Execute choice
python scripts/handle_data_patch.py <slug> --action [R|A]
```

---

## References

- [Source of Truth Document](source_of_truth.md) — Section 2 (Core Architectural Principles)
- [AGENTS.md](../AGENTS.md) — OOF Contract, Two-Mode Feature Contract
- [Cost Optimization Guide](cost_optimization.md) — Storage strategy

---

**Last Updated:** June 2026  
**Maintained by:** Orioki — MCS 4.2, JKUAT
