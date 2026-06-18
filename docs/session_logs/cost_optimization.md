# Zindian Orchestrator — Cost Optimization Guide

**Target environments:** AWS SageMaker, Local development  
**Author:** Orioki — MCS 4.2, JKUAT  
**Last updated:** June 2026

---

## Overview

Zindian Orchestrator is designed with a **local-first architecture** that minimizes cloud compute costs and API rate limit consumption. This guide covers cost optimization strategies for both Zindi API submissions and AWS SageMaker infrastructure.

---

## 1. Zindi API Rate Limits & Budget Management

### 1.1 Submission Budget System

**Three-tier budget guard** (implemented in `skill_16_submit.py`):

```python
# Tier 1: Hard abort (0 remaining)
if live_remaining <= 0:
    raise HardAbortException("Submission budget exhausted.")

# Tier 2: Warning + confirmation (1 remaining)
if live_remaining == 1:
    state["budget_warning"] = {
        "remaining_submissions": 1,
        "source": "live",
        "timestamp": ISO_timestamp
    }
    # Requires explicit YES confirmation

# Tier 3: Normal flow (≥2 remaining)
```

**Budget tracking fields in `SKILL_STATE.json`:**
- `submissions_used_today` — daily counter
- `submissions_used_total` — lifetime counter
- `remaining_submissions` — live API query before each submit
- `budget_warning` — last warning timestamp and source

### 1.2 Gate System (Cost Prevention)

Five human gates prevent wasteful submissions:

| Gate | Checkpoint | Prevents |
|------|-----------|----------|
| **Gate 1** | After anchor baseline | Submitting weak baseline |
| **Gate 2** | Per-branch promotion | Submitting untested variants |
| **Gate 3** | Before ensemble fusion | Submitting uncalibrated models |
| **Gate 4** | Before inference | Submitting malformed CSVs |
| **Gate 5** | Final selection | Wrong private LB picks |

**Gating logic** (from `skill_11_gate.py`):
```python
# Only submit if OOF score beats baseline by margin
if metric_direction == "maximize":
    improved = oof_score - baseline > effective_gate_margin
else:
    improved = baseline - oof_score > effective_gate_margin

# Variance check (stability filter)
if fold_score_variance > effective_variance_threshold:
    # Block unstable models
```

### 1.3 Typical Competition Budget Profile

**Zindi typical limit:** 5 submissions/day  
**Framework target:** ≤7 total submissions per competition (multi-day strategy)  
**Gate rejection rate:** ~40% (saves 3-4 submissions)

```
Phase 0: Foundation          → 0 submissions
Phase 1: Integrity + Intake  → 0 submissions
Phase 2: Anchor Baseline     → 1 submission (after Gate 1)
Phase 3: Feature variants    → 2-4 submissions (gated)
Phase 4: Calibration         → 1-2 submissions (gated)
Phase 5: Final selection     → 0 submissions (selects from existing)
─────────────────────────────────────────────
Total: 4-7 submissions per competition
```

### 1.4 Local Validation (Zero API Cost)

All validation runs locally before any API call:

```python
# Structural validation (8 checks)
validate(sub_path, sample_path, config)

# Task-aware value validation
_value_validation_errors(df, target_col, task_type, use_probs, bounds)

# OOF score computation
determine_submission_metrics(submission_file, state)

# Only then: API submission (consumes budget)
client.submit(filepath, comment)
```

### 1.5 Budget Monitoring Commands

```bash
# Check current budget and submission history
python -m zindian.skills.skill_16_submit --submission-board

# View state-tracked budget
cat competitions/<slug>/SKILL_STATE.json | jq '.submissions_used_today'
cat competitions/<slug>/SKILL_STATE.json | jq '.remaining_submissions'
cat competitions/<slug>/SKILL_STATE.json | jq '.budget_warning'
```

---

## 2. AWS SageMaker Cost Optimization

### 2.1 Instance Pricing Reference

| Instance Type | vCPU | RAM | Price/hour | Use Case |
|--------------|------|-----|------------|----------|
| ml.t3.medium | 2 | 4GB | $0.05 | Interactive work, Phase 0-1, Phase 5 |
| ml.m5.large | 2 | 8GB | $0.115 | Light training, Phase 2, Phase 4 |
| ml.m5.xlarge | 4 | 16GB | $0.23 | Heavy training, Phase 3 |
| ml.c5.2xlarge | 8 | 16GB | $0.408 | Compute-intensive jobs |

**Storage costs:**
- EFS Standard: $0.30/GB-month
- EFS Infrequent Access: $0.025/GB-month
- S3 Standard: $0.023/GB-month
- S3 Glacier: $0.004/GB-month

### 2.2 Typical Competition Cost Profile

**Optimized workflow:**

| Phase | Duration | Instance | Cost |
|-------|----------|----------|------|
| Phase 0-1 (Setup) | 15 min | ml.t3.medium | $0.01 |
| Phase 2 (Anchor) | 30 min | ml.m5.large | $0.06 |
| Phase 3 (Features) | 2 hours | ml.m5.xlarge | $0.46 |
| Phase 4 (Calibration) | 30 min | ml.m5.large | $0.06 |
| Phase 5 (Selection) | 10 min | ml.t3.medium | $0.01 |
| **Total** | **3.5 hours** | | **$0.60** |

**vs. keeping ml.m5.xlarge running 24/7:**
- Always-on cost: $0.23/hr × 24 × 30 = **$165.60/month**
- Optimized cost: $0.60 × 5 competitions = **$3.00/month**
- **Savings: 98.2%**

### 2.3 Architecture Cost Advantages

**Local-first design:**
```python
# All validation runs in-instance (no additional compute)
validate(sub_path, sample_path, config)  # Free
determine_submission_metrics(...)         # Free
_value_validation_errors(...)             # Free

# DuckDB ledger (no S3 PUT/GET charges)
ledger.log_experiment(branch, oof_score, features)

# Atomic state updates (single write per skill)
state_store.update(dag_phase="phase_2_done", anchor_oof_score=0.85)
```

**No SageMaker API calls:**
- Framework runs entirely in-instance
- Only external API: Zindi submission (5/day limit)
- No risk of hitting AWS service quotas

### 2.4 Cost Optimization Strategies

#### A. Use Lifecycle Configurations

Auto-stop idle instances after 30 minutes:
```bash
# Saves ~$0.05/hour × 16 hours/day = $0.80/day
# In SageMaker Studio: File → Shut Down → Shut Down All
```

#### B. Leverage SageMaker Processing Jobs

For heavy training (Phase 3), use Processing Jobs instead of notebooks:

```python
from sagemaker.processing import ScriptProcessor

processor = ScriptProcessor(
    role=role,
    image_uri='<your-container>',
    instance_type='ml.m5.xlarge',
    instance_count=1,
    max_runtime_in_seconds=3600  # Auto-terminate after 1 hour
)

processor.run(
    code='zindian/skills/skill_08_anchor.py',
    arguments=['--config', 'challenge_config.json']
)
# Spins up compute only when needed, auto-terminates
```

#### C. Use Spot Instances (70% discount)

For non-critical training jobs:

```python
processor = ScriptProcessor(
    ...,
    use_spot_instances=True,
    max_wait_time_in_seconds=7200
)
# ml.m5.xlarge: $0.23/hr → $0.069/hr
```

#### D. Batch Experiments

Run all Phase 3 variants in one session:

```python
# Single instance session
for variant in ["v1", "v2", "v3"]:
    skill_07.run(variant)
    skill_11.run()  # Gate locally

# vs. 3 separate instance sessions (3× cost)
```

#### E. Use EFS Lifecycle Policies

Move old competition data to Infrequent Access:
```bash
# $0.30/GB → $0.025/GB after 30 days
# Configure in EFS console: Lifecycle management
```

#### F. Store Archives in S3

```bash
# After competition ends, archive to S3
aws s3 sync competitions/<slug>/ s3://my-bucket/archives/<slug>/ \
  --storage-class GLACIER

# EFS: $0.30/GB-month → S3 Glacier: $0.004/GB-month (98.7% savings)
```

### 2.5 AWS Service Quotas

**SageMaker API limits:**
```
CreateProcessingJob:     5 TPS (transactions/second)
DescribeProcessingJob:   10 TPS
StopProcessingJob:       5 TPS

Studio Apps:             10 concurrent per account
Processing Jobs:         100 concurrent per account
Training Jobs:           100 concurrent per account
```

**Zindian impact:** Zero SageMaker API calls (runs in-instance)

### 2.6 Cost Monitoring

**Check current instance:**
```bash
cat /opt/ml/metadata/resource-metadata.json | jq '.ResourceName'
```

**Monitor resource usage:**
```bash
htop                    # CPU/memory
df -h /home/sagemaker-user/shared  # EFS storage
```

**Track compute time in state:**
```python
# Add to SKILL_STATE.json
{
  "compute_tracking": {
    "phase_2_duration_minutes": 30,
    "phase_3_duration_minutes": 120,
    "total_instance_hours": 3.5,
    "estimated_cost_usd": 0.60
  }
}
```

**AWS Cost Explorer queries:**
```bash
# Check running instances
aws sagemaker list-notebook-instances --status-equals InService

# Check processing jobs (last 7 days)
aws sagemaker list-processing-jobs \
  --creation-time-after $(date -d '7 days ago' -Iseconds)

# Estimate current month cost
aws ce get-cost-and-usage \
  --time-period Start=$(date +%Y-%m-01),End=$(date +%Y-%m-%d) \
  --granularity MONTHLY \
  --metrics BlendedCost \
  --filter file://sagemaker-filter.json
```

---

## 3. Best Practices Checklist

### ✅ DO

- **Use lifecycle configs** to auto-stop idle instances
- **Run heavy training in Processing Jobs** (auto-terminate)
- **Use Spot instances** for non-critical jobs (70% savings)
- **Batch experiments** in single sessions
- **Store data in S3** (cheaper than EFS for archives)
- **Always approve gates manually** — prevents auto-submission of bad models
- **Review OOF scores before Gate 2** — only promote strong branches
- **Use variance threshold** — blocks unstable models early
- **Leverage secondary metrics** (MAE, MAPE, R²) — diagnostic confidence
- **Run preflight checks** — `python scripts/preflight_enforce.py`

### ❌ DON'T

- **Leave instances running overnight** ($0.23 × 8 = $1.84 wasted)
- **Use oversized instances for light work** (t3 → m5 = 2.3× cost)
- **Store large datasets in EFS long-term** (use S3 Glacier)
- **Run interactive notebooks for long training** (use Jobs)
- **Bypass gates** — they exist to save budget
- **Submit without OOF validation** — blind submissions waste quota
- **Ignore budget warnings** — last submission should be your best

---

## 4. Cost Tracking Template

Add to `SKILL_STATE.json` for cost awareness:

```json
{
  "cost_tracking": {
    "competition_start_date": "2026-06-01",
    "total_instance_hours": 3.5,
    "instance_breakdown": {
      "ml.t3.medium": 0.5,
      "ml.m5.large": 1.0,
      "ml.m5.xlarge": 2.0
    },
    "estimated_compute_cost_usd": 0.60,
    "storage_gb": 2.5,
    "estimated_storage_cost_usd": 0.75,
    "total_estimated_cost_usd": 1.35,
    "submissions_used": 5,
    "submissions_remaining": 0,
    "cost_per_submission": 0.27
  }
}
```

---

## 5. Quick Reference

### Zindi Budget Commands

```bash
# View submission board
python -m zindian.skills.skill_16_submit --submission-board

# Check state budget
jq '.submissions_used_today, .remaining_submissions, .budget_warning' \
  competitions/<slug>/SKILL_STATE.json
```

### SageMaker Cost Commands

```bash
# Stop all instances
# In Studio: File → Shut Down → Shut Down All

# Check instance type
cat /opt/ml/metadata/resource-metadata.json | jq '.ResourceName'

# Monitor usage
htop
df -h /home/sagemaker-user/shared

# List running instances
aws sagemaker list-notebook-instances --status-equals InService
```

### Cost Estimation Formula

```
Total Cost = (Instance Hours × Hourly Rate) + (Storage GB × $0.30/month)

Example:
  3.5 hours × $0.23/hr (ml.m5.xlarge) = $0.805
  2.5 GB × $0.30/month = $0.75
  Total = $1.56 per competition
```

---

## 6. Advanced Optimizations

### A. SageMaker Pipelines

Define entire Phase 0-5 as a pipeline:
```python
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep

# Auto-scales compute per step
# Only pay for active steps
```

### B. SageMaker Model Registry

Store anchor models in registry (free):
```python
# vs. re-training from scratch each session
model_package = model.register(...)
```

### C. S3 Intelligent-Tiering

For competition data archives:
```bash
# Auto-moves to cheaper storage after 30 days
aws s3api put-bucket-intelligent-tiering-configuration \
  --bucket my-bucket \
  --id archive-config \
  --intelligent-tiering-configuration file://config.json
```

### D. CloudWatch Cost Anomaly Detection

Alert if daily spend > $5 (catches runaway instances):
```bash
aws ce create-anomaly-monitor \
  --anomaly-monitor file://monitor-config.json
```

---

## Summary

**Zindian Orchestrator's local-first architecture minimizes costs:**

- **Zindi API:** 4-7 submissions per competition (vs. 15-20 manual)
- **SageMaker:** $0.60 per competition (vs. $165/month always-on)
- **Total savings:** 95-98% vs. unoptimized workflows

**Key principles:**
1. Validate locally before API calls
2. Use smallest instance for each phase
3. Auto-stop idle instances
4. Batch experiments in single sessions
5. Gate every submission manually

---

**Last Updated:** June 2026  
**Maintained by:** Orioki — MCS 4.2, JKUAT
