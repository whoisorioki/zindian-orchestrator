# Storage and Cost Governance

**Target Environment:** AWS SageMaker Studio  
**Author:** Orioki — MCS 4.2, JKUAT  
**Last Updated:** June 2026

---

## Overview

Zindian Orchestrator implements a **cost-conscious storage strategy** that balances performance, durability, and cost across the competition lifecycle. This document defines storage tiers, cost allocation, and governance policies that keep infrastructure costs predictable and minimal.

---

## Storage Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Active Competition (EFS)                               │
│  /home/sagemaker-user/shared/zindian-orchestrator/     │
│  ├── competitions/<slug>/                              │
│  │   ├── challenge_config.json        (5 KB)          │
│  │   ├── SKILL_STATE.json             (50 KB)         │
│  │   ├── data/                        (500 MB)        │
│  │   └── reports/                     (10 MB)         │
│  Cost: $0.30/GB-month                                  │
└─────────────────────────────────────────────────────────┘
                    ↓ (competition close + 7 days)
┌─────────────────────────────────────────────────────────┐
│  Recent Archive (S3 Standard)                           │
│  s3://my-archives/zindian/<slug>/                      │
│  Cost: $0.023/GB-month (92% savings vs EFS)            │
└─────────────────────────────────────────────────────────┘
                    ↓ (30 days)
┌─────────────────────────────────────────────────────────┐
│  Long-Term Archive (S3 Glacier)                         │
│  s3://my-archives/zindian/<slug>/                      │
│  Cost: $0.004/GB-month (98.7% savings vs EFS)          │
└─────────────────────────────────────────────────────────┘
```

---

## Storage Tiers

### Tier 1: Active Competition (EFS)

**Purpose:** High-performance storage for active pipeline execution

**Location:** `/home/sagemaker-user/shared/zindian-orchestrator/competitions/<slug>/`

**Characteristics:**
- Persistent across instance restarts
- Shared across multiple instances
- Low-latency access (sub-millisecond)
- Automatic backups
- POSIX-compliant filesystem

**Cost:** $0.30/GB-month (EFS Standard)

**Typical size per competition:**
```
challenge_config.json:     5 KB
SKILL_STATE.json:         50 KB
data/ (raw):             500 MB
data/processed/:         200 MB
reports/:                 10 MB
notebooks/:               20 MB
─────────────────────────────────
Total:                   ~730 MB → $0.22/month
```

**Retention:** Until competition close + 7 days

**Cleanup trigger:**
```bash
# Automated cleanup (runs daily)
python scripts/cleanup_efs.py --days-after-close 7
```

---

### Tier 2: Recent Archive (S3 Standard)

**Purpose:** Fast-access archive for recent competitions

**Location:** `s3://my-archives/zindian/<slug>/<timestamp>/`

**Characteristics:**
- 99.999999999% (11 9's) durability
- Millisecond retrieval latency
- Versioning enabled
- Lifecycle policies for automatic tiering

**Cost:** $0.023/GB-month (92% savings vs EFS)

**Typical size per competition:**
```
challenge_config.json:     5 KB
SKILL_STATE.json:         50 KB
data.tar.gz:             300 MB (compressed)
reports/:                 10 MB
─────────────────────────────────
Total:                   ~310 MB → $0.007/month
```

**Retention:** 30 days

**Lifecycle policy:**
```json
{
  "Rules": [
    {
      "Id": "ArchiveToGlacier",
      "Status": "Enabled",
      "Transitions": [
        {
          "Days": 30,
          "StorageClass": "GLACIER"
        }
      ]
    }
  ]
}
```

---

### Tier 3: Long-Term Archive (S3 Glacier)

**Purpose:** Cost-optimized long-term storage

**Location:** `s3://my-archives/zindian/<slug>/<timestamp>/`

**Characteristics:**
- 99.999999999% (11 9's) durability
- 3-5 hour retrieval latency (Standard)
- 1-5 minute retrieval (Expedited, higher cost)
- Immutable after write

**Cost:** $0.004/GB-month (98.7% savings vs EFS)

**Typical size per competition:**
```
Compressed archive:      300 MB → $0.001/month
```

**Retention:** Indefinite

**Retrieval workflow:**
```bash
# Initiate retrieval (Standard: 3-5 hours)
aws s3api restore-object \
    --bucket my-archives \
    --key zindian/<slug>/<timestamp>/data.tar.gz \
    --restore-request Days=7,GlacierJobParameters={Tier=Standard}

# Check restoration status
aws s3api head-object \
    --bucket my-archives \
    --key zindian/<slug>/<timestamp>/data.tar.gz

# Download after restoration completes
aws s3 cp s3://my-archives/zindian/<slug>/<timestamp>/ . --recursive
```

---

## Cost Allocation Model

### Per-Competition Cost Breakdown

```
┌─────────────────────────────────────────────────────────┐
│  Competition Lifecycle Cost (30-day active period)      │
├─────────────────────────────────────────────────────────┤
│  Compute (3.5 hours optimized):                         │
│    Phase 0-1: ml.t3.medium × 0.5h    = $0.025          │
│    Phase 2:   ml.m5.large × 1.0h     = $0.115          │
│    Phase 3:   ml.m5.xlarge × 2.0h    = $0.460          │
│    Phase 4:   ml.m5.large × 0.5h     = $0.058          │
│    Phase 5:   ml.t3.medium × 0.5h    = $0.025          │
│                                Subtotal: $0.683         │
├─────────────────────────────────────────────────────────┤
│  Storage (730 MB):                                      │
│    EFS (30 days):    0.73 GB × $0.30 = $0.219          │
│    S3 Standard (30d): 0.31 GB × $0.023 = $0.007        │
│    S3 Glacier (∞):   0.30 GB × $0.004 = $0.001/month   │
│                                Subtotal: $0.227         │
├─────────────────────────────────────────────────────────┤
│  Data Transfer:                                         │
│    S3 uploads:       0.31 GB × $0.00  = $0.000         │
│    (first 1 GB/month free)                              │
│                                Subtotal: $0.000         │
├─────────────────────────────────────────────────────────┤
│  TOTAL PER COMPETITION:                    $0.910       │
└─────────────────────────────────────────────────────────┘
```

### Monthly Cost Projection

**Scenario: 5 competitions per month**

```
Compute:  $0.683 × 5 = $3.415
Storage:  $0.227 × 5 = $1.135
Transfer: $0.000 × 5 = $0.000
─────────────────────────────
Total:                 $4.550/month
```

**vs. Always-on ml.m5.xlarge:**
```
$0.23/hr × 24 × 30 = $165.60/month
Savings: $161.05/month (97.3%)
```

---

## Cost Governance Policies

### Policy 1: Instance Auto-Shutdown

**Rule:** Shut down instances after 30 minutes of inactivity

**Implementation:**
```bash
# Lifecycle configuration
IDLE_TIME=1800  # 30 minutes

while true; do
    LAST_ACTIVITY=$(stat -c %Y /home/sagemaker-user/.jupyter/lab/workspaces/*.jupyterlab-workspace 2>/dev/null | sort -n | tail -1)
    CURRENT_TIME=$(date +%s)
    IDLE_DURATION=$((CURRENT_TIME - LAST_ACTIVITY))
    
    if [ $IDLE_DURATION -gt $IDLE_TIME ]; then
        sudo shutdown -h now
    fi
    
    sleep 300
done
```

**Cost impact:** Prevents $0.23/hr × 16 hrs/day = $3.68/day waste

---

### Policy 2: Right-Sized Instances

**Rule:** Use smallest instance type sufficient for each phase

**Decision matrix:**

| Phase | CPU Req | Memory Req | Instance | Cost/hr |
|-------|---------|------------|----------|---------|
| 0-1 | Low | <4GB | ml.t3.medium | $0.05 |
| 2 | Medium | 4-8GB | ml.m5.large | $0.115 |
| 3 | High | 8-16GB | ml.m5.xlarge | $0.23 |
| 4 | Medium | 4-8GB | ml.m5.large | $0.115 |
| 5 | Low | <4GB | ml.t3.medium | $0.05 |

**Enforcement:**
```python
# scripts/verify_instance_size.py
def verify_instance_for_phase(phase, current_instance):
    recommended = {
        "phase_0": "ml.t3.medium",
        "phase_1": "ml.t3.medium",
        "phase_2": "ml.m5.large",
        "phase_3": "ml.m5.xlarge",
        "phase_4": "ml.m5.large",
        "phase_5": "ml.t3.medium"
    }
    
    if current_instance != recommended[phase]:
        print(f"WARNING: Using {current_instance} for {phase}")
        print(f"Recommended: {recommended[phase]}")
        print(f"Potential waste: ${calculate_cost_delta(current_instance, recommended[phase])}/hr")
```

---

### Policy 3: EFS Lifecycle Management

**Rule:** Move inactive data to Infrequent Access after 30 days

**Configuration:**
```bash
# EFS Lifecycle Policy (AWS Console or CLI)
aws efs put-lifecycle-configuration \
    --file-system-id fs-12345678 \
    --lifecycle-policies \
        TransitionToIA=AFTER_30_DAYS
```

**Cost impact:**
- EFS Standard: $0.30/GB-month
- EFS IA: $0.025/GB-month
- Savings: 91.7% on inactive data

---

### Policy 4: S3 Intelligent-Tiering

**Rule:** Automatically move S3 objects to optimal storage class

**Configuration:**
```json
{
  "Id": "IntelligentTiering",
  "Status": "Enabled",
  "Tierings": [
    {
      "Days": 90,
      "AccessTier": "ARCHIVE_ACCESS"
    },
    {
      "Days": 180,
      "AccessTier": "DEEP_ARCHIVE_ACCESS"
    }
  ]
}
```

**Cost impact:**
- S3 Standard: $0.023/GB-month
- S3 Intelligent-Tiering Archive: $0.004/GB-month
- Savings: 82.6% on rarely accessed data

---

### Policy 5: Submission Budget Enforcement

**Rule:** Hard abort at zero remaining submissions

**Implementation (skill_16):**
```python
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
```

**Cost impact:** Prevents wasted submissions on weak models

---

## Cost Tracking and Reporting

### State-Based Cost Tracking

```json
{
  "cost_tracking": {
    "competition_start_date": "2026-06-01",
    "total_instance_hours": 3.5,
    "instance_breakdown": {
      "ml.t3.medium": 1.0,
      "ml.m5.large": 1.5,
      "ml.m5.xlarge": 2.0
    },
    "estimated_compute_cost_usd": 0.683,
    "storage_gb": 0.73,
    "estimated_storage_cost_usd": 0.227,
    "total_estimated_cost_usd": 0.910,
    "submissions_used": 5,
    "submissions_remaining": 0,
    "cost_per_submission": 0.182
  }
}
```

### Cost Estimation Script

```python
# scripts/estimate_session_cost.py
import json
from datetime import datetime, timedelta

INSTANCE_COSTS = {
    "ml.t3.medium": 0.05,
    "ml.m5.large": 0.115,
    "ml.m5.xlarge": 0.23,
    "ml.m5.2xlarge": 0.46
}

def estimate_cost(state_path):
    with open(state_path) as f:
        state = json.load(f)
    
    cost_tracking = state.get("cost_tracking", {})
    instance_breakdown = cost_tracking.get("instance_breakdown", {})
    
    compute_cost = sum(
        hours * INSTANCE_COSTS[instance_type]
        for instance_type, hours in instance_breakdown.items()
    )
    
    storage_gb = cost_tracking.get("storage_gb", 0)
    storage_cost = storage_gb * 0.30  # EFS Standard
    
    total_cost = compute_cost + storage_cost
    
    print(f"Compute: ${compute_cost:.3f}")
    print(f"Storage: ${storage_cost:.3f}")
    print(f"Total:   ${total_cost:.3f}")
    
    return total_cost
```

### CloudWatch Cost Metrics

```python
import boto3

cloudwatch = boto3.client('cloudwatch')

# Log cost metrics
cloudwatch.put_metric_data(
    Namespace='Zindian/Cost',
    MetricData=[
        {
            'MetricName': 'ComputeCost',
            'Value': 0.683,
            'Unit': 'None',
            'Dimensions': [
                {'Name': 'Competition', 'Value': competition_id},
                {'Name': 'Phase', 'Value': 'Phase3'}
            ]
        },
        {
            'MetricName': 'StorageCost',
            'Value': 0.227,
            'Unit': 'None',
            'Dimensions': [
                {'Name': 'Competition', 'Value': competition_id}
            ]
        }
    ]
)
```

---

## Storage Optimization Strategies

### Strategy 1: Compress Before Archive

```bash
# Compress data directory before S3 upload
tar -czf data.tar.gz competitions/<slug>/data/

# Typical compression ratio: 40-60%
# 500 MB → 300 MB (40% savings)
```

### Strategy 2: Exclude Temporary Files

```bash
# .gitignore-style exclusions
aws s3 sync competitions/<slug>/ s3://my-archives/zindian/<slug>/ \
    --exclude "*.pyc" \
    --exclude "__pycache__/*" \
    --exclude ".ipynb_checkpoints/*" \
    --exclude "*.log" \
    --exclude "data/processed/*"  # Regenerable
```

### Strategy 3: Deduplicate Across Competitions

```bash
# Use S3 object versioning and deduplication
# Identical files (e.g., requirements.txt) stored once

aws s3api put-bucket-versioning \
    --bucket my-archives \
    --versioning-configuration Status=Enabled
```

### Strategy 4: Selective Archival

```python
# Archive only essential files
ESSENTIAL_FILES = [
    "challenge_config.json",
    "SKILL_STATE.json",
    "reports/governance_report.json",
    "reports/reproducibility_audit.json"
]

# Skip regenerable artifacts
SKIP_PATTERNS = [
    "data/processed/*",
    "notebooks/*.ipynb",
    "*.pyc"
]
```

---

## Cost Monitoring and Alerts

### CloudWatch Alarms

```python
import boto3

cloudwatch = boto3.client('cloudwatch')

# Alert if daily cost > $5
cloudwatch.put_metric_alarm(
    AlarmName='ZindianDailyCostExceeded',
    ComparisonOperator='GreaterThanThreshold',
    EvaluationPeriods=1,
    MetricName='EstimatedCharges',
    Namespace='AWS/Billing',
    Period=86400,  # 1 day
    Statistic='Maximum',
    Threshold=5.0,
    ActionsEnabled=True,
    AlarmActions=['arn:aws:sns:us-east-1:123456789012:billing-alerts']
)
```

### Budget Alerts

```bash
# AWS Budgets (via CLI)
aws budgets create-budget \
    --account-id 123456789012 \
    --budget file://budget-config.json \
    --notifications-with-subscribers file://notifications.json
```

**budget-config.json:**
```json
{
  "BudgetName": "ZindianMonthlyBudget",
  "BudgetLimit": {
    "Amount": "10.0",
    "Unit": "USD"
  },
  "TimeUnit": "MONTHLY",
  "BudgetType": "COST"
}
```

---

## Cost Optimization Checklist

### ✅ DO

- Use lifecycle configs for auto-shutdown (saves $3.68/day)
- Right-size instances per phase (saves 50-70% compute)
- Compress data before S3 upload (saves 40-60% storage)
- Archive to S3 Glacier after 30 days (saves 98.7% storage)
- Enable EFS Infrequent Access (saves 91.7% on inactive data)
- Track costs in `SKILL_STATE.json` per competition
- Set CloudWatch alarms for cost anomalies
- Review AWS Cost Explorer monthly

### ❌ DON'T

- Leave instances running overnight (wastes $1.84/night)
- Use ml.m5.xlarge for Phase 0-1 (2.3× overspend)
- Store large datasets in EFS long-term (13× more expensive than S3)
- Skip compression before archival (wastes 40-60% storage)
- Ignore budget warnings in skill_16
- Archive without cleanup (wastes EFS storage)
- Use Expedited Glacier retrieval unless urgent (10× cost)

---

## Cost Comparison: Optimized vs Unoptimized

### Unoptimized Workflow

```
Always-on ml.m5.xlarge:        $165.60/month
EFS storage (5 GB):              $1.50/month
No archival strategy:            $1.50/month (ongoing)
No auto-shutdown:               +$55.20/month (idle time)
─────────────────────────────────────────────
Total:                          $223.80/month
```

### Optimized Workflow (5 competitions/month)

```
Right-sized instances:           $3.42/month
EFS storage (active only):       $0.66/month
S3 Standard (30 days):           $0.04/month
S3 Glacier (long-term):          $0.01/month
Auto-shutdown enabled:           $0.00 (no idle waste)
─────────────────────────────────────────────
Total:                           $4.13/month
Savings:                        $219.67/month (98.2%)
```

---

## Troubleshooting

### Issue: Unexpected high EFS costs

```bash
# Diagnose: Check EFS usage
df -h /home/sagemaker-user/shared

# Solution: Archive old competitions
python scripts/cleanup_efs.py --days-after-close 7

# Verify cleanup
du -sh /home/sagemaker-user/shared/zindian-orchestrator/competitions/*
```

### Issue: S3 retrieval costs higher than expected

```bash
# Cause: Frequent Glacier retrievals
# Solution: Use S3 Standard for recent archives (30 days)

# Check lifecycle policy
aws s3api get-bucket-lifecycle-configuration --bucket my-archives

# Adjust transition period
# Standard → Glacier: 30 days (not 7 days)
```

### Issue: Instance running during idle hours

```bash
# Cause: Lifecycle config not applied
# Solution: Verify and reapply

# Check current lifecycle configs
aws sagemaker list-notebook-instance-lifecycle-configs

# Apply to instance
aws sagemaker update-notebook-instance \
    --notebook-instance-name my-instance \
    --lifecycle-config-name auto-shutdown-config
```

---

## References

- [Cost Optimization Guide](cost_optimization.md) — Detailed cost strategies
- [SageMaker Workspace Strategy](sagemaker_workspace_strategy.md) — Instance management
- [AWS Pricing Calculator](https://calculator.aws) — Cost estimation
- [AWS Cost Explorer](https://console.aws.amazon.com/cost-management/home) — Usage analysis

---

**Last Updated:** June 2026  
**Maintained by:** Orioki — MCS 4.2, JKUAT
