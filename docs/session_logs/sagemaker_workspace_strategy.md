# SageMaker-First Workspace Strategy

**Target Environment:** AWS SageMaker Studio  
**Author:** Orioki — MCS 4.2, JKUAT  
**Last Updated:** June 2026

---

## Overview

Zindian Orchestrator is architected with a **SageMaker-first design philosophy** that leverages AWS SageMaker's managed infrastructure while maintaining local-first execution patterns for cost optimization and reproducibility.

---

## Core Principles

### 1. Local-First Execution

All pipeline logic executes within the SageMaker instance environment:
- No external API calls except Zindi submissions
- No SageMaker API dependencies (CreateProcessingJob, etc.)
- Zero AWS service quota consumption
- Full reproducibility within instance boundaries

### 2. Managed Infrastructure Benefits

Leverage SageMaker's managed services without coupling:
- EFS for persistent storage across sessions
- IAM roles for secure credential management
- Lifecycle configurations for auto-shutdown
- Instance type flexibility per phase

### 3. Cost-Optimized Compute

Right-size instances per pipeline phase:
- **Phase 0-1:** ml.t3.medium ($0.05/hr) — config and intake
- **Phase 2:** ml.m5.large ($0.115/hr) — anchor baseline
- **Phase 3:** ml.m5.xlarge ($0.23/hr) — feature variants
- **Phase 4:** ml.m5.large ($0.115/hr) — calibration
- **Phase 5:** ml.t3.medium ($0.05/hr) — governance

---

## Workspace Structure

```
/home/sagemaker-user/shared/
└── zindian-orchestrator/
    ├── competitions/              ← Per-competition workspaces
    │   └── <slug>/
    │       ├── challenge_config.json
    │       ├── SKILL_STATE.json
    │       ├── data/
    │       ├── notebooks/
    │       └── reports/
    ├── competition_history/       ← Cross-competition learning
    │   └── history_log.jsonl
    ├── zindian/                   ← Framework code
    ├── scripts/                   ← Utilities
    └── .venv/                     ← Python environment
```

### Storage Strategy

**EFS (Elastic File System):**
- Persistent across instance restarts
- Shared across multiple instances
- Automatic backups
- Cost: $0.30/GB-month (Standard)

**S3 Integration:**
- Archive completed competitions
- Store large datasets
- Backup critical state files
- Cost: $0.023/GB-month

---

## Session Management

### Starting a Session

```bash
# 1. Activate environment
source /home/sagemaker-user/shared/zindian-orchestrator/.venv/bin/activate

# 2. Verify instance type
cat /opt/ml/metadata/resource-metadata.json | jq '.ResourceName'

# 3. Run preflight checks
python scripts/preflight_enforce.py

# 4. Check competition state
python scripts/verify_competition_state.py
```

### Ending a Session

```bash
# 1. Verify state integrity
python scripts/verify_competition_state.py

# 2. Backup to S3 (optional)
aws s3 sync competitions/<slug>/ s3://my-bucket/competitions/<slug>/

# 3. Shut down instance (Studio UI)
# File → Shut Down → Shut Down All
```

---

## Instance Type Selection

### Decision Matrix

| Phase | Workload | CPU | Memory | Instance | Cost/hr |
|-------|----------|-----|--------|----------|---------|
| 0-1 | Config, intake | Low | 4GB | ml.t3.medium | $0.05 |
| 2 | Anchor training | Medium | 8GB | ml.m5.large | $0.115 |
| 3 | Feature variants | High | 16GB | ml.m5.xlarge | $0.23 |
| 4 | Calibration | Medium | 8GB | ml.m5.large | $0.115 |
| 5 | Governance | Low | 4GB | ml.t3.medium | $0.05 |

### Switching Instances

**Option 1: Manual (Recommended)**
```bash
# 1. Save state
python scripts/verify_competition_state.py

# 2. Shut down current instance
# Studio UI: File → Shut Down

# 3. Start new instance with desired type
# Studio UI: Select instance type → Launch
```

**Option 2: Processing Jobs (Advanced)**
```python
from sagemaker.processing import ScriptProcessor

processor = ScriptProcessor(
    role=role,
    image_uri='<container>',
    instance_type='ml.m5.xlarge',
    instance_count=1,
    max_runtime_in_seconds=7200
)

processor.run(
    code='zindian/skills/skill_07_features.py',
    arguments=['--config', 'challenge_config.json']
)
```

---

## Lifecycle Configuration

### Auto-Shutdown Script

Prevent idle instance costs:

```bash
#!/bin/bash
# Save as: lifecycle-config-auto-shutdown.sh

IDLE_TIME=1800  # 30 minutes

while true; do
    LAST_ACTIVITY=$(stat -c %Y /home/sagemaker-user/.jupyter/lab/workspaces/*.jupyterlab-workspace 2>/dev/null | sort -n | tail -1)
    CURRENT_TIME=$(date +%s)
    IDLE_DURATION=$((CURRENT_TIME - LAST_ACTIVITY))
    
    if [ $IDLE_DURATION -gt $IDLE_TIME ]; then
        echo "Idle for $IDLE_DURATION seconds. Shutting down..."
        sudo shutdown -h now
    fi
    
    sleep 300  # Check every 5 minutes
done
```

### Startup Script

Initialize environment on instance start:

```bash
#!/bin/bash
# Save as: lifecycle-config-startup.sh

cd /home/sagemaker-user/shared/zindian-orchestrator

# Activate environment
source .venv/bin/activate

# Update dependencies (if needed)
pip install -r requirements.txt --quiet

# Run preflight checks
python scripts/preflight_enforce.py --mode ENFORCE

echo "Zindian Orchestrator ready."
```

---

## IAM Role Configuration

### Required Permissions

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::my-competitions-bucket/*",
        "arn:aws:s3:::my-competitions-bucket"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:*"
    }
  ]
}
```

### Security Best Practices

- Store Zindi credentials in AWS Secrets Manager
- Use IAM roles instead of access keys
- Enable CloudTrail for audit logging
- Restrict S3 bucket access to specific prefixes

---

## Monitoring and Observability

### Resource Monitoring

```bash
# CPU and memory usage
htop

# Disk usage
df -h /home/sagemaker-user/shared

# Network activity
iftop

# Process monitoring
ps aux | grep python
```

### Cost Tracking

```bash
# Check current instance type and cost
cat /opt/ml/metadata/resource-metadata.json | jq '.ResourceName'

# Estimate session cost
python scripts/estimate_session_cost.py

# View cost tracking in state
jq '.cost_tracking' competitions/<slug>/SKILL_STATE.json
```

### CloudWatch Integration

```python
import boto3
from datetime import datetime, timedelta

cloudwatch = boto3.client('cloudwatch')

# Log custom metrics
cloudwatch.put_metric_data(
    Namespace='Zindian/Orchestrator',
    MetricData=[
        {
            'MetricName': 'PhaseCompletionTime',
            'Value': 120.5,
            'Unit': 'Seconds',
            'Timestamp': datetime.utcnow(),
            'Dimensions': [
                {'Name': 'Phase', 'Value': 'Phase3'},
                {'Name': 'Competition', 'Value': competition_id}
            ]
        }
    ]
)
```

---

## Backup and Recovery

### Automated Backups

```bash
#!/bin/bash
# Save as: scripts/backup_competition.sh

COMPETITION_SLUG=$1
BACKUP_BUCKET="s3://my-backups/zindian"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Backup competition workspace
aws s3 sync \
    competitions/$COMPETITION_SLUG/ \
    $BACKUP_BUCKET/$COMPETITION_SLUG/$TIMESTAMP/ \
    --exclude "*.pyc" \
    --exclude "__pycache__/*"

echo "Backup completed: $BACKUP_BUCKET/$COMPETITION_SLUG/$TIMESTAMP/"
```

### Recovery Procedure

```bash
# 1. List available backups
aws s3 ls s3://my-backups/zindian/<slug>/

# 2. Restore from backup
aws s3 sync \
    s3://my-backups/zindian/<slug>/<timestamp>/ \
    competitions/<slug>/

# 3. Verify integrity
python scripts/verify_competition_state.py

# 4. Resume pipeline
python -m zindian.orchestrator --resume
```

---

## Multi-Competition Management

### Parallel Competitions

```bash
# Competition 1: Active development
cd competitions/competition-a/
python -m zindian.orchestrator

# Competition 2: Monitoring only
cd competitions/competition-b/
python -m zindian.skills.skill_00_zindi_monitor
```

### Resource Allocation

- **Single instance:** Run one competition at a time
- **Multiple instances:** Isolate competitions per instance
- **Processing Jobs:** Run heavy phases in parallel

---

## Troubleshooting

### Common Issues

**Issue: Out of memory during Phase 3**
```bash
# Solution: Switch to larger instance
# 1. Save state
# 2. Shut down instance
# 3. Launch ml.m5.2xlarge (32GB RAM)
```

**Issue: EFS storage full**
```bash
# Solution: Archive old competitions
python scripts/archive_competition.py <slug>
aws s3 sync competitions/<slug>/ s3://archives/<slug>/
rm -rf competitions/<slug>/
```

**Issue: Instance won't start**
```bash
# Solution: Check service quotas
aws service-quotas get-service-quota \
    --service-code sagemaker \
    --quota-code L-<quota-code>
```

---

## Best Practices

### ✅ DO

- Use lifecycle configs for auto-shutdown
- Right-size instances per phase
- Backup state files to S3 regularly
- Monitor resource usage during training
- Use IAM roles instead of access keys
- Archive completed competitions

### ❌ DON'T

- Leave instances running overnight
- Use oversized instances for light work
- Store large datasets in EFS long-term
- Hardcode credentials in code
- Run multiple competitions on one instance
- Skip preflight checks

---

## Cost Optimization Summary

**Typical competition cost breakdown:**
- Compute: $0.60 (3.5 hours optimized)
- Storage: $0.75 (2.5GB EFS for 1 month)
- **Total: $1.35 per competition**

**vs. Always-on ml.m5.xlarge:**
- $0.23/hr × 24 × 30 = $165.60/month
- **Savings: 99.2%**

---

## References

- [AWS SageMaker Pricing](https://aws.amazon.com/sagemaker/pricing/)
- [EFS Pricing](https://aws.amazon.com/efs/pricing/)
- [SageMaker Best Practices](https://docs.aws.amazon.com/sagemaker/latest/dg/best-practices.html)
- [Cost Optimization Guide](cost_optimization.md)

---

**Last Updated:** June 2026  
**Maintained by:** Orioki — MCS 4.2, JKUAT
