# Cost-Efficiency Decision Framework

**Integration with Zindian Orchestrator**  
**Author:** Orioki — MCS 4.2, JKUAT  
**Last updated:** June 2026

---

## Overview

This framework integrates cost-per-improvement analysis, resource utilization monitoring, and budget-aware gating into the Zindian Orchestrator workflow. Every phase decision is evaluated through **five critical questions** before proceeding.

---

## The Five Questions Framework

Before every experiment, skill execution, or submission, ask:

### 1. Does it improve the metric?

**What to measure:**
- OOF score improvement
- Secondary metrics (MAE, MAPE, R²)
- Fold score consistency

**Decision rule:**
```python
if oof_improvement <= 0:
    gate_decision = "REJECT"
    reason = "No metric improvement"
```

---

### 2. Is the improvement stable?

**What to measure:**
- Fold score variance (ddof=1)
- Effective variance threshold (scale-normalized for regression)

**Decision rule:**
```python
if fold_score_variance > effective_variance_threshold:
    gate_decision = "REJECT"
    reason = "High variance — unstable model"
```

---

### 3. How much compute did it consume?

**What to measure:**
- Phase duration (minutes)
- Instance type and hourly rate
- Estimated cost

**Decision rule:**
```python
cost = duration_hours * instance_hourly_rate
if cost > phase_budget:
    warning = f"Phase exceeded budget: ${cost:.2f} > ${phase_budget}"
```

---

### 4. Is it worth a submission?

**What to measure:**
- Cost per improvement
- Submission budget remaining
- OOF-to-LB correlation history

**Decision rule:**
```python
cost_per_point = compute_cost / abs(oof_improvement)
if cost_per_point > max_cost_per_improvement:
    gate_decision = "REJECT"
    reason = f"Cost per improvement too high: ${cost_per_point:.2f}"
```

---

### 5. What did I learn?

**What to capture:**
- Feature engineering insights
- Model behavior patterns
- Leakage detection signals
- Calibration effects

**Decision rule:**
```python
if learning_value == "high":
    # Worth running even if not final winner
    # Feeds into skill_20 hypothesis validation
    log_to_sidecar(experiment_insights)
```

---

## Phase-Specific Cost Strategies

### Phase 0-1: Setup & Intake

| Metric | Target | Action |
|--------|--------|--------|
| **Instance** | ml.t3.medium | Minimize — no training |
| **Duration** | 15 min | Fast fingerprinting |
| **Cost priority** | Minimize | $0.01 target |
| **Key decision** | Config completeness | Gate: All fields populated |

**Cost optimization:**
- Use smallest instance
- No external API calls except Zindi metadata
- DuckDB ledger (local, no S3)

---

### Phase 2: Anchor Baseline

| Metric | Target | Action |
|--------|--------|--------|
| **Instance** | ml.m5.large | Moderate compute |
| **Duration** | 30 min | Single model training |
| **Cost priority** | Moderate | $0.06 target |
| **Key decision** | OOF score | Gate 1: Human approval |

**Cost optimization:**
- Train single anchor model
- No hyperparameter search
- Use config seed (reproducible)
- Track OOF score as baseline

**Decision framework:**
```python
# After anchor training
anchor_oof = 0.842
duration_minutes = 30
cost = 0.06

# Human Gate 1 decision
if anchor_oof < expected_minimum:
    decision = "[B] REJECT — regenerate"
elif cv_strategy_suspicious:
    decision = "[D] CHALLENGE CV STRATEGY"
else:
    decision = "[A] APPROVE"
```

---

### Phase 3: Feature Engineering

| Metric | Target | Action |
|--------|--------|--------|
| **Instance** | ml.m5.xlarge | Heavy compute |
| **Duration** | 2 hours | Multiple variants |
| **Cost priority** | **Highest** | $0.46 target |
| **Key decision** | Cost per improvement | Gate: skill_11 conditions |

**Cost optimization:**
- Batch all variants in single session
- Use cost_per_improvement metric
- Gate aggressively (5 conditions)
- Track resource utilization

**Decision framework:**
```python
# After each variant
baseline = 0.842
variant_oof = 0.850
improvement = 0.008
duration_minutes = 45
cost = 0.17

cost_per_point = cost / improvement  # $21.25 per point

if cost_per_point < max_cost_per_improvement:
    decision = "PROMOTE"
else:
    decision = "REJECT — cost too high"
    trigger_skill_20_analysis()
```

**Example comparison:**

| Variant | OOF | Improvement | Cost | Cost/Point | Decision |
|---------|-----|-------------|------|------------|----------|
| A | 0.850 | 0.008 | $0.17 | $21.25 | ✅ PROMOTE |
| B | 0.843 | 0.001 | $0.23 | $230.00 | ❌ REJECT |
| C | 0.855 | 0.013 | $0.29 | $22.31 | ✅ PROMOTE |

---

### Phase 4: Calibration

| Metric | Target | Action |
|--------|--------|--------|
| **Instance** | ml.m5.large | Moderate compute |
| **Duration** | 30 min | Calibration only |
| **Cost priority** | Moderate | $0.06 target |
| **Key decision** | Variance + MAE/MAPE | Gate: Secondary metrics |

**Cost optimization:**
- Use identical CV folds (no re-split)
- Classification only (skip for regression)
- Track secondary metrics

---

### Phase 5: Final Selection

| Metric | Target | Action |
|--------|--------|--------|
| **Instance** | ml.t3.medium | Minimal compute |
| **Duration** | 10 min | Selection only |
| **Cost priority** | Minimize | $0.01 target |
| **Key decision** | Submission quality | Gate 5: Human selection |

**Cost optimization:**
- No training (selection from existing)
- Validate format locally
- Human Gate 5 approval

---

## Resource Utilization Monitoring

### CPU Utilization Thresholds

```python
if cpu_utilization < 0.15:
    warning = "Instance oversized"
    recommendation = f"Downsize: {current_instance} → {smaller_instance}"
    # Example: ml.m5.xlarge → ml.m5.large (50% cost savings)
```

### Memory Utilization Thresholds

```python
if memory_utilization > 0.90:
    warning = "Memory bottleneck"
    recommendation = f"Upsize: {current_instance} → {larger_instance}"
    # Example: ml.m5.large → ml.m5.xlarge (prevent OOM crashes)
```

### Monitoring Commands

```python
# In orchestrator integration
from zindian.cost_monitor import CostMonitor

monitor = CostMonitor(competition_id="ey-frogs")

# Start phase
monitor.start_phase(
    phase_name="phase_3a",
    instance_type="ml.m5.xlarge",
    baseline_score=0.842,
)

# ... run skill_07, skill_08 ...

# End phase
phase_metrics = monitor.end_phase(
    final_score=0.850,
    cpu_util=0.45,  # 45% CPU
    memory_util=0.72,  # 72% memory
)

# Analyze efficiency
analysis = monitor.analyze_efficiency(phase_metrics)
print(analysis["recommendations"])
# Output: ["CPU utilization acceptable", "Memory utilization acceptable"]
```

---

## Budget Strategy

### Early Competition (Days 1-3)

**Strategy:** Spend budget slowly

```python
submissions_per_day = 1
max_cost_per_improvement = 50.0  # Strict threshold

# Preserve flexibility for later phases
```

**Rationale:**
- Competition rules may change
- Data patches may occur
- Better features discovered later

---

### Mid Competition (Days 4-7)

**Strategy:** Moderate spending

```python
submissions_per_day = 1-2
max_cost_per_improvement = 75.0  # Relaxed threshold

# Focus on validated improvements
```

**Rationale:**
- Anchor baseline established
- Feature engineering complete
- OOF-to-LB correlation known

---

### Final Days (Days 8-10)

**Strategy:** Spend remaining quota

```python
submissions_remaining = budget_total - submissions_used
max_cost_per_improvement = 100.0  # Most relaxed

# Submit only:
# - Ensemble candidates
# - Calibrated models
# - Top OOF performers
```

**Rationale:**
- No future opportunity to use budget
- Final ensemble selection
- Private LB optimization

---

## Integration with skill_11_gate

### Enhanced Gate Condition 6 (Cost-Aware)

```python
def skill_11_gate_with_cost(
    oof_score: float,
    baseline: float,
    fold_variance: float,
    compute_cost: float,
    effective_gate_margin: float,
    effective_variance_threshold: float,
    max_cost_per_improvement: float,
) -> tuple[bool, str]:
    """Enhanced gate with cost-per-improvement check."""
    
    # Existing conditions 1-5
    if branch in leaked_features:
        return False, "Condition 1: SHAP leak detected"
    
    if fold_variance > effective_variance_threshold:
        return False, "Condition 2: High variance"
    
    improvement = abs(oof_score - baseline)
    if improvement <= effective_gate_margin:
        return False, "Condition 3: Insufficient improvement"
    
    if not shap_audit_passed:
        return False, "Condition 4: SHAP audit failed"
    
    if not human_gate_2_approved:
        return False, "Condition 5: Human gate not approved"
    
    # NEW: Condition 6 — Cost-per-improvement
    cost_per_point = compute_cost / improvement
    if cost_per_point > max_cost_per_improvement:
        return False, (
            f"Condition 6: Cost per improvement too high "
            f"(${cost_per_point:.2f} > ${max_cost_per_improvement})"
        )
    
    return True, "All 6 conditions passed"
```

---

## Cost Tracking in SKILL_STATE.json

### Schema Extension

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
    "cost_per_submission": 0.27,
    "phase_costs": {
      "phase_1": 0.01,
      "phase_2": 0.06,
      "phase_3": 0.46,
      "phase_4": 0.06,
      "phase_5": 0.01
    },
    "cost_per_improvement_history": [
      {
        "branch": "variant-01",
        "improvement": 0.008,
        "cost": 0.17,
        "cost_per_point": 21.25,
        "decision": "PROMOTED"
      },
      {
        "branch": "variant-02",
        "improvement": 0.001,
        "cost": 0.23,
        "cost_per_point": 230.00,
        "decision": "REJECTED"
      }
    ]
  }
}
```

---

## Orchestrator Integration

### Modified orchestrator.py

```python
from zindian.cost_monitor import CostMonitor, get_recommended_instance

def run_phase_with_monitoring(
    phase: int,
    competition_id: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run phase with cost monitoring."""
    
    # Initialize monitor
    monitor = CostMonitor(competition_id=competition_id)
    
    # Get recommended instance
    phase_name = f"phase_{phase}"
    instance_type = get_recommended_instance(phase_name)
    
    # Start monitoring
    baseline_score = kwargs.get("baseline_score")
    monitor.start_phase(
        phase_name=phase_name,
        instance_type=instance_type,
        baseline_score=baseline_score,
    )
    
    # Run phase skills
    results = run_phase(phase, **kwargs)
    
    # End monitoring
    final_score = results.get("final_oof_score")
    phase_metrics = monitor.end_phase(final_score=final_score)
    
    # Analyze efficiency
    analysis = monitor.analyze_efficiency(phase_metrics)
    
    # Surface warnings
    if analysis["warnings"]:
        print(f"\n⚠️  Cost Efficiency Warnings:")
        for warning in analysis["warnings"]:
            print(f"   - {warning}")
        print(f"\n💡 Recommendations:")
        for rec in analysis["recommendations"]:
            print(f"   - {rec}")
    
    # Save tracking
    from zindian.paths import resolve_competition_paths
    paths = resolve_competition_paths()
    monitor.save(paths.reports_dir / "cost_tracking.json")
    
    return {
        **results,
        "cost_analysis": analysis,
    }
```

---

## Example: Full Competition Cost Profile

### Competition: EY Frogs (Classification)

| Phase | Duration | Instance | Cost | OOF Improvement | Cost/Point | Decision |
|-------|----------|----------|------|-----------------|------------|----------|
| 0-1 | 15 min | t3.medium | $0.01 | N/A | N/A | Setup |
| 2 | 30 min | m5.large | $0.06 | 0.842 (baseline) | N/A | Anchor |
| 3 (v1) | 45 min | m5.xlarge | $0.17 | 0.008 | $21.25 | ✅ Promote |
| 3 (v2) | 60 min | m5.xlarge | $0.23 | 0.001 | $230.00 | ❌ Reject |
| 3 (v3) | 50 min | m5.xlarge | $0.19 | 0.013 | $14.62 | ✅ Promote |
| 4 | 30 min | m5.large | $0.06 | 0.002 | $30.00 | Calibrate |
| 5 | 10 min | t3.medium | $0.01 | N/A | N/A | Select |
| **Total** | **4.0 hrs** | | **$0.73** | **0.024** | **$30.42** | **5 submissions** |

**Efficiency metrics:**
- Total cost: $0.73
- Total improvement: 0.024 (84.2% → 86.6%)
- Average cost per point: $30.42
- Submissions used: 5 / 5
- Cost per submission: $0.15

**vs. Unoptimized workflow:**
- Always-on ml.m5.xlarge: $165.60/month
- No gating: 15 submissions (10 rejected)
- Total cost: $3.60
- **Savings: 79.7%**

---

## Summary

**Key principles:**

1. **Plan before launch** — Choose instance size per phase
2. **Monitor during training** — Track CPU/memory utilization
3. **Gate before promotion** — Use cost-per-improvement threshold
4. **Check before submission** — Validate multiple metrics
5. **Learn from every experiment** — Feed insights to skill_20

**Integration points:**

- `cost_monitor.py` — Core tracking module
- `orchestrator.py` — Phase-level integration
- `skill_11_gate.py` — Enhanced gate condition 6
- `SKILL_STATE.json` — Cost tracking schema
- `skill_22_audit.py` — Cost efficiency verification

**Expected outcomes:**

- 75-85% cost reduction vs. unoptimized
- 40% fewer wasteful submissions
- Real-time resource utilization feedback
- Historical cost-per-improvement data for threshold tuning

---

**Last Updated:** June 2026  
**Maintained by:** Orioki — MCS 4.2, JKUAT
