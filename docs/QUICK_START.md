# v2.3 Refactor — Quick Start Guide

**Ready to start?** Follow this guide to begin the refactor work.

---

## Before You Start

### 1. Review Documentation
```bash
# Read these in order:
cat docs/REFACTOR_SUMMARY.md          # 5 min — Overview
cat docs/REFACTOR_PLAN_v2.3.md        # 15 min — Detailed plan
cat docs/PROGRESS_TRACKER.md          # 5 min — Checklist
```

### 2. Verify Environment
```bash
# Check Python version
python --version  # Should be 3.9+

# Verify dependencies
pip list | grep -E "lightgbm|pandas|numpy|pytest"

# Run existing tests
pytest tests/ -v --tb=short
```

### 3. Create Working Branch
```bash
git checkout -b refactor/v2.3.1-gaps
git pull origin main
```

---

## Day 1: DRIFT-1 (Hardcoded Targets)

### Step 1: Understand the Problem
```bash
# Read the current implementation
cat zindian/skills/skill_07_features.py | grep -A 5 -B 5 "total_goals\|Target"

# Expected output: Lines 1006-1007 with hardcoded strings
```

### Step 2: Write the Test First
```bash
# Create test file
touch tests/test_a5_compliance.py
```

```python
# tests/test_a5_compliance.py
"""
Test A5 compliance: No hardcoded competition-specific strings.

Verifies that no skill contains hardcoded:
- Column names
- Target names
- Metric names
- Competition identifiers
"""
import pytest
import re
from pathlib import Path

def test_no_hardcoded_targets_in_skill_07():
    """skill_07 must not hardcode target names."""
    skill_path = Path("zindian/skills/skill_07_features.py")
    content = skill_path.read_text()
    
    # Forbidden patterns
    forbidden = [
        r'"total_goals"',
        r'"Target"',
        r'"label"',
        r'"TargetF1"',
    ]
    
    for pattern in forbidden:
        matches = re.findall(pattern, content)
        assert len(matches) == 0, (
            f"Found hardcoded target name {pattern} in skill_07. "
            f"Use config['target_config']['targets'] instead."
        )

def test_no_hardcoded_metrics_in_skills():
    """No skill should hardcode metric names."""
    skill_dir = Path("zindian/skills")
    forbidden = [
        r'"anchor_oof_f1"',
        r'"anchor_oof_auc"',
        r'"anchor_oof_rmse"',
    ]
    
    for skill_file in skill_dir.glob("skill_*.py"):
        content = skill_file.read_text()
        for pattern in forbidden:
            matches = re.findall(pattern, content)
            assert len(matches) == 0, (
                f"Found hardcoded metric {pattern} in {skill_file.name}. "
                f"Use f'anchor_oof_{{metric_key}}' pattern instead."
            )
```

### Step 3: Run Test (Should Fail)
```bash
pytest tests/test_a5_compliance.py -v
# Expected: FAILED — hardcoded strings detected
```

### Step 4: Fix the Code
```bash
# Open skill_07_features.py
vim zindian/skills/skill_07_features.py +1006
```

```python
# BEFORE (Lines 1006-1007):
oof_key = f"branch_{branch_name}_total_goals_oof"
# or
oof_key = f"branch_{branch_name}_Target_oof"

# AFTER:
# Read target names from config
target_names = [t["name"] for t in config["target_config"]["targets"]]

# For single-target competitions
if len(target_names) == 1:
    target_name = target_names[0]
    oof_key = f"branch_{branch_name}_oof"
else:
    # For multi-target, iterate over targets
    for target_name in target_names:
        oof_key = f"branch_{branch_name}_{target_name}_oof"
        # ... rest of logic
```

### Step 5: Run Test (Should Pass)
```bash
pytest tests/test_a5_compliance.py -v
# Expected: PASSED
```

### Step 6: Run Full Test Suite
```bash
pytest tests/ -v --tb=short
# Verify no regressions
```

### Step 7: Commit
```bash
git add zindian/skills/skill_07_features.py tests/test_a5_compliance.py
git commit -m "fix(skill_07): Remove hardcoded target names (DRIFT-1)

- Replace 'total_goals' and 'Target' literals with dynamic resolution
- Read target names from config['target_config']['targets']
- Add test_a5_compliance.py to prevent regressions
- Closes: DRIFT-1 from sot_audit_report.md"
```

### Step 8: Update Progress Tracker
```bash
# Mark DRIFT-1 as complete
vim docs/PROGRESS_TRACKER.md
# Change: ☐ DRIFT-1 → ✅ DRIFT-1
```

---

## Day 2-3: GAP-2 (Composite Fold Variance)

### Step 1: Understand Current Implementation
```bash
# Read skill_12_metric.py
cat zindian/skills/skill_12_metric.py | grep -A 20 "fold_score_variance"
```

### Step 2: Write Test First
```bash
touch tests/test_multi_target_composite_variance.py
```

```python
# tests/test_multi_target_composite_variance.py
"""
Test composite fold variance for multi-target competitions.

Verifies that skill_12 computes weighted composite variance
across all targets with proper normalization.
"""
import pytest
import numpy as np
from zindian.skills.skill_12_metric import run

def test_composite_fold_variance_multi_target():
    """Compute weighted composite variance for multi-target."""
    config = {
        "target_config": {
            "targets": [
                {
                    "name": "goals",
                    "task_type": "regression",
                    "metric": "rmse",
                    "weight": 0.6
                },
                {
                    "name": "label",
                    "task_type": "classification",
                    "metric": "f1",
                    "weight": 0.4
                }
            ]
        },
        "cv_strategy": {"n_splits": 5}
    }
    
    state = {
        "branch_test_goals_oof": {
            "fold_scores": [0.5, 0.52, 0.48, 0.51, 0.49]
        },
        "branch_test_label_oof": {
            "fold_scores": [0.8, 0.82, 0.79, 0.81, 0.80]
        },
        "eda": {
            "goals_std": 2.5  # For normalization
        }
    }
    
    result = run(config, state)
    
    # Verify composite variance computed
    assert "composite_fold_score_variance" in result
    assert result["composite_fold_score_variance"] > 0
    
    # Verify variance uses ddof=1
    # Manual calculation:
    # Fold 1: 0.6 * (0.5/2.5) + 0.4 * 0.8 = 0.44
    # ... compute for all folds
    # variance = np.var([...], ddof=1)
```

### Step 3: Implement Function
```bash
vim zindian/skills/skill_12_metric.py
```

```python
def _compute_composite_fold_variance(state: dict, config: dict, branch_name: str) -> float:
    """
    Compute fold score variance for multi-target composite metric.
    
    Reads per-target fold scores, applies weights, computes composite
    variance with ddof=1.
    """
    targets = config["target_config"]["targets"]
    n_folds = config["cv_strategy"]["n_splits"]
    
    composite_fold_scores = []
    for fold_idx in range(n_folds):
        fold_composite = 0.0
        
        for target in targets:
            target_name = target["name"]
            weight = target["weight"]
            metric_key = target["metric"]
            
            # Read per-target fold score
            if len(targets) == 1:
                oof_key = f"branch_{branch_name}_oof"
            else:
                oof_key = f"branch_{branch_name}_{target_name}_oof"
            
            fold_score = state[oof_key]["fold_scores"][fold_idx]
            
            # Normalize regression metrics by target_std
            if target["task_type"] == "regression" and metric_key != "rmsle":
                target_std = state["eda"].get(f"{target_name}_std", 1.0)
                if target_std > 0:
                    fold_score = fold_score / target_std
            
            fold_composite += weight * fold_score
        
        composite_fold_scores.append(fold_composite)
    
    # Compute variance with ddof=1 (unbiased)
    return float(np.var(composite_fold_scores, ddof=1))
```

### Step 4: Integrate into run()
```python
def run(config: dict, state: dict) -> dict:
    # ... existing logic ...
    
    # Check if multi-target
    targets = config.get("target_config", {}).get("targets", [])
    if len(targets) > 1:
        # Multi-target: compute composite variance
        variance = _compute_composite_fold_variance(state, config, branch_name)
        state["composite_fold_score_variance"] = variance
    else:
        # Single-target: existing logic
        # ... existing variance computation ...
    
    return state
```

### Step 5: Test and Commit
```bash
pytest tests/test_multi_target_composite_variance.py -v
pytest tests/ -v --tb=short

git add zindian/skills/skill_12_metric.py tests/test_multi_target_composite_variance.py
git commit -m "feat(skill_12): Implement composite fold variance for multi-target (GAP-2)"
```

---

## Day 4-7: R5 (Carbon Tracking)

### Overview
This is the largest task. Break it into sub-tasks:

1. **Day 4:** Create carbon_tracker.py module
2. **Day 5:** Hook into orchestrator
3. **Day 6:** Update skill_02, instrument skills
4. **Day 7:** Write tests, validate

### Quick Start
```bash
# Create module
touch zindian/carbon_tracker.py

# Create test
touch tests/test_r5_carbon_tracking.py

# Follow detailed implementation in REFACTOR_PLAN_v2.3.md
```

---

## Tips for Success

### 1. Test-First Development
Always write the test before the implementation. This ensures:
- Clear requirements
- Immediate validation
- Regression prevention

### 2. Small Commits
Commit after each logical change:
- Easier to review
- Easier to revert
- Better git history

### 3. Run Tests Frequently
```bash
# After each change
pytest tests/test_<current_feature>.py -v

# Before committing
pytest tests/ -v --tb=short
```

### 4. Update Documentation Concurrently
Don't wait until the end to update docs:
- Update PROGRESS_TRACKER.md after each task
- Update AGENTS.md when adding new patterns
- Update SoT when resolving gaps

### 5. Ask for Help
If you encounter:
- Unclear requirements → Check SoT
- Test failures → Check audit report
- Architecture questions → Check AGENTS.md

---

## Common Issues

### Issue: Test fails with KeyError
**Cause:** State key doesn't exist  
**Fix:** Use safe access pattern from AGENTS.md
```python
# WRONG:
value = state["key"]

# RIGHT:
value = state.get("key", default_value)
```

### Issue: Hardcoded string detected
**Cause:** A5 violation  
**Fix:** Read from config
```python
# WRONG:
target_name = "total_goals"

# RIGHT:
target_name = config["target_config"]["targets"][0]["name"]
```

### Issue: Test passes locally but fails in CI
**Cause:** Environment differences  
**Fix:** Check requirements.txt, Python version

---

## Daily Checklist

Before starting work:
- [ ] Pull latest changes: `git pull origin main`
- [ ] Review progress tracker
- [ ] Read relevant SoT section

After completing work:
- [ ] Run tests: `pytest tests/ -v`
- [ ] Update progress tracker
- [ ] Commit with descriptive message
- [ ] Push to remote: `git push origin refactor/v2.3.1-gaps`

---

## Need Help?

**Documentation:**
- `docs/REFACTOR_PLAN_v2.3.md` — Detailed implementation
- `docs/REFACTOR_SUMMARY.md` — Executive overview
- `docs/source_of_truth.md` — Architectural authority
- `AGENTS.md` — Operational guidelines

**Communication:**
- Daily standup: Progress + blockers
- Weekly review: Phase completion
- Slack: #zindian-refactor

---

**Ready to start?** Begin with DRIFT-1 (Day 1 section above).

**Last Updated:** June 26, 2026
