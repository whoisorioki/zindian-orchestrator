# Multi-Target Implementation Guide

**Status:** CLEARED FOR IMPLEMENTATION  
**Foundation:** 17/17 tests passing, SoT v2.2.1 compliant  
**Target:** Resolve Critical Gaps 1 & 2 (A11/A12)

---

## Implementation Sequence

### Critical Gap 1: Multi-Target Config & OOF Loops
**Files:** skill_02, skill_08, skill_10, skill_11  
**Objective:** Enable simultaneous evaluation of multiple targets (e.g., FIFA World Cup: 60% RMSE Goals + 40% F1 Stage)

### Critical Gap 2: Pseudo-Label Recombination Policy
**File:** skill_21  
**Objective:** Implement A12 mandatory recombination policy for mixed-task competitions

---

## Prompt 1: skill_02 Multi-Target Config Intake

**Target File:** `/zindian/skills/skill_02_intake.py`

**Implementation Prompt:**
```
Refactor skill_02_intake.py to dynamically handle multi-target Zindi competitions (Assumption A11).

1. Read the competition submission format file. If it contains more than one target column, 
   construct a target_config dictionary to write into challenge_config.json.

2. For each target, dynamically assign its name, task_type, metric, weight, and 
   target_domain_bounds.

3. Implement the mandatory A12 Rule: If target_config has more than one target AND at least 
   one is classification, inject the field "pseudo_label_recombination_policy".

4. Ensure backward compatibility: If the submission format has exactly one target, skip 
   building target_config and fall back to the legacy top-level task_type and metric mapping.
```

**Expected Output Structure:**
```json
{
  "target_config": {
    "targets": [
      {
        "name": "goals_scored",
        "task_type": "regression",
        "metric": "root_mean_squared_error",
        "metric_direction": "minimize",
        "weight": 0.6,
        "target_domain_bounds": {"min": 0, "max": 20}
      },
      {
        "name": "stage_reached",
        "task_type": "classification",
        "metric": "f1",
        "metric_direction": "maximize",
        "weight": 0.4,
        "target_domain_bounds": null
      }
    ],
    "composite_direction": "minimize_composite_distance",
    "pseudo_label_recombination_policy": "freeze_unaugmented_targets_at_original"
  }
}
```

**Validation:**
- Single-target competitions: target_config absent or has exactly one entry
- Multi-target competitions: target_config.targets has 2+ entries
- Mixed-task competitions: pseudo_label_recombination_policy present

---

## Prompt 2: skill_08 Target Loops & Transformation Lifecycle

**Target File:** `/zindian/skills/skill_08_anchor.py`

**Implementation Prompt:**
```
Refactor skill_08_anchor.py to support independent multi-target training loops and composite scoring.

1. Check if target_config exists in challenge_config.json. If yes, loop over 
   target_config["targets"]. For each target, instantiate a separate LightGBM model.

2. Implement the Regression Target Transformation Lifecycle: Before training, map the target 
   metric. If rmsle, apply ln(y + 1).

3. After generating out-of-fold (OOF) predictions, execute the inverse mapping. If rmsle, 
   apply exp(y') - 1. If root_mean_squared_error or mean_absolute_error, clip the predictions 
   using the target's specific target_domain_bounds.

4. Compute raw_score independently for each target.

5. Aggregate the scores into a single composite_score (sum(weighted_distances)), while 
   preserving a dictionary of anchor_oof_score_per_target for the Human Gate 1 operator prompt.
```

**Implementation Pattern:**
```python
def run():
    config = ChallengeConfig.load()
    targets = config.get("target_config", {}).get("targets")
    
    if not targets:
        # Single-target path (unchanged)
        return _run_single_target()
    
    # Multi-target path
    per_target_scores = {}
    
    for target_spec in targets:
        # 1. Load target column
        y = train_df[target_spec["name"]]
        
        # 2. Apply transformation lifecycle
        if target_spec["metric"] == "rmsle":
            y_transformed = np.log1p(y)
        else:
            y_transformed = y
        
        # 3. Train model per target
        model = LGBMRegressor() if target_spec["task_type"] == "regression" else LGBMClassifier()
        oof_preds, oof_score = _train_cv(model, X, y_transformed, cv_strategy)
        
        # 4. Inverse transformation
        if target_spec["metric"] == "rmsle":
            oof_preds = np.expm1(oof_preds)
        
        # 5. Clip to domain bounds
        if target_spec["target_domain_bounds"]:
            oof_preds = np.clip(
                oof_preds,
                target_spec["target_domain_bounds"]["min"],
                target_spec["target_domain_bounds"]["max"]
            )
        
        # 6. Compute raw score
        raw_score = compute_metric(y, oof_preds, target_spec["metric"])
        per_target_scores[target_spec["name"]] = raw_score
        
        # 7. Write per-target OOF
        write_oof_record(
            branch_name="anchor",
            target_name=target_spec["name"],
            scores=oof_score,
            cv_strategy_id=cv_strategy["selection_reason"],
            seed=config["reproducibility"]["seed"]
        )
    
    # 8. Compute composite score
    composite = compute_composite_score(targets, per_target_scores)
    
    # 9. Write to state
    SKILL_STATE["anchor_oof_score"] = composite
    SKILL_STATE["anchor_oof_score_per_target"] = per_target_scores
    
    return {"status": "OK", "composite_score": composite}
```

**Composite Score Formula:**
```python
def compute_composite_score(targets, per_target_scores):
    weighted_distances = []
    
    for target_spec in targets:
        raw_score = per_target_scores[target_spec["name"]]
        
        if target_spec["task_type"] == "regression":
            if target_spec["metric"] == "rmsle":
                normalized_distance = raw_score  # Already dimensionless
            else:
                target_std = SKILL_STATE["eda"][f"{target_spec['name']}_std"]
                normalized_distance = abs(raw_score) / target_std
        else:  # classification
            normalized_distance = 1.0 - raw_score
        
        weighted_distances.append(normalized_distance * target_spec["weight"])
    
    return sum(weighted_distances)
```

---

## Prompt 3: skill_10 Per-Target SHAP & skill_11 Leak Gate

**Target Files:** `/zindian/skills/skill_10_shap.py`, `/zindian/skills/skill_11_gate.py`

**Implementation Prompt:**
```
Refactor skill_10_shap.py and the promotion logic in skill_11_gate.py for multi-target compliance.

1. In skill_10, if multi-target is active, run the SHAP computation contract separately for 
   each target's OOF predictions. Save the results under the dynamic key 
   leaked_features_{target_name}.

2. In skill_11, update Promotion Condition 1 (the SHAP Leak Gate). The logic must dictate 
   that a branch must be absent from EVERY target's leaked-features list. Leakage on any 
   single target must trigger a hard block on the promotion of the entire branch.
```

**skill_10 Implementation:**
```python
def run():
    config = ChallengeConfig.load()
    targets = config.get("target_config", {}).get("targets")
    
    if not targets:
        # Single-target path (unchanged)
        return _run_single_target_shap()
    
    # Multi-target path
    for target_spec in targets:
        # Load per-target OOF predictions
        oof_key = f"branch_{branch_name}_{target_spec['name']}_oof"
        oof_data = SKILL_STATE[oof_key]
        
        # Compute SHAP per target
        shap_values = compute_shap_per_fold(model, X, oof_data)
        
        # Leak detection per target
        leaked = detect_leakage(shap_values, threshold=config["shap_leak_threshold"])
        
        # Write per-target leak list
        SKILL_STATE[f"leaked_features_{target_spec['name']}"] = leaked
    
    return {"status": "OK"}
```

**skill_11 Gate Condition 1 Update:**
```python
def check_promotion_condition_1(branch_name):
    """Branch must be absent from EVERY target's leaked-features list"""
    config = ChallengeConfig.load()
    targets = config.get("target_config", {}).get("targets")
    
    if not targets:
        # Single-target path
        leaked = SKILL_STATE.get("leaked_features", [])
        return branch_name not in leaked
    
    # Multi-target path: check ALL targets
    for target_spec in targets:
        leaked_key = f"leaked_features_{target_spec['name']}"
        leaked = SKILL_STATE.get(leaked_key, [])
        
        if branch_name in leaked:
            return False  # Blocked: leakage on this target
    
    return True  # Passed: absent from all leak lists
```

---

## Prompt 4: skill_21 Pseudo-Label Recombination Policy (Gap 2)

**Target File:** `/zindian/skills/skill_21_pseudo_label.py`

**Implementation Prompt:**
```
Refactor skill_21_pseudo_label.py to resolve Critical Gap 2 by enforcing the A12 Recombination Policy.

1. Immediately after the pseudo-label retraining loop completes (and before evaluating skill_11 
   gates), check target_config for pseudo_label_recombination_policy.

2. If the policy is "freeze_unaugmented_targets_at_original", the composite scorer must use 
   the augmented OOF for classification targets but pull the original OOF for regression targets.

3. If the policy is "block_composite_until_all_targets_augmented_or_none", halt the composite 
   calculation entirely unless every target in the config was successfully augmented.
```

**Implementation Pattern:**
```python
def run():
    # ... existing guard conditions ...
    
    # After retraining loop completes
    config = ChallengeConfig.load()
    targets = config.get("target_config", {}).get("targets")
    
    if not targets:
        # Single-target path (unchanged)
        return _run_single_target_pseudo_label()
    
    # Multi-target path: enforce A12 policy
    policy = config["target_config"]["pseudo_label_recombination_policy"]
    augmented_targets = SKILL_STATE.get("augmented_targets", [])
    
    if policy == "freeze_unaugmented_targets_at_original":
        # Use augmented OOF for classification, original for regression
        composite_scores = {}
        
        for target_spec in targets:
            target_name = target_spec["name"]
            
            if target_spec["task_type"] == "classification" and target_name in augmented_targets:
                # Use augmented OOF
                oof_key = f"branch_{branch_name}_{target_name}_oof_augmented"
            else:
                # Use original OOF (regression or unaugmented classification)
                oof_key = f"branch_{branch_name}_{target_name}_oof"
            
            composite_scores[target_name] = SKILL_STATE[oof_key]["scores"]
        
        # Compute composite with mixed OOF sources
        composite = compute_composite_score(targets, composite_scores)
        SKILL_STATE["anchor_oof_score_augmented"] = composite
    
    elif policy == "block_composite_until_all_targets_augmented_or_none":
        # All-or-nothing policy
        all_augmented = all(t["name"] in augmented_targets for t in targets)
        none_augmented = len(augmented_targets) == 0
        
        if not (all_augmented or none_augmented):
            # Partial augmentation: BLOCK composite calculation
            SKILL_STATE["pseudo_label_result"]["execution_failure_reason"] = (
                "partial_augmentation_blocked_by_policy"
            )
            return {"status": "BLOCKED", "reason": "A12 policy violation"}
        
        # Proceed with composite (all augmented or none augmented)
        if all_augmented:
            # Use all augmented OOFs
            composite = compute_composite_from_augmented(targets, branch_name)
        else:
            # Use all original OOFs
            composite = compute_composite_from_original(targets, branch_name)
        
        SKILL_STATE["anchor_oof_score_augmented"] = composite
    
    return {"status": "OK"}
```

---

## Testing Strategy

### Test 1: Single-Target Backward Compatibility
```bash
# Run existing single-target competition
python -m zindian.orchestrator run_phase "1"
python -m zindian.orchestrator run_phase "2A"
python -m zindian.orchestrator run_phase "2B"

# Verify: target_config absent, top-level fields used
cat challenge_config.json | jq '.target_config'  # Should be null
```

### Test 2: Multi-Target Config Generation
```bash
# Create mock World Cup submission format with 2 targets
echo "team_id,goals_scored,stage_reached" > SampleSubmission.csv

# Run skill_02
python -c "from zindian.skills.skill_02_intake import run; run()"

# Verify: target_config present with 2 entries
cat challenge_config.json | jq '.target_config.targets | length'  # Should be 2
```

### Test 3: Per-Target OOF Generation
```bash
# Run skill_08 with multi-target config
python -c "from zindian.skills.skill_08_anchor import run; run()"

# Verify: Per-target OOF keys exist
cat SKILL_STATE.json | jq 'keys | map(select(contains("_oof")))'
# Expected: ["branch_anchor_goals_scored_oof", "branch_anchor_stage_reached_oof"]
```

### Test 4: Composite Score Calculation
```python
# Verify composite aggregation
import json
state = json.load(open("SKILL_STATE.json"))

assert "anchor_oof_score" in state  # Composite scalar
assert "anchor_oof_score_per_target" in state  # Per-target dict
assert len(state["anchor_oof_score_per_target"]) == 2
```

### Test 5: A12 Policy Enforcement
```python
# Test freeze_unaugmented_targets_at_original
config = {
    "target_config": {
        "targets": [
            {"name": "goals", "task_type": "regression"},
            {"name": "stage", "task_type": "classification"}
        ],
        "pseudo_label_recombination_policy": "freeze_unaugmented_targets_at_original"
    }
}

# Run skill_21
result = skill_21.run()

# Verify: Classification uses augmented, regression uses original
assert "goals_oof" in state  # Original
assert "stage_oof_augmented" in state  # Augmented
```

---

## Implementation Checklist

### Phase 1: Config Intake
- [ ] skill_02 detects multi-target from submission format
- [ ] target_config written with all required fields
- [ ] A12 policy injected for mixed-task competitions
- [ ] Backward compatibility: single-target unchanged

### Phase 2: OOF Loops
- [ ] skill_08 loops over targets independently
- [ ] Regression transformation lifecycle implemented
- [ ] Per-target OOF records written with target_name
- [ ] Composite score aggregation working
- [ ] anchor_oof_score_per_target preserved for Gate 1

### Phase 3: SHAP & Gates
- [ ] skill_10 computes SHAP per target
- [ ] leaked_features_{target_name} keys written
- [ ] skill_11 Condition 1 checks ALL target leak lists
- [ ] Composite variance threshold uses weighted RMS

### Phase 4: Pseudo-Label Policy
- [ ] skill_21 reads pseudo_label_recombination_policy
- [ ] freeze_unaugmented_targets_at_original implemented
- [ ] block_composite_until_all_targets_augmented_or_none implemented
- [ ] Partial augmentation blocked correctly

---

## Success Criteria

✅ **Single-Target Baseline:** Existing competitions run unchanged  
✅ **Multi-Target Detection:** skill_02 correctly identifies 2+ targets  
✅ **Per-Target Training:** skill_08 trains separate models per target  
✅ **Composite Scoring:** Weighted distance aggregation working  
✅ **SHAP Per Target:** skill_10 generates per-target leak lists  
✅ **Gate Logic:** skill_11 blocks on any target leakage  
✅ **A12 Enforcement:** skill_21 implements both recombination policies  

---

**Status:** READY FOR IMPLEMENTATION  
**Next Action:** Execute Prompts 1-4 sequentially  
**Target Competition:** FIFA World Cup 2026 (60% RMSE Goals + 40% F1 Stage)
