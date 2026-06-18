# Zindian Orchestrator - State Management & Gate Protocol Audit

## State Write Authorization Matrix

### ✅ Skills Authorized to Write State

| Skill | Writes To | Purpose |
|-------|-----------|---------|
| skill_01 | `phase_1_complete`, `md5_*_file` | File integrity hashes |
| skill_02 | `dag_phase`, `md5_target_hash` | Challenge intake, target validation |
| skill_05 | `cv_strategy`, `cv_strategy_id` | CV strategy selection |
| skill_06 | `phase_2a_complete`, `dead_features`, `noise_features` | Data cleaning results |
| skill_07 | Feature metadata | Feature engineering artifacts |
| skill_08 | `anchor_oof_score`, `anchor_oof_f1`, `anchor_oof_rmse`, `anchor_multi_target_metrics`, `branch_anchor-baseline_oof`, `dag_phase`, `phase_2b_complete`, `competition`, `md5_target_hash` | Baseline training results |
| skill_09 | `calibration_method`, `calibration_written_at`, `calibration_oof_cv_strategy_id` | Probability calibration |
| skill_12 | `metric_analysis` | Fold score variance analysis |
| skill_21 | `pseudo_label_result`, `pseudo_label_best_iteration`, `pseudo_label_best_oof_f1`, `branch_pseudo_label_*_oof` | Pseudo-labeling results |
| skill_13 | `last_ensemble_path`, `last_ensemble_oof_metric`, `last_ensemble_variants` | Ensemble blending results |

### ❌ Skills NEVER Write State

- skill_03 (policy gate) - Read-only compliance checker
- skill_04 (EDA) - Read-only analysis
- skill_10 (SHAP) - Read-only feature importance
- skill_11 (branch gate) - Read-only gate checker
- skill_14 (inference) - Read-only prediction formatter
- skill_15 (reporter) - Read-only report generator
- skill_16 (submit) - Submission only, no state writes
- skill_17 (governance) - Read-only audit
- skill_22 (reproducibility) - Read-only verification

---

## Human Gate Protocol (Anti-AutoML)

### The 5 Human Gates

| Gate | Key | Written By | Phase | Purpose |
|------|-----|------------|-------|---------|
| Gate 1 | `human_gate_1_approved`, `human_gate_1_timestamp` | **HUMAN ONLY** | After Phase 2B | Approve anchor baseline before variants |
| Gate 2 | `human_gate_2_{branch}_approved` | **HUMAN ONLY** | After each variant | Approve individual variants for blending |
| Gate 3 | `human_gate_3_approved` | **HUMAN ONLY** | Before Phase 3B fusion | Approve ensemble blending |
| Gate 4 | `human_gate_4_approved` | **HUMAN ONLY** | Before Phase 4 submission | Approve final submission |
| Gate 5 | `human_gate_5_approved` | **HUMAN ONLY** | Post-submission | Approve code review package |

### Critical Rules

1. **No skill ever writes a `human_gate_*` key**
2. **No orchestrator logic auto-approves gates**
3. **Gate keys are absent by default** (not set to `false`)
4. **Absence of key = gate is closed**
5. **Human must manually edit SKILL_STATE.json to open gates**

---

## Current State Analysis (World Cup 2026)

### State Keys Present
```json
{
  "competition": "world-cup-2026-goal-prediction-challenge",
  "md5_target_hash": "7a56a04f0ea8fbbd3ce8d9693112ed23",
  "anchor_oof_score": 0.3509490231163602,
  "anchor_oof_f1": 0.7018980462327205,
  "anchor_oof_rmse": 2.5794530038150083,
  "dag_phase": "phase_2_anchor_confirmed",
  "phase_1_complete": true,
  "phase_2a_complete": true,
  "phase_2b_complete": true,
  "human_gate_1_approved": true,
  "human_gate_1_timestamp": "2026-06-18T14:19:00+00:00"
}
```

### Gate Status
- ✅ **Gate 1**: OPEN (manually approved)
- ❌ **Gate 2**: CLOSED (no variants approved yet)
- ❌ **Gate 3**: CLOSED (not reached)
- ❌ **Gate 4**: CLOSED (not reached)
- ❌ **Gate 5**: CLOSED (not reached)

### Why skill_11 Shows BLOCKED
- skill_11 checks for `human_gate_2_{branch}_approved` keys
- No variants have been trained yet beyond anchor baseline
- **This is correct behavior** - you cannot blend variants that don't exist
- skill_11 logs "BLOCKED" but doesn't halt the orchestrator
- Other skills in Phase 3B continue (skill_21, skill_13)

### Why skill_13 Shows PARTIAL
- skill_13 requires Human Gate 2 approved variants to blend
- No `human_gate_2_*` keys exist in state
- Returns `PARTIAL` status: "ran successfully but found no candidates"
- **This is correct behavior** - cannot blend without approved variants

---

## Multi-Target State Additions (Phase 2B)

### New Keys Written by skill_08
```json
{
  "anchor_multi_target_metrics": {
    "total_goals": {
      "oof_logloss": 2.579,
      "oof_auc": 0.0,
      "oof_f1": 0.0,
      "threshold": 0.0,
      "fold_scores": [2.508, 2.498, 2.735, 2.674, 2.470]
    },
    "Target": {
      "oof_logloss": 0.283,
      "oof_auc": 0.985,
      "oof_f1": 0.702,
      "threshold": 0.0,
      "fold_scores": [0.847, 0.878, 0.888, 0.918, 0.918]
    }
  },
  "branch_anchor-baseline_total_goals_oof": {
    "branch_name": "anchor-baseline_total_goals",
    "scores": [...],
    "cv_strategy_id": "stratified_5fold",
    "model_config": {
      "feature_count": 11,
      "n_splits": 5,
      "threshold": 0.0,
      "fold_scores": [2.508, 2.498, 2.735, 2.674, 2.470],
      "target_name": "total_goals"
    }
  },
  "branch_anchor-baseline_Target_oof": {
    "branch_name": "anchor-baseline_Target",
    "scores": [...],
    "cv_strategy_id": "stratified_5fold",
    "model_config": {
      "feature_count": 11,
      "n_splits": 5,
      "threshold": 0.0,
      "fold_scores": [0.847, 0.878, 0.888, 0.918, 0.918],
      "target_name": "Target"
    }
  }
}
```

---

## Violations to Watch For

### ❌ NEVER DO THIS
1. Skills writing `human_gate_*` keys
2. Skills writing `challenge_config.json` after Phase 1
3. Skills modifying other skills' state keys
4. Orchestrator auto-approving gates based on metrics
5. Skills writing `dag_phase` outside their authorized phase

### ✅ ALWAYS DO THIS
1. Skills only write their designated state keys
2. Human manually edits SKILL_STATE.json for gates
3. Skills read state as read-only except for their keys
4. Gate checks return BLOCKED status, don't halt execution
5. State updates include `last_updated` timestamp

---

## Next Steps for World Cup 2026

1. **Train variants** (Phase 2B loop with `--variant` flag)
2. **Manually approve variants** by adding `human_gate_2_variant-1_approved: true` to SKILL_STATE.json
3. **Open Gate 3** by adding `human_gate_3_approved: true`
4. **Run Phase 3B** - skill_13 will now blend approved variants
5. **Open Gate 4** before submission

---

## Audit Timestamp
Generated: 2026-06-18T14:30:00+00:00
Pipeline Status: Phase 2B Complete, Gate 1 Open, Ready for Variants
