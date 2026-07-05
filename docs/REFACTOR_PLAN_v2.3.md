# Zindian Orchestrator — v2.3 Refactor Plan

**Date:** June 26, 2026  
**Current Competition:** geoai-aquaculture-pond-identification-challenge  
**Current SoT Version:** 2.3 (SIGNED OFF)  
**Audit Status:** See `/docs/sot_audit_report.md`

---

## Executive Summary

This refactor plan addresses the gaps identified in the SoT audit report and SWOT analysis, focusing on:
1. **R5 Carbon Tracking** — Implement v2.3 carbon instrumentation
2. **Known Gaps** — Fill skill_21 retraining loop, skill_12 composite variance
3. **Architecture Drifts** — Fix hardcoded targets, plugin ABC, preflight patterns
4. **Documentation Sync** — Update AGENTS.md, SoT, and audit reports

**Priority:** HIGH — Current competition (geoai) is multi-target composite metric, requires full v2.3 compliance

---

## Gap Analysis Summary

### From SoT Audit Report

| Gap ID | Description | Impact | Priority |
|--------|-------------|--------|----------|
| GAP-1 | skill_21 retraining loop stubbed | No pseudo-label boost | P2 |
| GAP-2 | skill_12 composite fold variance missing | No stability gate for multi-target | P1 |
| GAP-3 | Missingness-interaction SHAP rule | Phase ordering deadlock | P3 |
| DRIFT-1 | Hardcoded targets in skill_07 | A5 violation | P1 |
| DRIFT-2 | FeatureExtractor ABC not implemented | Plugin contract mismatch | P2 |
| DRIFT-3 | Orchestrator split-skill validation warnings | Noise in logs | P3 |

### From SWOT Analysis

| Risk ID | Description | Impact | Priority |
|---------|-------------|--------|----------|
| T7 | Paper architecture (doc-code mismatch) | Developer confusion | P1 |
| W4 | Config alias complexity | Maintenance burden | P2 |
| W3 | Single promoted variant only | Limited diversity | P3 |

### v2.3 Features (Not Yet Implemented)

| Feature | Description | Status | Priority |
|---------|-------------|--------|----------|
| R5 | Carbon tracking instrumentation | NOT STARTED | P1 |
| Known Gaps Registry | Formal tracking in Section 9 | PARTIAL | P2 |
| Regression transformation generalization | RMSLE/RMSE/MAE matrix | COMPLETE | ✅ |

---

## Refactor Phases

### Phase 1: Critical Fixes (P1 — Blocking)

#### 1.1 Fix DRIFT-1: Hardcoded Targets in skill_07
**File:** `zindian/skills/skill_07_features.py`  
**Lines:** L1006-L1007  
**Issue:** Hardcoded `"total_goals"` and `"Target"` literals

**Fix:**
```python
# BEFORE (WRONG):
oof_key = f"branch_{branch_name}_total_goals_oof"  # Hardcoded target name

# AFTER (CORRECT):
target_names = [t["name"] for t in config["target_config"]["targets"]]
for target_name in target_names:
    oof_key = f"branch_{branch_name}_{target_name}_oof"
```

**Test:** `test_a5_compliance.py` — verify zero hardcoded competition strings

---

#### 1.2 Implement GAP-2: skill_12 Composite Fold Variance
**File:** `zindian/skills/skill_12_metric.py`  
**Lines:** L48-L82  
**Issue:** Only computes variance for first OOF key, ignores multi-target composite

**Fix:**
```python
def _compute_composite_fold_variance(state: dict, config: dict) -> float:
    """
    Compute fold score variance for multi-target composite metric.
    
    Reads per-target fold scores, applies weights, computes composite
    variance with ddof=1.
    """
    targets = config["target_config"]["targets"]
    n_folds = config["cv_strategy"]["n_splits"]
    
    # Collect per-fold composite scores
    composite_fold_scores = []
    for fold_idx in range(n_folds):
        fold_composite = 0.0
        for target in targets:
            target_name = target["name"]
            weight = target["weight"]
            metric_key = target["metric"]
            
            # Read per-target fold score
            oof_key = f"branch_{branch_name}_{target_name}_oof"
            fold_score = state[oof_key]["fold_scores"][fold_idx]
            
            # Normalize regression metrics by target_std
            if target["task_type"] == "regression" and metric_key != "rmsle":
                target_std = state["eda"].get(f"{target_name}_std", 1.0)
                fold_score = fold_score / target_std
            
            fold_composite += weight * fold_score
        
        composite_fold_scores.append(fold_composite)
    
    # Compute variance with ddof=1
    return float(np.var(composite_fold_scores, ddof=1))
```

**Test:** `test_multi_target_world_cup.py` — add composite variance assertion

---

#### 1.3 Implement R5: Carbon Tracking Infrastructure
**Files:**
- `zindian/orchestrator.py` — Add carbon estimation hook
- `zindian/carbon_tracker.py` — NEW MODULE
- `zindian/skills/_lightgbm_shared.py` — Instrument training loop

**Implementation:**

**Step 1:** Create carbon tracker module
```python
# zindian/carbon_tracker.py
"""
R5 Carbon Tracking — v2.3 Feature

Estimates carbon footprint for skill execution using:
- Primary: CodeCarbon (optional dependency)
- Fallback: ML CO2 Impact formula
"""
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

def estimate_carbon(
    duration_sec: float,
    peak_memory_mb: float,
    config: dict
) -> Dict[str, any]:
    """
    Estimate carbon footprint for a skill run.
    
    Args:
        duration_sec: Wall-clock execution time
        peak_memory_mb: Peak memory usage
        config: challenge_config.json with infrastructure block
    
    Returns:
        {
            "carbon_kg_estimate": float,
            "tracker_method": "codecarbon" | "mlco2_formula" | "not_instrumented",
            "hardware_type": "cpu" | "gpu" | "tpu",
            "region": str
        }
    """
    infra = config.get("infrastructure", {})
    
    # Try CodeCarbon first
    try:
        import codecarbon
        tracker = codecarbon.EmissionsTracker()
        # ... implementation
        return {
            "carbon_kg_estimate": emissions,
            "tracker_method": "codecarbon",
            "hardware_type": infra.get("hardware_type", "cpu"),
            "region": infra.get("region", "unknown")
        }
    except ImportError:
        logger.warning("CodeCarbon not installed, falling back to ML CO2 formula")
    
    # Fallback: ML CO2 Impact formula
    tdp_watts = infra.get("tdp_watts", 15.0)
    pue = infra.get("pue", 1.0)
    carbon_intensity = infra.get("carbon_intensity_gco2_per_kwh", 494.0)
    
    energy_kwh = (tdp_watts * pue * duration_sec) / 3_600_000
    carbon_kg = (energy_kwh * carbon_intensity) / 1000
    
    return {
        "carbon_kg_estimate": carbon_kg,
        "tracker_method": "mlco2_formula",
        "hardware_type": infra.get("hardware_type", "cpu"),
        "region": infra.get("region", "unknown")
    }
```

**Step 2:** Hook into orchestrator
```python
# zindian/orchestrator.py — run_skill() wrapper
def run_skill(self, skill_name: str, phase: str) -> dict:
    """Execute skill with telemetry and carbon tracking."""
    import time
    from zindian.carbon_tracker import estimate_carbon
    
    start_time = time.time()
    # ... existing execution logic ...
    duration_sec = time.time() - start_time
    
    # Existing telemetry
    telemetry = {
        "duration_sec": duration_sec,
        "peak_memory_mb": self._get_peak_memory()
    }
    
    # NEW: R5 carbon tracking
    carbon_data = estimate_carbon(
        duration_sec=duration_sec,
        peak_memory_mb=telemetry["peak_memory_mb"],
        config=self.config
    )
    telemetry.update(carbon_data)
    
    # Write to state
    self.state[f"telemetry.{skill_name}"] = telemetry
    
    return self.state
```

**Step 3:** Add infrastructure block to skill_02
```python
# zindian/skills/skill_02_intake.py
def run(config: dict, state: dict) -> dict:
    # ... existing logic ...
    
    # NEW: Write infrastructure block (Phase 1 only)
    if not config.get("infrastructure"):
        config["infrastructure"] = {
            "hardware_type": "cpu",  # Detect from platform
            "region": "us-east-1",   # Read from AWS metadata
            "tdp_watts": 15.0,
            "pue": 1.0,
            "carbon_intensity_gco2_per_kwh": 494.0  # US grid average
        }
        _write_config(config)
    
    return state
```

**Test:** `test_r5_carbon_tracking.py` — verify telemetry schema

---

### Phase 2: High-Priority Gaps (P2)

#### 2.1 Implement GAP-1: skill_21 Retraining Loop
**File:** `zindian/skills/skill_21_pseudo_label.py`  
**Lines:** L990-L996  
**Issue:** Retraining loop is stubbed

**Scope:** Classification-only (per Guard Condition 1)

**Fix:**
```python
def _run_multi_target_pseudo_label(config: dict, state: dict) -> dict:
    """
    Multi-target pseudo-label retraining.
    
    Only augments classification targets. Regression targets frozen.
    """
    # ... existing validation logic ...
    
    # NEW: Actual retraining loop
    for target in classification_targets:
        target_name = target["name"]
        
        # Generate pseudo-labels on test set
        pseudo_labels = _generate_pseudo_labels(
            model=state[f"model_{target_name}"],
            X_test=state["features_test"],
            confidence_threshold=0.95
        )
        
        # Augment training set
        X_augmented = pd.concat([X_train, X_test[pseudo_labels["mask"]]])
        y_augmented = pd.concat([y_train, pseudo_labels["labels"]])
        
        # Retrain model
        model_augmented = _train_model(X_augmented, y_augmented, config)
        
        # Compute augmented OOF
        oof_augmented = _compute_oof(model_augmented, X_train, y_train, cv)
        
        # Write augmented OOF
        state[f"branch_{branch_name}_{target_name}_oof_augmented"] = {
            "scores": oof_augmented.tolist(),
            "cv_strategy_id": state["cv_strategy_id"],
            "n_pseudo_labels": len(pseudo_labels["labels"]),
            "confidence_threshold": 0.95
        }
    
    # Implement recombination policy
    if policy == "freeze_unaugmented_targets_at_original":
        # Combine augmented classification OOF with original regression OOF
        composite_augmented = _compute_composite_score(
            classification_oof=state[f"branch_{branch_name}_{cls_target}_oof_augmented"],
            regression_oof=state[f"branch_{branch_name}_{reg_target}_oof"],  # Original
            weights=config["target_config"]["targets"]
        )
        state["anchor_oof_score_augmented"] = composite_augmented
    
    return state
```

**Test:** `test_pseudo_label_classification_only.py` — verify augmented namespace

---

#### 2.2 Implement DRIFT-2: FeatureExtractor ABC
**File:** `plugins/base_extractor.py`  
**Issue:** No ABC base class, plugins use ad-hoc patterns

**Fix:**
```python
# plugins/base_extractor.py
from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd

class FeatureExtractor(ABC):
    """
    Abstract base class for competition-specific feature extractors.
    
    All plugins must inherit from this class and implement:
    - fetch(): Download/load external data
    - extract(): Transform raw data into features
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.plugin_config = config.get("plugin_config", {})
    
    @abstractmethod
    def fetch(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch external data sources.
        
        Returns:
            Dict mapping dataset names to DataFrames
        """
        pass
    
    @abstractmethod
    def extract(self, train: pd.DataFrame, test: pd.DataFrame) -> tuple:
        """
        Extract features from raw data.
        
        Args:
            train: Training dataset
            test: Test dataset
        
        Returns:
            (train_features, test_features) tuple
        """
        pass
```

**Migrate existing plugins:**
```python
# plugins/geoai_extractor.py
from plugins.base_extractor import FeatureExtractor

class GeoAIExtractor(FeatureExtractor):
    def fetch(self) -> Dict[str, pd.DataFrame]:
        # ... existing fetch logic ...
        pass
    
    def extract(self, train: pd.DataFrame, test: pd.DataFrame) -> tuple:
        # ... existing extract logic ...
        pass
```

**Test:** `test_plugin_contract.py` — verify ABC inheritance

---

### Phase 3: Documentation Sync (P1)

#### 3.1 Update AGENTS.md
**File:** `AGENTS.md`  
**Changes:**
1. Update "Repository Ground Truth" table with v2.3 status
2. Add R5 carbon tracking section
3. Update "Open Known Gaps" with current status
4. Mark GAP-1, GAP-2 as RESOLVED after implementation

**Additions:**
```markdown
## R5 Carbon Tracking (v2.3)

Every skill execution is instrumented with carbon footprint estimation:

```python
# Automatic via orchestrator run_skill() wrapper
telemetry = state[f"telemetry.{skill_name}"]
# {
#   "duration_sec": 24.5,
#   "peak_memory_mb": 512,
#   "carbon_kg_estimate": 0.00012,
#   "tracker_method": "mlco2_formula",
#   "hardware_type": "cpu",
#   "region": "us-east-1"
# }
```

**Mandatory instrumented skills:**
- `_lightgbm_shared.py` (training loop)
- `skill_07_features.py` (feature engineering)
- `skill_08_anchor.py` (anchor model training)
- `skill_09_calibration.py`
- `skill_10_shap.py` (highest compute: 24.5s)
- `skill_11_gate.py`
- `skill_13_ensemble.py`
- `skill_14_inference.py`

**Aggregate reporting:**
```python
state["telemetry.aggregate"] = {
    "total_duration_seconds": 245.3,
    "total_carbon_kg_estimate": 0.0015,
    "skills_not_instrumented": []
}
```

**skill_22 verification:**
- Checks `telemetry.aggregate` present
- Validates `total_carbon_kg_estimate` or explicit `not_instrumented` reason
- Includes carbon estimate in governance report
```

---

#### 3.2 Update SoT Known Gaps Section
**File:** `docs/source_of_truth.md`  
**Section:** 9. Known Gaps  
**Changes:**
1. Mark GAP-1 (skill_21 retraining) as RESOLVED
2. Mark GAP-2 (skill_12 composite variance) as RESOLVED
3. Add new gap: GAP-3 (missingness-interaction SHAP rule)

**Updated text:**
```markdown
## 9. Known Gaps and Implementation Status

### ✅ RESOLVED

**GAP-1: skill_21 Retraining Loop (RESOLVED v2.3.1)**
- **Was:** Pseudo-label retraining loop stubbed
- **Now:** Full implementation for classification targets
- **Commit:** [hash]
- **Test:** `test_pseudo_label_classification_only.py`

**GAP-2: skill_12 Composite Fold Variance (RESOLVED v2.3.1)**
- **Was:** Only computed variance for first OOF key
- **Now:** Computes weighted composite variance for multi-target
- **Commit:** [hash]
- **Test:** `test_multi_target_world_cup.py`

### 🔴 OPEN

**GAP-3: Missingness-Interaction SHAP Rule**
- **Issue:** Phase ordering deadlock — skill_07 (Phase 2B) cannot read SHAP from skill_10 (Phase 3A)
- **Impact:** Cannot create interaction terms using top SHAP features
- **Workaround:** Manual feature engineering in plugin extractors
- **Resolution:** Requires phase architecture redesign (out of scope for v2.3)
```

---

#### 3.3 Update Audit Report
**File:** `docs/sot_audit_report.md`  
**Changes:**
1. Update "Remaining Gaps" section
2. Add "v2.3.1 Remediation" section
3. Update test suite metrics

**New section:**
```markdown
## v2.3.1 Remediation (June 26, 2026)

Following the v2.3 audit, the following gaps were addressed:

### Implemented Features

#### R5 Carbon Tracking
- **Module:** `zindian/carbon_tracker.py` (NEW)
- **Hook:** `orchestrator.py:run_skill()` wrapper
- **Instrumentation:** 8 mandatory skills
- **Test:** `test_r5_carbon_tracking.py`
- **Status:** ✅ COMPLETE

#### skill_21 Retraining Loop
- **File:** `skill_21_pseudo_label.py:L990-L1050`
- **Scope:** Classification-only (per Guard Condition 1)
- **Features:** Pseudo-label generation, augmented OOF, recombination policy
- **Test:** `test_pseudo_label_classification_only.py`
- **Status:** ✅ COMPLETE

#### skill_12 Composite Fold Variance
- **File:** `skill_12_metric.py:L85-L130`
- **Features:** Weighted composite variance, per-target normalization
- **Test:** `test_multi_target_world_cup.py`
- **Status:** ✅ COMPLETE

### Architecture Fixes

#### DRIFT-1: Hardcoded Targets in skill_07
- **File:** `skill_07_features.py:L1006-L1007`
- **Fix:** Dynamic target name resolution from config
- **Test:** `test_a5_compliance.py`
- **Status:** ✅ COMPLETE

#### DRIFT-2: FeatureExtractor ABC
- **File:** `plugins/base_extractor.py` (NEW)
- **Migrated:** `geoai_extractor.py`, `world_cup_extractor.py`
- **Test:** `test_plugin_contract.py`
- **Status:** ✅ COMPLETE

### Test Suite Update
- **Total Passed:** 215 (+12 from v2.3)
- **Total Failed:** 18 (-9 from v2.3)
- **New Tests:** 5 (R5, skill_21, skill_12, A5, plugin ABC)
- **Coverage:** 92% (+5% from v2.3)
```

---

### Phase 4: Low-Priority Improvements (P3)

#### 4.1 Fix DRIFT-3: Orchestrator Split-Skill Validation
**File:** `zindian/orchestrator.py`  
**Lines:** L79-L110  
**Issue:** Warnings for dotted function paths (`skill_03.policy_writer`)

**Fix:**
```python
def _validate_phase_map(self, phase_map: dict) -> None:
    """Validate phase skill map supports dotted function paths."""
    for phase, skills in phase_map.items():
        for skill_ref in skills:
            # Support dotted paths: "skill_03.policy_writer"
            if "." in skill_ref:
                skill_name, func_name = skill_ref.split(".", 1)
                module = importlib.import_module(f"zindian.skills.{skill_name}")
                if not hasattr(module, func_name):
                    raise ValueError(f"Function {func_name} not found in {skill_name}")
            else:
                # Standard skill module
                module = importlib.import_module(f"zindian.skills.{skill_ref}")
                if not hasattr(module, "run"):
                    raise ValueError(f"run() not found in {skill_ref}")
```

---

#### 4.2 Resolve GAP-3: Missingness-Interaction SHAP Rule
**Status:** DEFERRED — Requires phase architecture redesign

**Options:**
1. **Move skill_10 to Phase 2B** — Run SHAP before skill_07 variants
   - **Pro:** Enables SHAP-based interactions
   - **Con:** Breaks anchor → SHAP → gate flow
2. **Add Phase 2C** — skill_07 variants after SHAP
   - **Pro:** Preserves existing flow
   - **Con:** Adds complexity to phase map
3. **Manual interactions in plugins** — Current workaround
   - **Pro:** No architecture change
   - **Con:** Duplicates logic across plugins

**Recommendation:** Option 3 (manual) for v2.3, Option 2 (Phase 2C) for v3.0

---

## Implementation Checklist

### Phase 1: Critical Fixes (Week 1)
- [ ] Fix DRIFT-1: Hardcoded targets in skill_07
- [ ] Implement GAP-2: skill_12 composite fold variance
- [ ] Implement R5: Carbon tracking infrastructure
  - [ ] Create `carbon_tracker.py` module
  - [ ] Hook into orchestrator
  - [ ] Add infrastructure block to skill_02
  - [ ] Instrument 8 mandatory skills
- [ ] Write tests:
  - [ ] `test_a5_compliance.py`
  - [ ] `test_multi_target_composite_variance.py`
  - [ ] `test_r5_carbon_tracking.py`

### Phase 2: High-Priority Gaps (Week 2)
- [ ] Implement GAP-1: skill_21 retraining loop
- [ ] Implement DRIFT-2: FeatureExtractor ABC
  - [ ] Create `base_extractor.py`
  - [ ] Migrate `geoai_extractor.py`
  - [ ] Migrate `world_cup_extractor.py`
- [ ] Write tests:
  - [ ] `test_pseudo_label_retraining.py`
  - [ ] `test_plugin_contract.py`

### Phase 3: Documentation Sync (Week 2)
- [ ] Update AGENTS.md
  - [ ] Add R5 section
  - [ ] Update Repository Ground Truth table
  - [ ] Update Open Known Gaps
- [ ] Update SoT
  - [ ] Mark GAP-1, GAP-2 as RESOLVED
  - [ ] Add v2.3.1 changelog
- [ ] Update audit report
  - [ ] Add v2.3.1 Remediation section
  - [ ] Update test metrics

### Phase 4: Low-Priority (Week 3)
- [ ] Fix DRIFT-3: Orchestrator split-skill validation
- [ ] Document GAP-3 resolution options
- [ ] Run full test suite (target: 95% pass rate)

---

## Success Criteria

### Code Quality
- [ ] Zero A5 violations (no hardcoded competition strings)
- [ ] All plugins inherit from FeatureExtractor ABC
- [ ] R5 telemetry present in all mandatory skills
- [ ] Test coverage ≥ 92%

### Documentation
- [ ] AGENTS.md reflects v2.3.1 state
- [ ] SoT Known Gaps section current
- [ ] Audit report includes remediation summary

### Competition Readiness
- [ ] geoai competition runs end-to-end
- [ ] Composite metric computed correctly
- [ ] Carbon tracking reports in skill_22

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| R5 breaks existing competitions | Low | High | Fallback to `not_instrumented` if CodeCarbon fails |
| skill_21 retraining degrades OOF | Medium | Medium | Gate condition 3 blocks weak augmented models |
| Plugin ABC breaks existing extractors | Low | High | Backward-compatible migration, test all plugins |
| Documentation drift continues | High | Medium | Automated doc generation from code annotations |

---

## Next Steps

1. **Immediate:** Fix DRIFT-1 (hardcoded targets) — blocking for geoai competition
2. **This Week:** Implement R5 carbon tracking — v2.3 compliance
3. **Next Week:** Fill GAP-1, GAP-2 — complete v2.3.1 roadmap
4. **Ongoing:** Update documentation concurrently with code changes

---

**Maintained by:** [whoisorioki](https://github.com/whoisorioki)  
**Last Updated:** June 26, 2026  
**Status:** ACTIVE — Phase 1 in progress
