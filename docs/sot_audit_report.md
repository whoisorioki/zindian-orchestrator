# Source of Truth v2.2.1-Multi-Target Audit Report

**Audit Date:** June 2026  
**Auditor:** Amazon Q Developer  
**Scope:** Comparison of SoT documentation against actual codebase implementation

---

## Executive Summary

This audit identifies gaps between the signed-off Source of Truth document v2.2.1-Multi-Target and the actual codebase implementation. The audit focuses on:
1. Multi-target extensions (15 patches)
2. Resolved blocking issues (A1, A2, A3)
3. Core architectural contracts
4. Skill-level implementation compliance

**Overall Status:** ⚠️ **PARTIAL IMPLEMENTATION** — Core v2.2 features implemented, multi-target v2.2.1 extensions NOT implemented

---

## Critical Findings

### 🔴 CRITICAL GAP 1: Multi-Target Extensions Not Implemented

**SoT Claims:** v2.2.1 extends single-target architecture to multi-target competitions with 15 patches

**Reality:** Zero multi-target code found in any skill

**Evidence:**
```bash
$ grep -l "target_config\|multi.target" zindian/skills/*.py
# No results
```

**Impact:** 
- All 15 multi-target patches in SoT are documentation-only
- Skills 02, 04, 07, 08, 10, 11, 12, 21 lack multi-target loops
- `anchor_oof_score_per_target`, `composite_score`, `leaked_features_{target_name}` not implemented
- Plugin contract (FeatureExtractor ABC) not implemented
- Existing plugins (nedbank_extractor.py, terraclimate_extractor.py) use ad-hoc `fetch()`/`extract()` pattern, not the documented ABC interface

**Recommendation:** Either:
1. Downgrade SoT status from "SIGNED OFF" to "PROPOSED" until implementation complete
2. Create v2.2.1-Implementation-Roadmap.md documenting implementation plan
3. Add explicit "NOT YET IMPLEMENTED" markers to all multi-target sections in SoT

---

### 🔴 CRITICAL GAP 2: A12 Pseudo-Label Recombination Policy Not Implemented

**SoT Claims:** A12 requires `pseudo_label_recombination_policy` field for mixed-task multi-target competitions

**Reality:** No implementation in skill_21 or any other skill

**Evidence:**
```bash
$ grep -n "pseudo_label_recombination_policy" zindian/skills/*.py
# No results
```

**Impact:**
- Blocking issue A1 marked "RESOLVED" in SoT but not implemented
- skill_21 cannot handle multi-target pseudo-labeling
- Phase 3B gate checklist references unimplemented feature

**Recommendation:** Reopen A1 as blocking issue until implementation complete

---

### 🔴 CRITICAL GAP 3: Missingness-Interaction SHAP Rule (skill_07)

**SoT Claims:** skill_07 creates interaction terms between MNAR indicators and top SHAP features from anchor when `missingness_level == "high"`

**Reality:** skill_07 does NOT read any SHAP values (confirmed in previous investigation)

**Evidence:**
```bash
$ grep -rn "anchor.*shap\|shap.*anchor" zindian/skills/skill_07*.py
# No results
```

**Impact:**
- Documentation describes unimplemented feature
- Creates false expectation of capability
- Phase-ordering deadlock (skill_07 runs Phase 2B, skill_10 SHAP runs Phase 3A) would occur if implemented

**Recommendation:** Remove this rule from SoT Section 4 skill_07 contract OR implement with proper phase-ordering solution

---

## Moderate Findings

### 🟡 MODERATE GAP 1: Plugin Contract Mismatch

**SoT Claims:** Plugin contract requires `FeatureExtractor` ABC with `extract_features()` method

**Reality:** Existing plugins use `fetch()` + `extract()` pattern, no ABC base class

**Evidence:**
- nedbank_extractor.py: `def fetch(...)` and `def extract(...)`
- terraclimate_extractor.py: Similar pattern
- No `plugins/base_extractor.py` file exists

**Impact:**
- Plugin interface inconsistent with documentation
- New plugin developers will follow wrong pattern

**Recommendation:** Either:
1. Create `plugins/base_extractor.py` with documented ABC
2. Update SoT to document actual `fetch()`/`extract()` pattern

---

### 🟡 MODERATE GAP 2: skill_15 Multi-Invocation Claim

**SoT Claims:** Section 8 Definition of Done states skill_15 logs Phase 2B per-branch metrics

**Reality:** skill_15 is single-invocation function running only in Phase 1 (confirmed in previous investigation)

**Evidence:**
- skill_15_reporter.py implements single `run()` function
- No multi-invocation logic
- `secondary_metrics` already written by skill_07/skill_08 to OOF records

**Impact:**
- Documentation inconsistency (already identified in Issue 2 investigation)
- Section 8 checklist item is redundant

**Recommendation:** Remove Phase 2B logging requirement from Section 8 checklist (already recommended in Issue 2 resolution)

---

## Positive Findings (Implemented Correctly)

### ✅ IMPLEMENTED: Secondary Metrics (A2 Resolution)

**SoT Claims:** skill_07 and skill_08 write `secondary_metrics` nested dict to OOF records

**Reality:** ✅ Correctly implemented

**Evidence:**
```python
# skill_07_features.py:1313-1342
secondary_metrics = None
if config.get("task_type") == "regression":
    from zindian.state import compute_secondary_metrics
    secondary_metrics = compute_secondary_metrics(y_true, result["oof_probs"])
# ... writes to OOF record

# skill_08_anchor.py:554-591
secondary_metrics = None
if config.get("task_type") == "regression":
    from zindian.state import compute_secondary_metrics
    secondary_metrics = compute_secondary_metrics(y_true, oof_preds)
# ... writes to OOF record
```

---

### ✅ IMPLEMENTED: Effective Threshold Normalization (A2 Resolution)

**SoT Claims:** skill_11 uses `effective_variance_threshold` and `effective_gate_margin` with scale normalization for regression

**Reality:** ✅ Correctly implemented

**Evidence:**
```python
# skill_11_gate.py:58, 198, 241-242, 274, 294, 296
effective_variance_threshold, effective_gate_margin, threshold_warning = (...)
# Used in gate conditions 2 and 3
```

---

### ✅ IMPLEMENTED: OOF Contract (A7)

**SoT Claims:** All OOF-generating skills use CV strategy from config, tag outputs with `cv_strategy_id`

**Reality:** ✅ Verified in skill_07, skill_08 (from previous investigation)

---

### ✅ IMPLEMENTED: skill_22 Reproducibility Audit

**SoT Claims:** skill_22 verifies lockfile, AutoML imports, OOF tags, branch tracking

**Reality:** ✅ Correctly implemented (verified in previous investigation)

**Evidence:**
- Lockfile verification present
- AST scan for AutoML imports present
- OOF cv_strategy_id audit present
- Branch tracking audit present

---

## Recommendations by Priority

### Priority 1 (Blocking)

1. **Add "NOT YET IMPLEMENTED" warnings** to all multi-target sections in SoT
2. **Reopen A1** as blocking issue until `pseudo_label_recombination_policy` implemented
3. **Remove missingness-interaction SHAP rule** from skill_07 contract OR implement properly

### Priority 2 (High)

4. **Create implementation roadmap** for v2.2.1 multi-target features
5. **Document actual plugin interface** (fetch/extract pattern) OR implement documented ABC

### Priority 3 (Medium)

6. **Remove Phase 2B logging requirement** from skill_15 Section 8 checklist
7. **Add audit CI check** to prevent future doc/code drift

---

## Audit Checklist

| SoT Section | Feature | Status | Notes |
|-------------|---------|--------|-------|
| A11 | Multi-target config declaration | ❌ NOT IMPL | No target_config handling |
| A12 | Pseudo-label recombination policy | ❌ NOT IMPL | Blocking issue A1 |
| Section 2 | Composite score computation | ❌ NOT IMPL | No multi-target scoring |
| Section 2 | Composite variance threshold | ❌ NOT IMPL | Formula documented but unused |
| Section 2 | Secondary metrics | ✅ IMPL | skill_07, skill_08 correct |
| Section 3 | Preflight OOF regex validation | ⚠️ PARTIAL | A3 resolution not verified in code |
| Section 3 | Preflight completeness check | ⚠️ PARTIAL | A3 resolution not verified in code |
| Section 4 | skill_02 target_config writing | ❌ NOT IMPL | No multi-target intake |
| Section 4 | skill_04 per-target std | ❌ NOT IMPL | Only single target_std |
| Section 4 | skill_07 missingness-SHAP rule | ❌ NOT IMPL | Documented but not coded |
| Section 4 | skill_08 multi-target loop | ❌ NOT IMPL | No per-target training |
| Section 4 | skill_08 anchor_oof_score_per_target | ❌ NOT IMPL | Composite scoring missing |
| Section 4 | skill_10 per-target SHAP | ❌ NOT IMPL | No multi-target SHAP |
| Section 4 | skill_11 effective thresholds | ✅ IMPL | Correctly normalized |
| Section 4 | skill_11 composite gate logic | ❌ NOT IMPL | No multi-target gating |
| Section 4 | skill_21 recombination policy | ❌ NOT IMPL | A12 not implemented |
| Section 4 | Plugin FeatureExtractor ABC | ❌ NOT IMPL | Ad-hoc pattern used instead |
| Section 8 | skill_15 Phase 2B logging | ❌ INCORRECT | Single-invocation only |

---

## Conclusion

The Source of Truth v2.2.1-Multi-Target document is **signed off prematurely**. While the base v2.2 single-target architecture is correctly implemented, the 15 multi-target patches that define v2.2.1 are documentation-only.

**Recommended Action:** Change SoT status from "SIGNED OFF" to "PROPOSED — IMPLEMENTATION PENDING" until multi-target features are coded and tested.

---

## Phase Architecture Audit

### Orchestrator Phase Definitions vs. SoT Documentation

**Orchestrator Hardcoded Phases** (`zindian/orchestrator.py:15-19`):
```python
PHASE_1_SKILLS = ["skill_01", "skill_02", "skill_15"]
PHASE_2_SKILLS = ["skill_03", "skill_08"]
PHASE_3_SKILLS = ["skill_04", "skill_05", "skill_09", "skill_10"]
PHASE_4_SKILLS = ["skill_11", "skill_16"]
PHASE_5_SKILLS = ["skill_13", "skill_14", "skill_17"]
```

**SoT Documentation Claims:**
- **Phase 1**: skill_01 → skill_02 → skill_03 (policy_writer) → skill_04 → skill_05 → skill_15
- **Phase 2A**: skill_03 (policy_gate) → skill_06
- **Phase 2B**: skill_08 → skill_07
- **Phase 3A**: skill_10 → skill_09 → skill_12
- **Phase 3B**: skill_11 → skill_21 → skill_13
- **Phase 4**: skill_14 → skill_16 → skill_17 → skill_22

---

### 🔴 CRITICAL MISMATCH: Phase Definitions Completely Inconsistent

| Skill | SoT Phase | Orchestrator Phase | Status |
|-------|-----------|-------------------|--------|
| skill_01 | Phase 1 | Phase 1 | ✅ MATCH |
| skill_02 | Phase 1 | Phase 1 | ✅ MATCH |
| skill_03 (policy_writer) | Phase 1 | Phase 2 | ❌ MISMATCH |
| skill_03 (policy_gate) | Phase 2A | Phase 2 | ⚠️ PARTIAL |
| skill_04 | Phase 1 | Phase 3 | ❌ MISMATCH |
| skill_05 | Phase 1 | Phase 3 | ❌ MISMATCH |
| skill_06 | Phase 2A | NOT IN ORCHESTRATOR | ❌ MISSING |
| skill_07 | Phase 2B | NOT IN ORCHESTRATOR | ❌ MISSING |
| skill_08 | Phase 2B | Phase 2 | ⚠️ PARTIAL |
| skill_09 | Phase 3A | Phase 3 | ⚠️ PARTIAL |
| skill_10 | Phase 3A | Phase 3 | ⚠️ PARTIAL |
| skill_11 | Phase 3B | Phase 4 | ❌ MISMATCH |
| skill_12 | Phase 3A | NOT IN ORCHESTRATOR | ❌ MISSING |
| skill_13 | Phase 3B | Phase 5 | ❌ MISMATCH |
| skill_14 | Phase 4 | Phase 5 | ❌ MISMATCH |
| skill_15 | Phase 1 | Phase 1 | ✅ MATCH |
| skill_16 | Phase 4 | Phase 4 | ✅ MATCH |
| skill_17 | Phase 4 | Phase 5 | ❌ MISMATCH |
| skill_21 | Phase 3B | NOT IN ORCHESTRATOR | ❌ MISSING |
| skill_22 | Phase 4 | NOT IN ORCHESTRATOR | ❌ MISSING |

---

### Critical Issues

#### 🔴 ISSUE 1: skill_03 Split Function Not Implemented

**SoT Claims:** skill_03 implements two separate functions:
- `policy_writer()` runs in Phase 1
- `policy_gate()` runs in Phase 2A

**Reality:** skill_03 only has single `run()` function, no split implementation

**Evidence:**
```bash
$ grep "^def " zindian/skills/skill_03_legality.py
def _normalize_policy_token(value: Any) -> str:
def _collect_banned_features(
def _normalize_planned_feature_entries(entries: Any) -> List[Dict[str, Any]]:
def synthesise_feature_policy(
def check_planned_features(
def _write_feature_policy(paths, policy: Dict[str, Any]) -> None:
def _write_legality_report(
def run(
# No policy_writer() or policy_gate() functions
```

**Impact:** 
- SoT architectural principle violated
- Phase 1 vs Phase 2A separation not enforced
- Cannot independently test policy writing vs policy enforcement

---

#### 🔴 ISSUE 2: Phase 1 Skills Misplaced in Orchestrator

**SoT Claims:** Phase 1 = skill_01 → skill_02 → skill_03 → skill_04 → skill_05 → skill_15

**Reality:** Orchestrator Phase 1 = skill_01, skill_02, skill_15 ONLY

**Missing from Phase 1:**
- skill_03 (moved to Phase 2)
- skill_04 (moved to Phase 3)
- skill_05 (moved to Phase 3)

**Impact:**
- Config lock timing violated (skill_05 writes cv_strategy, should lock after Phase 1)
- EDA outputs (skill_04) not available when Phase 1 gate checks
- Policy filters (skill_03) not written before Phase 1 gate

---

#### 🔴 ISSUE 3: Critical Skills Missing from Orchestrator

**Skills documented in SoT but NOT in any orchestrator phase:**
- skill_06 (cleaning) — Phase 2A
- skill_07 (features) — Phase 2B
- skill_12 (metric) — Phase 3A
- skill_21 (pseudo-label) — Phase 3B
- skill_22 (reproducibility audit) — Phase 4

**Impact:** 
- 5 core skills cannot be executed via `run_phase()`
- Must be called manually via `run_skill()`
- Phase dependency chain not enforced

---

#### 🔴 ISSUE 4: Phase Numbering Inconsistency

**SoT uses 6 phases:**
- Phase 1: Competition Fingerprint + Config Lock
- Phase 2A: Data Cleaning
- Phase 2B: Signal Search
- Phase 3A: Generalisation Audit
- Phase 3B: Promotion and Fusion
- Phase 4: Governance

**Orchestrator uses 5 phases:**
- Phase 1, 2, 3, 4, 5 (no sub-phases)

**Impact:**
- Cannot express Phase 2A vs 2B distinction
- Cannot express Phase 3A vs 3B distinction
- Phase gate logic cannot enforce sub-phase dependencies

---

### Recommendations

#### Priority 1 (Blocking)

1. **Implement skill_03 split functions** (`policy_writer()` and `policy_gate()`)
2. **Fix Phase 1 definition** in orchestrator to include skill_03, skill_04, skill_05
3. **Add missing skills** to orchestrator phase definitions (skill_06, skill_07, skill_12, skill_21, skill_22)

#### Priority 2 (High)

4. **Implement sub-phase support** in orchestrator (2A/2B, 3A/3B)
5. **Add phase gate validation** to enforce dependency chain
6. **Update SoT** to match actual orchestrator implementation OR fix orchestrator to match SoT

#### Priority 3 (Medium)

7. **Add CI test** to verify orchestrator phase definitions match SoT
8. **Document phase_skill_map** override mechanism in SoT

---

### Phase Execution Order Verification

**SoT Documented Order:**
```
Phase 1: 01 → 02 → 03(writer) → 04 → 05 → 15
Phase 2A: 03(gate) → 06
Phase 2B: 08 → 07
Phase 3A: 10 → 09 → 12
Phase 3B: 11 → 21 → 13
Phase 4: 14 → 16 → 17 → 22
```

**Orchestrator Actual Order:**
```
Phase 1: 01 → 02 → 15
Phase 2: 03 → 08
Phase 3: 04 → 05 → 09 → 10
Phase 4: 11 → 16
Phase 5: 13 → 14 → 17
```

**Skills Never Executed by run_phase():**
- skill_06, skill_07, skill_12, skill_21, skill_22

---

*End of Phase Architecture Audit*
