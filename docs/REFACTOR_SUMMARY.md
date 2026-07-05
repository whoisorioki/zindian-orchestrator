# v2.3 Refactor Initiative — Executive Summary

**Date:** June 26, 2026  
**Status:** PLANNING COMPLETE → READY FOR EXECUTION  
**Competition Context:** geoai-aquaculture-pond-identification-challenge (composite metric, multi-target)

---

## Quick Reference

| Document | Purpose | Status |
|----------|---------|--------|
| `REFACTOR_PLAN_v2.3.md` | Detailed implementation plan | ✅ COMPLETE |
| `AGENTS.md` | Agent operational guidelines | 🔄 NEEDS UPDATE |
| `docs/source_of_truth.md` | Architectural authority (v2.3) | 🔄 NEEDS UPDATE |
| `docs/sot_audit_report.md` | Gap analysis | 🔄 NEEDS UPDATE |
| `docs/session_logs/swot_analysis.md` | Strategic analysis | ✅ CURRENT |

---

## The Problem

**Documentation-Code Mismatch:** SoT v2.3 describes features not yet implemented:
- R5 carbon tracking (v2.3 headline feature) — 0% implemented
- skill_21 retraining loop — stubbed placeholder
- skill_12 composite variance — single-target only
- FeatureExtractor ABC — plugins use ad-hoc pattern

s
- Hardcoded targets in skill_07 — A5 violation

**Impact:** Developers implementing from SoT will write code that doesn't integrate with actual orchestrator.

---

## The Solution

### 3-Phase Refactor

**Phase 1: Critical Fixes (Week 1)**
- Fix hardcoded targets (DRIFT-1)
- Implement composite fold variance (GAP-2)
- Implement R5 carbon tracking (v2.3 feature)

**Phase 2: High-Priority Gaps (Week 2)**
- Implement skill_21 retraining loop (GAP-1)
- Create FeatureExtractor ABC (DRIFT-2)

**Phase 3: Documentation Sync (Week 2)**
- Update AGENTS.md with R5 section
- Mark GAP-1, GAP-2 as RESOLVED in SoT
- Add v2.3.1 remediation to audit report

---

## Priority Matrix

```
┌─────────────────────────────────┬─────────────────────────────────┐
│  HIGH IMPACT + HIGH URGENCY     │  HIGH IMPACT + LOW URGENCY      │
│  (DO FIRST)                     │  (SCHEDULE)                     │
├─────────────────────────────────┼─────────────────────────────────┤
│  • DRIFT-1: Hardcoded targets   │  • GAP-1: skill_21 retraining   │
│  • GAP-2: Composite variance    │  • DRIFT-2: FeatureExtractor ABC│
│  • R5: Carbon tracking          │  • Documentation updates        │
└─────────────────────────────────┴─────────────────────────────────┘
┌─────────────────────────────────┬─────────────────────────────────┐
│  LOW IMPACT + HIGH URGENCY      │  LOW IMPACT + LOW URGENCY       │
│  (DELEGATE)                     │  (DEFER)                        │
├─────────────────────────────────┼─────────────────────────────────┤
│  • DRIFT-3: Orchestrator warnings│ • GAP-3: SHAP interaction rule  │
│  • Config alias cleanup         │  • Phase architecture redesign  │
└─────────────────────────────────┴─────────────────────────────────┘
```

---

## Implementation Order

### Day 1-2: DRIFT-1 (Hardcoded Targets)
**Why First:** Blocking for geoai competition (composite metric requires dynamic target resolution)

**Files:**
- `zindian/skills/skill_07_features.py` (L1006-L1007)

**Test:**
- `test_a5_compliance.py`

**Estimated Time:** 2 hours

---

### Day 3-4: GAP-2 (Composite Fold Variance)
**Why Second:** Required for multi-target stability gating

**Files:**
- `zindian/skills/skill_12_metric.py` (L48-L82)

**Test:**
- `test_multi_target_composite_variance.py`

**Estimated Time:** 4 hours

---

### Day 5-7: R5 (Carbon Tracking)
**Why Third:** v2.3 headline feature, affects all skills

**Files:**
- `zindian/carbon_tracker.py` (NEW)
- `zindian/orchestrator.py` (run_skill wrapper)
- `zindian/skills/skill_02_intake.py` (infrastructure block)
- 8 mandatory skills (instrumentation)

**Test:**
- `test_r5_carbon_tracking.py`

**Estimated Time:** 8 hours

---

### Week 2: GAP-1 + DRIFT-2
**Why Fourth:** High-priority but not blocking current competition

**Files:**
- `skill_21_pseudo_label.py` (retraining loop)
- `plugins/base_extractor.py` (NEW ABC)
- `plugins/geoai_extractor.py` (migration)
- `plugins/world_cup_extractor.py` (migration)

**Tests:**
- `test_pseudo_label_retraining.py`
- `test_plugin_contract.py`

**Estimated Time:** 12 hours

---

### Week 2: Documentation Sync
**Why Last:** Update docs after code is stable

**Files:**
- `AGENTS.md` (R5 section, gap status)
- `docs/source_of_truth.md` (Known Gaps section)
- `docs/sot_audit_report.md` (v2.3.1 remediation)

**Estimated Time:** 4 hours

---

## Success Metrics

### Code Quality
- ✅ Zero A5 violations (no hardcoded strings)
- ✅ Test coverage ≥ 92%
- ✅ All plugins inherit from ABC
- ✅ R5 telemetry in 8 mandatory skills

### Documentation
- ✅ AGENTS.md reflects v2.3.1
- ✅ SoT Known Gaps current
- ✅ Audit report includes remediation

### Competition Readiness
- ✅ geoai runs end-to-end
- ✅ Composite metric computed correctly
- ✅ Carbon tracking in skill_22 report

---

## Risk Management

| Risk | Mitigation |
|------|------------|
| R5 breaks existing competitions | Fallback to `not_instrumented` if CodeCarbon unavailable |
| skill_21 degrades OOF | Gate condition 3 blocks weak augmented models |
| Plugin ABC breaks extractors | Backward-compatible migration, test all plugins |
| Documentation drift continues | Update docs concurrently with code |

---

## Decision Framework

**When to proceed with a fix:**
1. ✅ Gap confirmed in audit report
2. ✅ Test written before implementation
3. ✅ SoT section identified for update
4. ✅ Backward compatibility verified

**When to defer:**
1. ❌ Requires phase architecture redesign (GAP-3)
2. ❌ No clear SoT specification
3. ❌ Breaking change without migration path
4. ❌ Low impact + low urgency

---

## Next Actions

### Immediate (Today)
1. Review this summary with team
2. Confirm priority order
3. Start DRIFT-1 implementation

### This Week
1. Complete Phase 1 (DRIFT-1, GAP-2, R5)
2. Run test suite (target: 215 passing)
3. Validate geoai competition end-to-end

### Next Week
1. Complete Phase 2 (GAP-1, DRIFT-2)
2. Update all documentation
3. Archive refactor session logs

---

## Communication Plan

### Daily Standups
- Progress on current phase
- Blockers identified
- Test results

### Weekly Reviews
- Phase completion status
- Documentation sync check
- Competition validation results

### Completion Report
- Final test metrics
- Documentation updates
- Lessons learned

---

**Maintained by:** [whoisorioki](https://github.com/whoisorioki)  
**Last Updated:** June 26, 2026  
**Next Review:** June 27, 2026 (after DRIFT-1 completion)
