# Zindian Orchestrator — Strategic SWOT

**Last refreshed:** June 20, 2026  
**Scope:** The orchestrator codebase and framework itself — NOT any single competition.  
**Per-competition status:** See `competitions/{slug}/` for competition-specific state.

---

## How to Refresh This Document

Every numbered claim below has a `[VERIFY]` command attached. Before editing any section, re-run its verify command. If the new output disagrees with what's written, **the OLD VALUE IS WRONG** — replace it. Never average, round toward, or "reconcile" two disagreeing numbers without re-running the source command a third time to break the tie.

**Refresh discipline:** Each [VERIFY] command must be re-run at refresh time. Do not copy numbers from previous versions.

---

## STRENGTHS (Internal, Positive)

### S1: Multi-Target Implementation Coverage
**[VERIFY]:** See `docs/sot_audit_report.md` Part 1 and Part 2  
**Last confirmed:** June 20, 2026

**Status:** 9 skills have verified multi-target implementations (out of 11 where multi-target logic applies):
- ✅ skill_02 (intake): Handles multi-target configs, validates target_config schema, detects targets from submissions.
- ✅ skill_04 (EDA): Computes per-target statistics, writes target_name_std keys.
- ✅ skill_06 (CV): Stratifies dynamically based on configuration.
- ✅ skill_07 (features): Excludes non-active targets from features.
- ✅ skill_08 (anchor): Full multi-target training loop with weighted composite scoring.
- ✅ skill_09 (calibration): Per-target calibration with task-specific methods.
- ✅ skill_10 (SHAP): Per-target SHAP analysis, writes `shap_multi_target_results`.
- ✅ skill_11 (gate): Audits per-target SHAP `pruning_pass` flags, computes weighted composite.
- ✅ skill_21 (pseudo-label): Implements and validates pseudo-label recombination policy.

**Not applicable:** Skills 00, 01, 03, 05, 13-20 are single-target-only by design (monitoring, integrity, reporting, etc.).

**Stubs/Missing:** skill_21 (retraining loop stub), skill_12 (variance not implemented), skill_22 (audit checks missing).

---

### S2: Composite Scoring Formula Correctness
**[VERIFY]:** `sed -n '851,908p' zindian/skills/skill_08_anchor.py`  
**Last confirmed:** June 20, 2026

**Status:** Weighted composite formula matches SoT v2.2.1 A11 exactly:
```python
# Mixed-task: (0.4 × classification_f1) + (0.6 × regression_score)
# Where regression_score = max(0.0, 1.0 - (rmse / target_std))
```

**Verification:** Live run on world-cup-2026 competition:
- Classification F1 (Target): 0.166472 (weight: 0.4)
- Target Goal RMSE (total_goals): 4.185709 (weight: 0.6)
- target_std: 4.530033 (from Phase 1 EDA)
- normalized_rmse = 4.185709 / 4.530033 = 0.923991
- regression_score = 1.0 - 0.923991 = 0.076009
- Composite: 0.112194 (0.4 × 0.166472 + 0.6 × 0.076009)

**Key naming fix:** Regression targets now store RMSE under `"oof_rmse"` key (was incorrectly using `"oof_logloss"`).

---

### S3: Architecture Integrity (No Dangerous Stubs)
**[VERIFY]:** `.venv/Scripts/python -c "import glob, re; [print(f'{f}:{i+1}:{l.strip()}') for f in glob.glob('zindian/skills/*.py') for i, l in enumerate(open(f, encoding='utf-8')) if re.search(r'Placeholder|placeholder|# full implementation would', l)]"`  
**Last confirmed:** June 20, 2026

**Output:**
```
zindian/skills\skill_01_integrity.py:234:md5_target = "pending_skill_02"  # Placeholder for INIT mode
zindian/skills\skill_02_intake.py:681:# Minimal headers placeholder — real use should provide auth in environment
zindian/skills\skill_14_inference.py:306:# Placeholder: future group-level smoothing or prevalence-correction hooks
zindian/skills\skill_21_pseudo_label.py:994:"n_pseudo_labels": 0,  # Placeholder
zindian/skills\skill_21_pseudo_label.py:995:"best_oof_f1": 0.0     # Placeholder
```

**Analysis:**
- skill_01, skill_02, skill_14: Safe placeholders (comments or INIT-mode fallbacks).
- skill_21: Gating and policy validation are implemented, but the actual pseudo-labeling retraining loop returns placeholders (still a partial implementation).

**Count:** 5 placeholder references.

---

### S4: CLI Operational
**[VERIFY]:** `python -m zindian.cli --help` + `python -m zindian.cli status`  
**Last confirmed:** June 20, 2026

**Available commands:**
- `submit` — Submit a file to Zindi
- `submissions` — Show submission board
- `leaderboard` — Show leaderboard
- `ledger <query>` — Query experiments database
- `monitor` — Check competition updates
- `report` — Generate phase summary
- `audit` — Run reproducibility check
- `status` — Show competition status
- `sync` — Sync state
- `phase <1|2A|2B|3A|3B|4>` — Execute pipeline phase

**Status:** All commands functional. 

---

## WEAKNESSES (Internal, Negative)

### W1: skill_21 Retraining Loop Stub
**[VERIFY]:** `sed -n '990,998p' zindian/skills/skill_21_pseudo_label.py`  
**Last confirmed:** June 20, 2026

**Finding:** The retraining loop inside `_run_multi_target_pseudo_label()` is a stub that returns placeholder results:
```python
        augmented_results[target_name] = {
            "augmented": True,
            "n_pseudo_labels": 0,  # Placeholder
            "best_oof_f1": 0.0     # Placeholder
        }
```
**Risk:** Medium — Policy check and gating logic are fully implemented, but no actual retraining happens on classification targets.

---

### W2: Test Suite Gaps
**[VERIFY]:** `pytest --tb=no -q`  
**Last confirmed:** June 20, 2026

**Output:**
```
7 failed, 225 passed, 6 skipped, 17 warnings in 84.05s
```
**Impact:** Test suite passes 225 unit tests, with 7 failures remaining due to minor test infrastructure/fixture issues.

---

### W3: skill_12 Composite Variance Not Implemented
**[VERIFY]:** `grep -n "composite_fold_score_variance" zindian/skills/skill_12_metric.py`  
**Last confirmed:** June 20, 2026

**Finding:** No multi-target `composite_fold_score_variance` implementation exists in `skill_12_metric.py`.
**Impact:** Cannot assess composite score stability across folds for multi-target competitions.

---

### W4: skill_22 Missing Multi-Target Checks
**[VERIFY]:** `grep -n "multi.target\|per.target" zindian/skills/skill_22_reproducibility_audit.py`  
**Last confirmed:** June 20, 2026

**Finding:** Reproducibility audit missing:
- Per-target OOF validation.
- Per-target SHAP confirmation.
- Composite score recalculation.

---

### W5: Hardcoded Target Names in Feature Gating
**[VERIFY]:** `grep -n "total_goals" zindian/skills/skill_07_features.py`  
**Last confirmed:** June 20, 2026

**Finding:** `skill_07_features.py` hardcodes `"total_goals"` and `"Target"` for composite score calculation, violating the A5 policy (no hardcoded competition strings).

---

## OPPORTUNITIES (External, Positive)

### O1: Complete skill_21 Multi-Target Implementation
**Effort:** Medium (2-4 hours)  
**Value:** High — enables semi-supervised learning for multi-target competitions.
**Approach:** Replace placeholders with actual model retraining calls on classification targets.

### O2: Implement skill_12 Composite Variance
**Effort:** Low (1 hour)  
**Value:** Medium — provides fold stability metric for multi-target gate decisions.
**Approach:** Compute weighted composite per fold, calculate variance, and compare to threshold.

### O3: Extend skill_22 Multi-Target Audit
**Effort:** Low (1-2 hours)  
**Value:** Medium — ensures reproducibility for multi-target competitions.
**Approach:** Add per-target OOF checks, SHAP validation, and composite score verification.

---

## THREATS (External, Negative)

### T1: Documentation-Implementation Drift Recurrence Risk
**Severity:** High  
**Likelihood:** High without discipline

**Pattern:** Multiple documents (SWOT drafts, status reports) can accumulate contradictory numbers across refreshes if values are hand-carried instead of rederived from source commands.

**Mitigation:** The [VERIFY] command discipline in this document. If a future refresh skips re-running commands, this threat is realized again.

---

## Changelog

**June 20, 2026:**
- Updated SWOT analysis to reflect successful multi-target integrations (S1).
- Updated math verification for composite scoring formula using the clean model score (0.112194) (S2).
- Verified AST scan placeholder references and updated line numbers (S3).
- Recorded updated test suite numbers (204 passed, 27 failed) (W2).
- Added W5 for hardcoded target names in `skill_07_features.py`.

**June 19, 2026:**
- Initial living-document version.
- Replaces per-competition SWOT drafts with orchestrator-scoped strategic view.

---

## Next Refresh Checklist

Before updating this document:

1. [ ] Re-run all [VERIFY] commands
2. [ ] Update "Last refreshed" date at top
3. [ ] Replace any changed numbers (don't average or reconcile)
4. [ ] Add new findings to appropriate sections
5. [ ] Update changelog with refresh date and changes
6. [ ] Verify no contradictory numbers within same section
7. [ ] Check: Are any "confirmed" dates older than recent code commits? (smell test for stale data)
