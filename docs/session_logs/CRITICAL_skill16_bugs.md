# CRITICAL: skill_16 Submission Tracking Bugs

**Date:** 2026-06-16  
**Severity:** HIGH  
**Impact:** Corrupted validation tracking, wrong submissions selected for final judging

---

## Bug 1: Gate 5 Selection Defiance

**Evidence:** Zindi platform shows submission `k1Y7WH52` (score 179.98) is checked for final judging  
**Expected:** Only `sub_010_anchor.csv` and `sub_009_anchor.csv` should be selected  
**Impact:** Catastrophic raw-scale model wasting 1 of 2 final slots

**Root Cause:** Platform selection != SKILL_STATE selection

---

## Bug 2: Stale OOF Comment Injection

**Evidence:** All 4 submissions have identical comment: `branch:main|oof_score:131.3773|features:28|calib:none`  
**Expected:** Each submission should have unique OOF score matching its model  
**Impact:** Cannot trace which model generated which submission

**Root Cause:** skill_16 reads stale global state key instead of branch-specific OOF

**Fix Required:**
```python
# Current (WRONG):
oof_score = state.get("anchor_oof_score")  # Stale global

# Fixed:
active_branch_oof = state.get(f"branch_{git_branch}_oof", {})
oof_score = active_branch_oof.get("scores", [None])[0] or state.get("anchor_oof_score")
```

---

## Bug 3: Pre-Increment Budget Drift

**Evidence:**  
- Zindi platform: 4 submissions  
- SKILL_STATE: 8 submissions  
- Discrepancy: 4 submissions

**Root Cause:** skill_16 increments budget BEFORE API call succeeds

**Fix Required:**
```python
# Current (WRONG):
store.update(submissions_used_total=total + 1)
result = client.submit(...)

# Fixed:
result = client.submit(...)
if result.get("status") == "success" or "id" in result:
    store.update(submissions_used_total=total + 1)
else:
    raise RuntimeError(f"API rejection: {result.get('error')}")
```

---

## Immediate Actions

1. **Manually fix Zindi selections:**
   - Uncheck `k1Y7WH52` (179.98 score)
   - Verify only log-scale models selected

2. **Patch skill_16_submit.py:**
   - Fix comment generation (Bug 2)
   - Fix budget increment timing (Bug 3)

3. **Sync platform state:**
   - Query live API for actual submission count
   - Update SKILL_STATE to match reality

---

## Code Patches

### Patch 1: Dynamic Comment Generation
Location: `zindian/skills/skill_16_submit.py` line ~425

```python
# Replace stale comment generation
branch = _branch_from_state(skill_state)
active_oof = skill_state.get(f"branch_{branch}_oof", {})
oof_scores = active_oof.get("scores", [])
oof_score = float(np.mean(oof_scores)) if oof_scores else skill_state.get("anchor_oof_score")
```

### Patch 2: Safe Budget Mutation
Location: `zindian/skills/skill_16_submit.py` line ~455

```python
# Move state update AFTER successful API call
result = client.submit(filepath=str(sub_path), comment=comment)

if not result or result.get("error"):
    raise RuntimeError(f"Submission failed: {result}")

# Only increment if API succeeded
store.update(
    submissions_used_today=used_today + 1,
    submissions_used_total=total + 1,
    remaining_submissions=client.remaining_submissions
)
```

---

## Verification Steps

1. Check Zindi platform selections match SKILL_STATE
2. Verify next submission has correct OOF in comment
3. Confirm budget only increments on API success
