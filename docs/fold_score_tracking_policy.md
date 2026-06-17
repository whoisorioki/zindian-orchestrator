# Fold Score Tracking Policy

**Version:** 1.0  
**Effective:** 2026-06-17  
**Scope:** All OOF-generating skills (07, 08, 09, 21)

---

## Problem Statement

The score externalization architecture (`scores/branch_*_oof.json`) prevents auditing of fold-level performance metrics without file I/O. The SWOT verification audit identified 3 unverifiable claims due to missing fold scalars in `SKILL_STATE.json`.

---

## Solution: Dual-Write Pattern

**Principle:** Externalize massive arrays, internalize audit scalars.

### Write Pattern

Every OOF-generating skill MUST write:

1. **External file** (`scores/branch_{name}_oof.json`):
   - Full row-level predictions (N×1 array)
   - Used for: inference, ensemble blending, detailed analysis

2. **Internal state** (`SKILL_STATE.json`):
   - Fold-level RMSLE scalars (5×1 array)
   - Used for: auditing, variance calculation, gate evaluation

### Schema

```json
{
  "branch_{name}_oof": {
    "scores_file": "scores/branch_{name}_oof.json",
    "fold_scores": [0.5747, 0.5494, 0.5544, 0.5545, 0.5391],
    "fold_variance": 0.00016767596242461537,
    "mean_score": 0.5544,
    "cv_strategy_id": "config:KFold",
    "seed": 42,
    "branch_name": "anchor-baseline"
  }
}
```

### Implementation

**File:** `zindian/skills/_oof_writer.py` (or equivalent)

```python
def write_oof_record(
    branch_name: str,
    predictions: np.ndarray,
    fold_scores: list[float],
    cv_strategy_id: str,
    seed: int
) -> None:
    """
    Dual-write OOF record: externalize predictions, internalize scalars.
    
    Args:
        branch_name: Branch identifier
        predictions: Full row-level predictions (externalized)
        fold_scores: Fold-level RMSLE scalars (internalized)
        cv_strategy_id: CV strategy tag
        seed: Reproducibility seed
    """
    import json
    import statistics
    from pathlib import Path
    
    # External write: massive array
    scores_dir = Path("scores")
    scores_dir.mkdir(exist_ok=True)
    scores_file = scores_dir / f"branch_{branch_name}_oof.json"
    with open(scores_file, 'w') as f:
        json.dump(predictions.tolist(), f)
    
    # Internal write: audit scalars
    state_record = {
        "scores_file": str(scores_file),
        "fold_scores": fold_scores,
        "fold_variance": statistics.variance(fold_scores) if len(fold_scores) > 1 else 0.0,
        "mean_score": statistics.mean(fold_scores),
        "cv_strategy_id": cv_strategy_id,
        "seed": seed,
        "branch_name": branch_name
    }
    
    # Write to SKILL_STATE.json via state manager
    from zindian.state import SkillStateStore
    state = SkillStateStore.read()
    state[f"branch_{branch_name}_oof"] = state_record
    SkillStateStore.write(state)
```

---

## Affected Skills

| Skill | Current Behavior | Required Change |
|-------|------------------|-----------------|
| skill_08_anchor | Externalizes only | Add fold_scores to state |
| skill_07_features | Externalizes only | Add fold_scores to state |
| skill_09_calibration | Externalizes only | Add fold_scores to state |
| skill_21_pseudo_label | Externalizes only | Add fold_scores to state |

---

## Verification Command

```bash
# Verify fold scores present in state
python3 -c "
import json
state = json.load(open('competitions/june-study-jam-series-transaction-volume-forecasting-challenge/SKILL_STATE.json'))
for key in state:
    if 'oof' in key and isinstance(state[key], dict):
        if 'fold_scores' in state[key]:
            print(f'{key}: {state[key][\"fold_scores\"]}')
        else:
            print(f'{key}: MISSING fold_scores')
"
```

---

## Benefits

1. **Audit transparency**: Fold variance verifiable without file I/O
2. **Gate efficiency**: Variance check reads state, not disk
3. **Memory efficiency**: Row-level predictions remain externalized
4. **Reproducibility**: Fold scalars locked in state for history

---

## Migration Path

**Existing competitions:**
- No retroactive update required
- Policy applies to new OOF writes only

**New competitions:**
- Enforced via preflight check (A7 extension)
- Gate 1 blocks if fold_scores missing

---

**Sign-off:** Policy documented. Implementation deferred to next OOF write cycle.
