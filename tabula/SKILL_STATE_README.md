# Skill State Management

Automatic score externalization to prevent memory issues with large SKILL_STATE.json files.

## Problem
SKILL_STATE.json files can grow to 40,000+ lines with embedded score arrays, causing:
- Memory exhaustion when reading
- Slow file operations
- Git bloat

## Solution
Automatically externalizes score arrays >100 items to `scores/` directory.

## Quick Start

### 1. Migrate Existing Competition
```bash
python refactor_skill_state.py path/to/SKILL_STATE.json
```

### 2. Update Your Code
```python
# OLD:
import json
with open("SKILL_STATE.json") as f:
    state = json.load(f)
with open("SKILL_STATE.json", "w") as f:
    json.dump(state, f, indent=2)

# NEW:
from tabula.skill_state import read_state, write_state
state = read_state("SKILL_STATE.json")
write_state("SKILL_STATE.json", state)
```

### 3. For New Competitions
Just use `read_state`/`write_state` from the start - it handles both formats automatically.

## API

```python
from tabula.skill_state import read_state, write_state

# Read (auto-loads externalized scores)
state = read_state("SKILL_STATE.json")

# Write (auto-externalizes large arrays >100 items)
write_state("SKILL_STATE.json", state)
```

## File Structure
```
competition/
├── SKILL_STATE.json          # 157 lines (5 KB) ✓
└── scores/
    ├── branch_anchor-baseline_oof.json
    ├── branch_variant-06_oof.json
    └── ...
```

## Results
- **Before:** 42,040 lines (1.1 MB) - kills memory
- **After:** 157 lines (5 KB) - 99.5% reduction ✓

## Backward Compatibility
✓ Reads both old (embedded) and new (externalized) formats
✓ Automatically converts on save
✓ No breaking changes
