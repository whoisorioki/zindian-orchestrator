# Test Execution Guide

**Purpose:** Validate orchestrator refactor against SoT v2.2.1 specifications  
**Test Suite:** `/tests/test_orchestrator_refactor.py`  
**Coverage:** 5 critical integration tests

---

## Prerequisites

```bash
# Install test dependencies
pip install pytest pytest-cov

# Ensure project structure
cd /home/sagemaker-user/shared/zindian-orchestrator
```

---

## Test Execution

### Run All Tests
```bash
pytest tests/test_orchestrator_refactor.py -v --tb=short
```

### Run Individual Test Classes

**Test 1: Preflight Mode Check**
```bash
pytest tests/test_orchestrator_refactor.py::TestPreflightModeCheck -v
```

**Test 2: Dependency Chain Enforcement**
```bash
pytest tests/test_orchestrator_refactor.py::TestDependencyChainEnforcement -v
```

**Test 3: skill_03 Split Contract**
```bash
pytest tests/test_orchestrator_refactor.py::TestSkill03SplitContract -v
```

**Test 4: Plugin ABC Contract**
```bash
pytest tests/test_orchestrator_refactor.py::TestPluginABCContract -v
```

**Test 5: Single-Target Baseline**
```bash
pytest tests/test_orchestrator_refactor.py::TestSingleTargetBaseline -v
```

---

## Expected Results

### ✅ PASS Criteria

**Test 1:** INIT mode bypasses schema checks, ENFORCE mode validates strictly  
**Test 2:** Phase 2B blocked without 2A complete, Phase 3B blocked without 3A complete  
**Test 3:** policy_writer in Phase 1, policy_gate first in Phase 2A, split functions callable  
**Test 4:** ABC raises TypeError without extract_features(), hardcoded strings detectable  
**Test 5:** Single-target config works unchanged, skill_06 MCAR fallback functional  

### ❌ FAIL Scenarios

**Test 1 Fails:** Preflight mode detection broken, schema validation incorrect  
**Test 2 Fails:** Phase dependencies not enforced, out-of-order execution allowed  
**Test 3 Fails:** skill_03 split not working, policy_gate not first in Phase 2A  
**Test 4 Fails:** ABC not enforcing contract, hardcoded strings not caught  
**Test 5 Fails:** Single-target baseline broken, backward compatibility lost  

---

## Manual Validation Steps

### Step 1: Preflight INIT Mode
```bash
# Delete config to trigger INIT mode
rm competitions/test-competition/challenge_config.json

# Run Phase 1
python -c "from zindian.orchestrator import run_phase; print(run_phase('1'))"

# Verify: Phase 1 completes, config generated
ls competitions/test-competition/challenge_config.json
```

### Step 2: Phase Dependency Blocking
```bash
# Manually set incomplete state
echo '{"phase_1_complete": true, "phase_2a_complete": false}' > SKILL_STATE.json

# Attempt Phase 2B (should fail)
python -c "from zindian.orchestrator import run_phase; print(run_phase('2B'))"

# Expected output: "Phase 2B blocked: Phase 2A must complete first"
```

### Step 3: skill_03 Split Functions
```bash
# Test policy_writer
python -c "from zindian.orchestrator import run_skill; print(run_skill('skill_03.policy_writer'))"

# Test policy_gate
python -c "from zindian.orchestrator import run_skill; print(run_skill('skill_03.policy_gate'))"

# Verify: Both execute without error
```

### Step 4: Plugin ABC Enforcement
```python
# Create test file: test_plugin_abc.py
from plugins.base_extractor import FeatureExtractor

class InvalidPlugin(FeatureExtractor):
    pass  # Missing extract_features()

try:
    plugin = InvalidPlugin()
    print("FAIL: ABC not enforcing contract")
except TypeError as e:
    print(f"PASS: ABC enforced - {e}")
```

### Step 5: Single-Target Baseline
```bash
# Run full pipeline on single-target competition
python -c "
from zindian.orchestrator import run_phase
for phase in ['1', '2A', '2B', '3A', '3B', '4']:
    result = run_phase(phase)
    print(f'Phase {phase}: {result.get(\"status\", \"OK\")}')
"

# Verify: All phases complete successfully
```

---

## Coverage Report

```bash
# Generate coverage report
pytest tests/test_orchestrator_refactor.py --cov=zindian.orchestrator --cov-report=html

# View report
open htmlcov/index.html
```

---

## Troubleshooting

### Import Errors
```bash
# Ensure PYTHONPATH includes project root
export PYTHONPATH=/home/sagemaker-user/shared/zindian-orchestrator:$PYTHONPATH
```

### State File Conflicts
```bash
# Clean state between tests
rm competitions/*/SKILL_STATE.json
```

### Plugin Import Errors
```bash
# Verify plugin structure
ls -la plugins/base_extractor.py
ls -la plugins/nedbank_extractor.py
```

---

## Success Criteria Summary

| Test | Criterion | Status |
|------|-----------|--------|
| Test 1 | Preflight mode detection | ⏳ Pending |
| Test 2 | Phase dependency enforcement | ⏳ Pending |
| Test 3 | skill_03 split contract | ⏳ Pending |
| Test 4 | Plugin ABC enforcement | ⏳ Pending |
| Test 5 | Single-target baseline | ⏳ Pending |

**Overall Status:** READY FOR EXECUTION

---

## Next Steps After Tests Pass

1. Update audit report with RESOLVED status
2. Create migration guide for existing competitions
3. Document phase execution API changes
4. Begin multi-target implementation (A11/A12)

---

**Last Updated:** 2026-06-17  
**Test Suite Version:** 1.0  
**Orchestrator Version:** v2.2.1-refactored
