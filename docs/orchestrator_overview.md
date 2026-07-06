# Zindian Orchestrator - Complete Overview

**Version:** 2.3
**Last Updated:** June 2026
**Status:** Production Ready

---

## Table of Contents

1. [Non-Technical Overview](#non-technical-overview)
2. [Technical Deep Dive](#technical-deep-dive)
3. [Quick Reference](#quick-reference)

---

# Non-Technical Overview

## What is Zindian Orchestrator?

Zindian Orchestrator is an **intelligent assistant for tabular machine learning competitions on the Zindi platform.** Think of it as an experienced data scientist that follows a strict, repeatable playbook to systematically ingest raw data, engineer features, train models, audit validation folds, and manage submissions.

### The Problem It Solves

Imagine entering a cooking competition where you must:
1.  Understand recipe requirements
2.  Verify ingredients are fresh
3.  Follow food safety regulations
4.  Try different cooking techniques
5.  Taste-test everything
6.  Submit your best dish
7.  Document every step

The Orchestrator does exactly this for **data science competitions** - it takes raw data and systematically builds, tests, and submits machine learning models while following strict rules.

---

## Core Philosophy: The "Three Lenses"

Every decision is evaluated through three simultaneous perspectives:

| Lens | Question | Example |
|------|----------|---------|
| **General** | What does research say? | "Classification problems typically need stratified sampling" |
| **Specific** | What does THIS data show? | "This dataset has severe class imbalance" |
| **Generalization** | Will this work on new data? | "Will this pattern hold on unseen test data?" |


---

## The Journey: 4 Main Phases

### Phase 1: Understanding the Competition [SEARCH]
**What happens:** Reads competition rules, examines data structure, locks configuration
**Duration:** ~5 minutes
**Output:** Locked `challenge_config.json` (the rulebook)

**Key Activities:**
- Hash all data files (detect tampering)
- Identify task type (classification/regression)
- Detect temporal/spatial/group patterns
- Select cross-validation strategy
- Lock configuration (no changes allowed after this)

---

### Phase 2: Building the Baseline [ANCHOR]
**What happens:** Creates a simple, reliable model as starting point
**Duration:** ~15-30 minutes
**Output:** Anchor model with baseline score

**Key Activities:**
- Apply data cleaning rules
- Generate initial features
- Train anchor model
- **HUMAN GATE 1** - Review baseline (approve/reject/challenge)

---

### Phase 3: Testing & Validation [TEST]
**What happens:** Checks for data leakage, tests stability, validates predictions
**Duration:** ~30-60 minutes
**Output:** Approved models that passed all safety checks

**Key Activities:**
- SHAP analysis (detect target leakage)
- Calibration (for classification)
- Fold variance analysis
- Gate evaluation (5 conditions must pass)
- **HUMAN GATE 2** - Approve each variant
- **HUMAN GATE 3** - Approve fusion strategy

---

### Phase 4: Final Submission [SUBMIT]
**What happens:** Format predictions, submit, document everything
**Duration:** ~10 minutes
**Output:** Competition submission + complete audit trail

**Key Activities:**
- Format predictions to competition schema
- Validate submission file
- Submit to Zindi platform
- **HUMAN GATE 4** - Approve inference
- **HUMAN GATE 5** - Select final 2 submissions
- Generate reproducibility report

---

## Key Safety Features

### 1. Human Gates (5 Checkpoints) [GATE]

The system **stops and waits for human approval** at 5 critical points:

| Gate | Trigger | Decision |
|------|---------|----------|
| **Gate 1** | After anchor model | "Does baseline look reasonable?" |
| **Gate 2** | Before promoting variants | "Keep this model?" (per variant) |
| **Gate 3** | Before model fusion | "Ready to blend models?" |
| **Gate 4** | Before inference | "Generate final predictions?" |
| **Gate 5** | Before competition close | "Which 2 submissions to use?" |

**Why this matters:** Prevents autonomous decisions. Humans always have final say.

---

### 2. Reproducibility Contract 🔁

Everything must be repeatable. Run twice with same data = identical results.

**Requirements:**
-  Fixed random seed (set once, never changed)
-  Pinned dependencies (`requirements.txt`)
-  No custom packages
-  Complete audit trail
-  Submission reproducible from config + state alone

**Real-world analogy:** Scientific experiment - anyone following exact steps gets same results.

---

### 3. No "Black Box" AutoML 🚫

The system doesn't use automated ML tools that make unexplainable decisions.

**What's allowed:**
-  LightGBM, XGBoost, scikit-learn
-  Manual hyperparameter tuning
-  Documented feature engineering

**What's banned:**
-  AutoML libraries (H2O, TPOT, Auto-sklearn)
-  Neural architecture search
-  Automated feature selection without documentation

**Why this matters:** Competition rules often ban AutoML. This system is compliant by design.

---

## Special Features

### Carbon Tracking (R5) [CARBON]

Measures and reports environmental impact (CO2 emissions) of model training.

**Metrics tracked:**
- Duration (seconds)
- Peak memory (MB)
- Carbon estimate (kg CO2)
- Hardware type (CPU/GPU)
- Region (for carbon intensity)

**Real-world analogy:** Like a car showing fuel efficiency - understand environmental cost.

---

### Pseudo-Labeling (Skill 21) [LABEL]

For classification: uses confident predictions on test data to expand training set.

**Guard conditions (all must pass):**
1. Classification task only
2. Not time-series data
3. No leaked features detected
4. Low fold variance
5. Calibrated probabilities available
6. Confidence threshold met (top 10%)

**Real-world analogy:** Teacher using confident student answers as teaching examples.

---

### Multi-Target Support [TARGET]

Handles competitions predicting multiple targets simultaneously.

**Example:** Predict both temperature AND humidity from weather data.

**How it works:**
- Trains separate model per target
- Computes weighted composite score
- Normalizes by target standard deviation
- Single gate decision for all targets

---

## What Makes This Valuable?

### For Data Scientists
-  **Consistency:** Same rigorous process every time
-  **Safety:** Multiple checkpoints prevent costly mistakes
-  **Efficiency:** Automates repetitive tasks
-  **Learning:** Captures knowledge from past competitions

### For Managers
-  **Transparency:** Every decision documented
-  **Compliance:** Built to follow competition rules
-  **Risk Reduction:** Prevents common mistakes (leakage, overfitting)
-  **Audit Trail:** Complete documentation for review

### For Organizations
-  **Scalability:** Same framework for multiple competitions
-  **Knowledge Capture:** Cross-competition learning
-  **Cost Tracking:** Carbon and compute monitoring
-  **Quality Assurance:** Reproducible results

---

## What It's NOT

-  **Not magic AI** - Doesn't solve everything automatically
-  **Not autonomous** - Requires human approval at key points
-  **Not a replacement** - Tool for data scientists, not replacement
-  **Not a black box** - Everything is traceable and explainable

---

## Simple Metaphor

Think of it as a **highly organized research lab**:

| Lab Component | Orchestrator Equivalent |
|---------------|-------------------------|
| Protocols | Phases and rules |
| Quality control | Gates and validation |
| Lab notebooks | State and config files |
| Safety inspectors | Human gates |
| Standard procedures | Skills (22 modules) |
| Equipment | LightGBM, scikit-learn |

The lab runs experiments efficiently, but a human scientist must approve all major decisions and sign off on results.

---

## Success Metrics

### Competition Performance
- Consistent top-tier placements
- Reproducible results
- Zero disqualifications for rule violations

### Operational Efficiency
- 80% reduction in manual work
- Complete audit trail in <5 minutes
- Cross-competition knowledge reuse

### Risk Management
- Zero data leakage incidents
- 100% reproducible submissions
- Full compliance with platform rules

---

# Technical Deep Dive

## Architecture Overview

### Design Principles

1. **Competition Agnosticism** - Zero hardcoded competition-specific values
2. **Atomic State Management** - Tempfile + os.replace prevents corruption
3. **Immutable Config** - Locked after Phase 1, read-only thereafter
4. **Single Source of Truth** - `SKILL_STATE.json` for execution state
5. **Dependency Chain Enforcement** - Strict phase ordering

---

## Core Components

### State Management

```python
# SKILL_STATE.json - Execution state
{
  "dag_phase": "phase_3a_complete",
  "anchor_oof_score": 0.8234,
  "branch_variant-01_oof": {
    "scores": [0.82, 0.83, 0.81, 0.84, 0.82],
    "cv_strategy_id": "stratified_kfold",
    "seed": 42,
    "branch_name": "variant-01",
    "secondary_metrics": {
      "mae": 0.123,
      "mape": 4.56,
      "r2": 0.78
    }
  },
  "leaked_features": [],
  "human_gate_1_approved": true
}
```

**Key features:**
- Atomic writes (tempfile + os.replace)
- JSON schema validation
- Safe nested reads with defaults
- Never holds state between runs

---

### Configuration Management

```python
# challenge_config.json - Competition contract
{
  "competition_id": "example-competition",
  "task_type": "classification",
  "metric": "logloss",
  "metric_direction": "minimize",
  "target_col": "target",
  "cv_strategy": {
    "type": "StratifiedKFold",
    "n_splits": 5,
    "shuffle": true,
    "random_state": 42,
    "selection_reason": "imbalanced_classification"
  },
  "reproducibility": {"seed": 42},
  "shap_leak_threshold": 3.0,
  "variance_gate_threshold": 0.01,
  "gate_margin": 0.001
}
```

**Temporal boundary rule:**
- **Phase 1:** Mutable (skills 01-05 write)
- **Post-Phase 1:** Immutable (read-only, except skill_00 community signals)

---

## Phase Architecture

### Phase 1: Competition Fingerprint

**Skills:** 01 → 02 → 03 → 04 → 05 → 15

**Outputs:**
- File hashes (MD5) locked
- Task type, metric, target identified
- CV strategy selected (decision tree)
- Policy filters written
- EDA diagnostics computed
- Config locked (immutable)

**CV Strategy Decision Tree:**
```
IF temporal_signal_confirmed:
    → TimeSeriesSplit (prevent look-ahead bias)
ELIF group_signal OR spatial_signal:
    → GroupKFold (prevent group leakage)
ELIF classification AND minority_ratio < 0.15:
    → StratifiedKFold (minority class stability)
ELSE:
    → KFold (standard fallback)
```

---

### Phase 2: Anchor + Feature Search

**Skills:** policy_gate() → 06 → 08 → 07

**Phase 2A (Cleaning):**
- Enforce feature exclusions
- MNAR indicators (before fill)
- MCAR imputation (median/mode)
- Drop constant columns

**Phase 2B (Signal Search):**
- Train anchor model
- **HUMAN GATE 1** (approve/reject/challenge)
- Generate feature variants
- Tag all OOF with cv_strategy_id

**Anchor Contract:**
- Reads CV strategy from config (never defines own)
- Uses config seed for all training
- Writes OOF with cv_strategy_id tag
- Immutable comparison baseline

---

### Phase 3: Generalization Audit

**Skills:** 10 → 09 → 12 → 11 → 21 → 13

**Phase 3A (Audit):**
- SHAP leak detection (per-fold OOF only)
- Calibration (classification only)
- Fold variance analysis (ddof=1)

**Phase 3B (Promotion):**
- Gate evaluation (5 conditions)
- **HUMAN GATE 2** (per variant)
- Pseudo-labeling (optional, classification only)
- Oracle fusion (diversity check)
- **HUMAN GATE 3** (before fusion)

---

### Phase 4: Governance

**Skills:** 14 → 16 → 17 → 22

**Activities:**
- Format predictions (task-specific validation)
- **HUMAN GATE 4** (before inference)
- Submit to platform (budget guard)
- **HUMAN GATE 5** (final selection)
- Reproducibility audit (R1-R5)
- History log update

---

## The OOF Contract

**Universal validation plan:**

```
skill_05_cv writes ONE CV strategy
    ↓
Every OOF-generating skill reads that strategy
(skills 07, 08, 09, 21)
    ↓
Tags every OOF output with cv_strategy_id
    ↓
Orchestrator validates tags before passing scores
    ↓
Every OOF-evaluating skill reads same strategy
(skills 10, 11, 12)
    ↓
skill_22 verifies full contract at sign-off
```

**Contract violation = hard halt (not warning)**

---

## Gate Logic (skill_11)

All 5 conditions must pass:

### Condition 1: No Leaked Features
```python
branch not in SKILL_STATE["leaked_features"]
```

### Condition 2: Fold Variance Within Threshold
```python
# Scale-invariant normalization
if task_type == "regression":
    if metric == "rmsle":
        effective_threshold = variance_gate_threshold  # No scaling
    else:
        effective_threshold = variance_gate_threshold * (target_std ** 2)
else:
    effective_threshold = variance_gate_threshold  # Bounded metrics

fold_score_variance < effective_threshold
```

### Condition 3: OOF Improvement Over Baseline
```python
# Baseline selection (precedence order)
if pseudo_label_result.retraining_required:
    baseline = anchor_oof_score_augmented
elif anchor_challenge.active:
    baseline = anchor_oof_score_challenged
else:
    baseline = anchor_oof_score

# Scale-invariant margin
if task_type == "regression":
    if metric == "rmsle":
        effective_margin = gate_margin  # No scaling
    else:
        effective_margin = gate_margin * target_std
else:
    effective_margin = gate_margin

# Directional check
if metric_direction == "maximize":
    improvement = oof_score - baseline > effective_margin
else:
    improvement = baseline - oof_score > effective_margin
```

### Condition 4: SHAP Audit Passed
```python
SKILL_STATE["shap_audit_passed"][branch] == True
```

### Condition 5: Human Approval
```python
SKILL_STATE[f"human_gate_2_{branch}_approved"] == True
```

---

## SHAP Leak Detection (skill_10)

**Contract:**
```python
for fold in cv_folds:
    # Train on training fold
    model.fit(X_train_fold, y_train_fold)

    # Compute SHAP on OOF predictions ONLY
    shap_values = shap.TreeExplainer(model).shap_values(X_val_fold)

    # Store per-fold arrays
    fold_shap_arrays.append(shap_values)

# Aggregate across folds
mean_abs_shap = np.mean(np.abs(fold_shap_arrays), axis=0)

# Threshold comparison
if mean_abs_shap[top_feature] > shap_leak_threshold * mean_abs_shap[second_feature]:
    flag_as_leaked(top_feature)
```

**CRITICAL:** Full-train SHAP is strictly prohibited (introduces target into computation).

---

## Pseudo-Labeling (skill_21)

**Guard conditions (all must pass):**

```python
gc1 = config["task_type"] == "classification"
gc2 = config["cv_strategy"]["type"] != "TimeSeriesSplit"
gc3 = len(SKILL_STATE["leaked_features"]) == 0
gc4 = fold_score_variance < effective_variance_threshold
gc5 = "calibrated_probs" in SKILL_STATE
gc6 = confidence_threshold_met(top_10_percent)

if all([gc1, gc2, gc3, gc4, gc5, gc6]):
    run_pseudo_labeling()
else:
    skip_with_reason()
```

**Retraining loop:**
1. Assign pseudo-labels to confident test rows
2. Append to training matrix
3. Retrain anchor on augmented dataset
4. Write `anchor_oof_score_augmented`
5. Retrain promoted variants
6. Write to `branch_{name}_oof_augmented` namespace
7. Re-gate against augmented anchor
8. If zero pass: rollback (clear _augmented keys)

**CV fold assignment contract:**
- Pseudo-labeled rows → training split of EVERY fold
- NEVER assigned to validation folds
- OOF indices unchanged from Phase 1

---

## Multi-Target Support

**Composite score computation:**

```python
weighted_distances = []
for target_spec in target_config["targets"]:
    raw_score = oof_score_for_target

    if target_spec["task_type"] == "regression":
        target_std = SKILL_STATE["eda"][f"{target_spec['name']}_std"]
        if target_spec["metric"] == "rmsle":
            normalized_distance = raw_score  # Already dimensionless
        else:
            normalized_distance = abs(raw_score) / target_std
    else:  # classification
        normalized_distance = 1.0 - raw_score

    weighted_distances.append(normalized_distance * target_spec["weight"])

composite_score = sum(weighted_distances)
```

**Composite variance threshold:**

```python
regression_targets = [t for t in targets if t["task_type"] == "regression"]
effective_target_std = sqrt(
    sum(w_i * sigma_i**2 for i in regression_targets)
    / sum(w_i for i in regression_targets)
)
effective_variance_threshold = variance_gate_threshold * (effective_target_std ** 2)
```

---

## Reproducibility Contract (R1-R5)

### R1: Seed Always Set
```python
random_state = config["reproducibility"]["seed"]
np.random.seed(config["reproducibility"]["seed"])
random.seed(config["reproducibility"]["seed"])
```

### R2: Rerun = Identical Output
Bit-identical OOF scores and submission files on rerun.

### R3: No Custom Packages
All packages in `requirements.txt` (pinned via pip-compile).

### R4: Submission Reproducible
Third party can regenerate exact submission from config + state.

### R5: Carbon Tracking
```python
SKILL_STATE["telemetry.{skill_name}"] = {
    "duration_sec": 123.45,
    "peak_memory_mb": 2048,
    "carbon_kg_estimate": 0.0012,
    "tracker_method": "mlco2_formula",
    "hardware_type": "cpu",
    "region": "us-east-1"
}
```

---

## Research Sidecar (Non-Blocking)

**Skills:** 00 (continuous) → 18, 19, 20 (triggered)

**Trigger schedule:**

| Skill | Trigger | Informs |
|-------|---------|---------|
| skill_00 | Competition start → close | All phases |
| skill_18 | Phase 1 complete | Feature generation |
| skill_19 | Phase 2A complete | Feature patterns |
| skill_20 | Phase 2B complete | Audit + next variants |

**Consumption pattern (mandatory):**
```python
sidecar_recommendations = SKILL_STATE.get("sidecar_recommendations", default=[])
if not sidecar_recommendations:
    log("No sidecar - proceeding from fingerprint")
else:
    log(f"Sidecar consumed: {len(sidecar_recommendations)} items")
```

**Sidecar failure never halts main pipeline.**

---

## Preflight Validation

### INIT Mode
**Triggered:** `challenge_config.json` doesn't exist
**Permits:** Phase 1 skills only (01-05, 15)
**Checks:** Workspace, data files, environment lock, no AutoML imports

### ENFORCE Mode
**Triggered:** `challenge_config.json` exists
**Checks:**
- Config completeness
- File hash integrity
- OOF contract compliance
- Architecture integrity
- Human gate status
- Zindi compliance

---

## Key Technical Patterns

### Atomic State Updates
```python
def update(self, **kwargs):
    temp_path = self.state_path.with_suffix(".tmp")
    with open(temp_path, "w") as f:
        json.dump(updated_state, f, indent=2)
    os.replace(temp_path, self.state_path)  # Atomic
```

### Safe Nested Reads
```python
# CORRECT
override_active = SKILL_STATE.get("cv_strategy_override", {}).get("active", False)

# WRONG (KeyError if key missing)
override_active = SKILL_STATE["cv_strategy_override"]["active"]
```

### Target-Dependent Features (Two-Mode Contract)
```python
# During CV validation
spatial_lag = compute_spatial_lag(X_train_fold, y_train_fold)

# During final inference
spatial_lag = compute_spatial_lag(X_train_full, y_train_full)
```

---

## Performance Characteristics

### Typical Timeline
- Phase 1: 5 minutes
- Phase 2: 30-60 minutes
- Phase 3: 60-120 minutes
- Phase 4: 10 minutes
- **Total:** 2-3 hours per competition

### Resource Usage
- Memory: 2-8 GB (depends on dataset size)
- CPU: 4-8 cores recommended
- Storage: 1-5 GB per competition
- Carbon: ~0.01-0.05 kg CO2 per run

---

## Known Limitations

### C1: Bootstrap Phase String Mismatch
**Issue:** `bootstrap_competition.py` sets `dag_phase = "phase_1_integrity_locked"` but skills expect `"phase_1_integrity"`.
**Impact:** Config writes silently fail after bootstrap.
**Workaround:** Add `"phase_1_integrity_locked"` to `allowed_write_phases`.

### GAP-3: SHAP Interaction Features
**Issue:** Phase architecture doesn't support SHAP-derived features (would require phase redesign).
**Status:** Deferred to v3.0.

---

## Testing Strategy

### Unit Tests (pytest)
- 160+ test cases
- Coverage: state, config, ledger, cv, paths, schemas
- Run: `python -m pytest`

### Integration Tests
- Phase 1 end-to-end: `python -m zindian.cli phase 1`
- Skill verification: `python -m zindian.cli verify-phase-b`

### Validation Tests
- Preflight: `python -m zindian.cli preflight`
- Competition state: `python -m zindian.cli verify-state`


---

## Extension Points

### Adding a New Skill

1. Create `zindian/skills/skill_XX_name.py`
2. Implement `run()` function with standard signature
3. Add to phase map in `challenge_config.json`
4. Write unit tests in `tests/test_skill_XX.py`
5. Update `docs/source_of_truth.md`

### Adding a New Metric

1. Add to `skill_02_intake.py` metric detection
2. Update `skill_11_gate.py` threshold normalization
3. Add to `skill_12_metric.py` computation
4. Update regression/classification routing

### Adding a New CV Strategy

1. Add to `skill_05_cv.py` decision tree
2. Update `cv.py` strategy factory
3. Add selection reason documentation
4. Update preflight validation

---

# Quick Reference

## File Structure

```
competitions/<slug>/
├── challenge_config.json    # Competition contract (immutable after Phase 1)
├── SKILL_STATE.json         # Execution state (mutable)
├── data/
│   ├── raw/                 # Original data (MD5 locked)
│   └── processed/           # Cleaned data
├── reports/
│   ├── feature_policy.json
│   ├── eda_report.json
│   └── governance_report.json
└── submissions/
    └── submission_001.csv
```

## Key Commands

```bash
# Bootstrap new competition
python -m zindian.cli bootstrap <competition-slug>

# Run preflight
python -m zindian.cli preflight

# Run Phase 1
python -m zindian.cli phase 1

# Run full pipeline phases (e.g. 1, 2A, 2B, 3A, 3B, 4)
python -m zindian.cli phase <phase_id>

# Audit state
python -m zindian.cli verify-state

```

## Configuration Keys

### Essential Fields
- `competition_id`: Zindi competition slug
- `task_type`: "classification" | "regression"
- `metric`: "logloss" | "auc" | "rmse" | "mae" | "rmsle"
- `metric_direction`: "maximize" | "minimize"
- `target_col`: Target column name
- `cv_strategy`: CV configuration object
- `reproducibility.seed`: Fixed random seed

### Gate Thresholds
- `shap_leak_threshold`: 3.0 (default)
- `variance_gate_threshold`: 0.01 (default)
- `gate_margin`: 0.001 (default)

## State Keys

### Critical State
- `dag_phase`: Current phase
- `anchor_oof_score`: Baseline score
- `branch_{name}_oof`: Variant OOF records
- `leaked_features`: Flagged features
- `human_gate_X_approved`: Gate approvals

### Telemetry
- `telemetry.{skill_name}`: Per-skill metrics
- `telemetry.aggregate`: Cumulative metrics

## Common Patterns

### Reading Config
```python
from zindian.config import ChallengeConfig
config = ChallengeConfig.load("challenge_config.json")
task_type = config["task_type"]
```

### Updating State
```python
from zindian.state import SkillStateStore
state_store = SkillStateStore(Path("SKILL_STATE.json"))
state_store.update(dag_phase="phase_2a_complete", anchor_oof_score=0.82)
```

### Safe State Reads
```python
# With default
override_active = state_store.get("cv_strategy_override", {}).get("active", False)

# Check existence
if "pseudo_label_result" in state_store.state:
    retraining_required = state_store.state["pseudo_label_result"]["retraining_required"]
```

---

## Support Resources

- **Source of Truth:** [docs/source_of_truth.md](source_of_truth.md) — The authoritative system specification.
- **CLI Reference:** [docs/cli_integration_guide.md](cli_integration_guide.md) — CLI command usages.
- **Troubleshooting:** [docs/troubleshooting_guide.md](troubleshooting_guide.md) — Solutions for common failure modes.
- **Operational Master Spec:** [AGENTS.md](../AGENTS.md) — Operational guidelines for agent system prompt.

---

**Document Version:** 1.1
**Orchestrator Version:** 2.3
**Last Updated:** July 2026
