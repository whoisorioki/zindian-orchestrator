# Zindian Orchestrator - Complete Overview

**Version:** 2.8
**Last Updated:** August 2026
**Status:** Production Ready

---

## Table of Contents

1. [Documentation Landscape](#documentation-landscape)
2. [Non-Technical Overview](#non-technical-overview)
   - [What is Zindian Orchestrator?](#what-is-zindian-orchestrator)
   - [Core Philosophy: The Three Lenses](#core-philosophy-the-three-lenses)
   - [The Journey: 4 Main Phases](#the-journey-4-main-phases)
   - [Key Safety Features](#key-safety-features)
   - [Special Features](#special-features)
3. [Conceptual Analogy: Is This Reinforcement Learning?](#conceptual-analogy-is-this-reinforcement-learning)
4. [Success Metrics](#success-metrics)
5. [Technical Reference](#technical-reference)

---

## Documentation Landscape

| Document | Owns |
|---|---|
| [Source of Truth (docs/source_of_truth.md)](source_of_truth.md) | Architecture contracts, schemas, phase specs, gate logic, reproducibility |
| **[System Overview (docs/orchestrator_overview.md)](orchestrator_overview.md) (this file)** | Non-technical overview, phase summaries for new readers |
| [AGENTS.md](../AGENTS.md) | Agent implementation guide, safe access patterns, known live risks |
| [Quick Start Guide (docs/quick_start.md)](quick_start.md) | CLI bootstrap walkthrough, competition setup |
| [CLI Integration Guide (docs/cli_integration_guide.md)](cli_integration_guide.md) | All 21 CLI commands reference |
| [Ledger Architecture (docs/ledger_architecture.md)](ledger_architecture.md) | DuckDB experiment tracking schema |
| [Troubleshooting Guide (docs/troubleshooting_guide.md)](troubleshooting_guide.md) | Runtime errors, CI fixes |
| [Reporting & Logging Audit (docs/reporting_logging_audit.md)](reporting_logging_audit.md) | SWOT analysis, report footprint, action plan |
| [Document Structure Map (docs/document_map.md)](document_map.md) | Consolidated structure maps, cross-document overlaps, and ownership matrix |

---

## Non-Technical Overview

### What is Zindian Orchestrator?

Zindian Orchestrator is a structured execution framework for running tabular machine learning competitions on the Zindi platform. It automates and governs the entire lifecycle of a competition entry—from raw data ingestion, feature engineering, and cross-validation, to data leakage auditing, model ensembling, and submission management.

### The Problem It Solves

Entering machine learning competitions at scale is complex and prone to human and statistical errors. Specifically, it involves:
1. **Requirements Gathering:** Understanding target metrics, file formats, and submission rules.
2. **Data Integrity:** Ensuring the raw training and testing datasets are not corrupted or modified.
3. **Statistical Validation:** Designing split strategies (cross-validation) that mirror the test distribution without leakage.
4. **Iterative Search:** Exploring various feature transformations and model architectures.
5. **Quality Control:** Validating predictions through rigorous automated audits and human gates.
6. **Governance:** Enforcing daily submission budgets and documenting pipeline runs for full reproducibility.

The Orchestrator provides a unified, command-line interface (CLI) driven system that systematically handles these tasks, minimizing validation drift and ensuring complete transparency.

---

### Core Philosophy: The Three Lenses

Every decision is evaluated through three simultaneous perspectives:

| Lens | Question | Example |
|------|----------|---------|
| **General** | What does research say? | "Classification problems typically need stratified sampling" |
| **Specific** | What does THIS data show? | "This dataset has severe class imbalance" |
| **Generalization** | Will this work on new data? | "Will this pattern hold on unseen test data?" |


---

### The Journey: 4 Main Phases

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
- Two-Tier SHAP & Leakage Audit (Pearson blocking + advisory MI)
- Probability Calibration (for classification)
- Nadeau-Bengio Fold Variance Analysis (ddof=1)
- Gate evaluation (5 conditions must pass, incorporating 1-SE promotion margin)
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

### Key Safety Features

#### 1. Human Gates (5 Checkpoints)

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

#### 2. Reproducibility Contract

Everything must be repeatable. Run twice with same data = identical results.

**Requirements:**
-  Fixed random seed (set once, never changed)
-  Pinned dependencies (`requirements.txt`)
-  No custom packages
-  Complete audit trail
-  Submission reproducible from config + state alone

**Goal:** Ensures third-party verification and platform auditability without behavior drift.

---

#### 3. No AutoML

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

### Special Features

#### Carbon Tracking (R5)

Measures and reports environmental impact (CO2 emissions) of model training.

**Metrics tracked:**
- Duration (seconds)
- Peak memory (MB)
- Carbon estimate (kg CO2)
- Hardware type (CPU/GPU)
- Region (for carbon intensity)

**Goal:** Quantifies computational efficiency and resource footprints across model runs.

---

#### Pseudo-Labeling (Skill 21)

For classification: uses confident predictions on test data to expand training set.

**Guard conditions (all must pass):**
1. Classification task only
2. Not time-series data
3. No leaked features detected
4. Low fold variance
5. Calibrated probabilities available
6. Confidence threshold met (top 10%)

**Goal:** Leverages high-confidence semi-supervised predictions to enhance model decision boundaries.

---

#### Multi-Target Support

Handles competitions predicting multiple targets simultaneously.

**Example:** Predict both temperature AND humidity from weather data.

**How it works:**
- Trains separate models per target.
- Computes a weighted composite distance score where lower is better.
- Supports **Inverse-Variance Effective Weighting** (`w_k_eff = w_k / (sigma_k_NB^2 + epsilon)`) using Nadeau-Bengio corrected variances.
- Normalizes distance scores by target standard deviation (except RMSLE which is dimensionless).
- Evaluates a single composite gate decision for all targets.

---

### What Makes This Valuable?

-  **Consistency:** Same rigorous process every time
-  **Safety:** Multiple checkpoints prevent costly mistakes
-  **Efficiency:** Automates repetitive tasks
-  **Learning:** Captures knowledge from past competitions

---

### What It's NOT

-  **Not magic AI** - Doesn't solve everything automatically
-  **Not autonomous** - Requires human approval/intervention at key points
-  **Not a replacement** - Tool for data scientists, not replacement
-  **Not a black box** - Everything is traceable and explainable

---

## Conceptual Analogy: Is This Reinforcement Learning?

The orchestrator is not a classical RL algorithm, but it shares the core
state-action-reward feedback structure. Understanding the analogy helps clarify
how feedback loops are designed.

| RL Concept | Orchestrator Equivalent |
|---|---|
| Agent | Orchestrator control plane |
| Environment | Competition dataset + Zindi platform |
| State | `SKILL_STATE.json` + `challenge_config.json` |
| Action | Running a skill, promoting a branch, engineering a variant |
| Reward signal | OOF improvement, gate pass/fail, public LB delta |
| Policy | Phase map + gate conditions + three-lens rules |
| Episode | One competition lifecycle |

**Where it differs from true RL:**

- **Delayed and noisy rewards.** Public LB scores are time-lagged, budget-constrained, and cover only 20–30% of test data. OOF scores are faster proxy rewards but statistically imperfect.
- **Engineered policy.** Gate conditions, CV decision trees, and phase maps are hand-designed, not learned via gradient descent.
- **No value function.** The orchestrator makes greedy local decisions without modelling long-term multi-step consequences.

**Two feedback mechanisms approximate RL behaviour without gradient updates:**

1. **Cross-competition experience replay** — After every competition close, CV strategy choices, feature types, model architectures, and OOF-to-LB deltas are recorded in a history log. `skill_18` and `skill_20` read this log as prior knowledge for the next competition. See [Source of Truth §5](source_of_truth.md) for the history log schema.
2. **Bayesian threshold evolution** — After several competitions, `shap_leak_threshold`, `variance_gate_threshold`, and `gate_margin` are reviewed against historical OOF-to-LB data and updated in config.

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

## Technical Reference

For detailed technical specifications, refer to the following documents:

| Document | Content |
|---|---|
| [Source of Truth v2.8](source_of_truth.md) | Architecture contracts, phase specs, OOF schemas, gate logic formulas, known gaps |
| [AGENTS.md](../AGENTS.md) | Safe state access patterns, implementation conventions, known live risks |
| [Quick Start Guide](quick_start.md) | CLI bootstrap walkthrough, full skill matrix, competition setup |
| [CLI Integration Guide](cli_integration_guide.md) | All 21 CLI commands reference |
| [Ledger Architecture](ledger_architecture.md) | DuckDB experiment tracking schema |
| [Troubleshooting Guide](troubleshooting_guide.md) | Runtime errors, CI fixes, common failure modes |
| [Reporting & Logging Audit](reporting_logging_audit.md) | SWOT analysis, report footprint, action plan |
| [Document Structure Map](document_map.md) | Consolidated structure maps, cross-document overlaps, and ownership matrix |

---

**Document Version:** 1.5
**Orchestrator Version:** 2.8
**Last Updated:** August 2026
**Scope:** Non-technical overview and navigation hub. See [source_of_truth.md](source_of_truth.md) v2.8 for architecture details.
