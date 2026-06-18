# Zindian Orchestrator Architecture Matrix

Repository snapshot: current working commit `2d53250`.

## Purpose

This document is the onboarding map for the orchestrator. It describes the stable control flow, the feedback loops, and the extension boundaries that keep the system generic across any tabular or geospatial machine learning challenges.

It does not change runtime logic. It records how the current pipeline is structured so future algorithms and methods can be added without rewriting the control plane.

## Control Flow

```text
Skill 18 (Librarian)
    -> Skill 19 (Code Miner)
    -> Skill 20 (Scientist)
        -> validated_hypotheses.json

challenge_config.json
    -> Skill 07 (Feature Sandbox)
        -> OOF probability pool
            -> Skill 11 (Validation Gate)
                -> promotion / reset
                    -> Skill 17 (Governance)
```

## Stable Interfaces

### 0. Skill 01: Integrity Audit

Role:
- Locks the MD5 hash of the target column and raw files.
- Establishes the data integrity boundary before any transformation.

Current behavior:
- Reads training, test, and sample-submission files.
- Computes canonical MD5 hashes.
- Persists hashes into `SKILL_STATE.json`.
- Re-verifies hashes on later runs.

Why it matters:
- This is the reproducibility anchor for the whole pipeline.
- Every later step assumes the raw data fingerprint is stable.

### 0b. Skill 02: Challenge Intake

Role:
- Populates competition rules into `challenge_config.json`.
- Converts competition metadata into runtime policy.

Current behavior:
- Reads the competition API response.
- Derives metric direction, probability handling, limits, and domain notes.
- Leaves values null when the competition page does not provide them.

Why it matters:
- This is the policy boundary for the entire orchestrator.
- All later skills consult this file before applying competition-specific logic.

### 0c. Skill 08: Anchor Baseline

Role:
- Establishes the first confirmed anchor.
- Produces the baseline submission that later rounds must beat.
- Implements hierarchical configuration resolution (state → config → fallback).

Current behavior:
- Loads processed feature matrices from `features_{train|test}.csv`.
- Resolves target column via config hierarchy: `target_col` → `target_column` → inference.
- Resolves CV strategy via state override: `cv_strategy_override.active` → `config.cv_strategy.type` → "stratified".
- Applies policy filters from `challenge_config.json` to exclude features.
- Supports `anchor_challenge` state override for model_family, params, n_splits.
- Trains baseline LightGBM model with 3-layer seeding (random, numpy, model).
- Handles target transformations for regression (log1p for RMSLE, SoT v2.2).
- Computes secondary metrics (MAE, MAPE, R²) for regression tasks.
- Validates submission format before human gate.
- Updates state with anchor metrics, cv_strategy_id, and branch metadata.
- Creates git branch `anchor-baseline` for artifact locking.

Configuration hierarchy:
1. **State override** (`anchor_challenge.active=True`):
   - `model_family` / `framework` → Override model type (currently LightGBM only)
   - `params` / `hyperparams` → Override model parameters
   - `n_splits` → Override CV fold count
2. **Config** (`challenge_config.json`):
   - `target_col` / `target_column` → Training target
   - `cv_strategy.type` → Cross-validation strategy
   - `task_type` → classification / regression
   - `metric` → Primary metric (used for target transformation)
   - `policy_filters` → Excluded feature columns
3. **Fallback inference**:
   - Target: Train-only columns excluding ID/lat/lon
   - CV strategy: "stratified" default

Why it matters:
- This is the calibration point for the rest of the search process.
- The gate logic is only meaningful once an anchor exists.
- Hierarchical config resolution enables state-driven experimentation while maintaining backward compatibility.
- Policy filter enforcement ensures compliance with competition rules.
- Target transformation lifecycle (SoT v2.2) handles RMSLE/RMSE correctly.

### 0d. Skill 12: Metric Trade-off Analysis

Role:
- Scans existing OOF artifacts and ranks them by thresholded F1.
- Provides a deterministic comparison layer without training.

Current behavior:
- Loads OOF probability files from processed artifacts.
- Merges them with labels from training data.
- Sweeps thresholds and records the best F1 and stability gap.

Why it matters:
- This gives the pipeline a metric-centric feedback view.
- It is reusable across algorithms because it works on saved OOF outputs.

### 1. Skill 20: Scientist

Role:
- Sidecar research catalyst.
- Reads domain hypotheses and prior art.
- Produces validated hypotheses that downstream feature work can consume.

Current behavior:
- Builds structured feature hypotheses.
- Applies a static compatibility check against the current feature matrix.
- Applies an empirical validation check.
- Writes validated outputs and failure ledger entries for reuse.

Why it matters:
- This is the research-to-feature bridge.
- It keeps the model-building path grounded in evidence rather than ad hoc feature invention.

### 2. Skill 07: Feature Sandbox

Role:
- Deterministic feature and variant execution engine.
- Runs isolated variants against the current anchor.
- Persists OOF/test probability artifacts for later gating and governance.

Current behavior:
- Builds climate-algebra-derived features from validated hypotheses.
- Trains one variant at a time.
- Computes OOF F1, ROC-AUC, and threshold.
- Writes variant outputs and updates skill state.

Why it matters:
- This is the ingestion boundary for model experimentation.
- It is the primary place where new algorithms can be added while preserving the same outer contract.

### 3. Skill 11: Validation Gate

Role:
- Deterministic promotion gate.
- Promotes only if the round has passing variants.
- Resets the feature round after a successful promotion.

Current behavior:
- Compares best variant score to the anchor.
- Evaluates scale-invariant thresholds:
  - For RMSE: Scales variance gate by `target_std ** 2` and gate margin by `target_std`.
  - For RMSLE and Classification: Evaluates raw thresholds (dimensionless log-ratio/bounded).
- Evaluates directional check based on metric direction:
  - Maximization (e.g. AUC, F1): variant must exceed baseline + margin.
  - Minimization (e.g. RMSE, RMSLE, MAE): variant must be below baseline - margin.
- Creates or switches to the next anchor branch.
- Resets round counters and updates state.

Why it matters:
- This is the control point that prevents weak branches from becoming anchors.
- It preserves the feedback loop without changing the meaning of the underlying model logic.

### 4. Skill 17: Governance

Role:
- Final selection and write-lock mechanism.
- Converts scored submissions into the final two selections.

Current behavior:
- Fetches scored submissions.
- Applies a human gate.
- Writes the final selection report.
- Locks the chosen submission IDs into state.

Why it matters:
- This is the portfolio and audit layer.
- It closes the pipeline with reproducible final choices.

### 5. Skill 15: Reporter

Role:
- Initializes the DuckDB ledger and writes phase summaries.
- Serves as the bookkeeping boundary for experiments and submissions.

Current behavior:
- Opens or initializes the ledger database.
- Counts experiment and submission records.
- Writes phase summary JSON into reports.

Why it matters:
- This is the audit trail that makes branch history reviewable.
- It keeps the state machine observable across runs.

### 6. Skill 16: Submission Governance

Role:
- Validates a submission file before any Zindi submit call.
- Enforces budget and human confirmation rules.

Current behavior:
- Checks CSV shape and ID alignment against `SampleSubmission.csv`.
- Enforces remaining submission budget.
- Sends the submission with a structured comment.
- Logs the outcome and refreshes rank metadata.

Why it matters:
- This is the final execution guard before the external platform call.
- It preserves reproducibility and compliance at the submission boundary.

## Feedback Loops

### Research-to-Feature Loop

1. Skill 18 gathers literature and prior-art signals.
2. Skill 19 extracts public strategy patterns.
3. Skill 20 turns those signals into validated hypotheses.
4. Skill 07 consumes validated hypotheses and turns them into feature variants.

This loop is the main mechanism that makes the system extensible beyond one fixed model family.

### Data Integrity Loop

1. Skill 01 locks the raw data fingerprint.
2. Skill 02 records the competition policy.
3. Skill 08 and later skills inherit those constraints.

This loop ensures the rest of the pipeline is anchored to a known dataset and a known rule set.

### Evaluation-to-Reset Loop

1. Skill 07 trains a variant and writes OOF artifacts.
2. Skill 11 compares the variant against the anchor.
3. If the gate passes, the anchor is promoted.
4. State counters reset for the next round.
5. Skill 17 later uses the accumulated submission record for final selection.

This loop keeps the system deterministic while still allowing iterative improvement.

## Generic Extension Boundaries

The architecture can stay generic if the following boundaries remain stable:

- Input boundary: `challenge_config.json` controls metric, constraints, and competition rules.
- Research boundary: Skill 20 only proposes and validates hypotheses; it should not own training policy.
- Feature boundary: Skill 07 owns feature construction and variant execution.
- Gate boundary: Skill 11 owns promotion logic and branch reset.
- Governance boundary: Skill 17 owns final submission selection.
- Reporting boundary: Skill 15 owns experiment bookkeeping and phase summaries.
- Submission boundary: Skill 16 owns the external submit contract.
- Metric boundary: Skill 12 owns OOF threshold comparison without training.

## Current Bottlenecks To Watch

These are structural, not behavioral:

- Scientist validation still uses a tree-model style empirical check.
- Variant selection in Skill 07 is currently defined inside code rather than by external registry.
- Deep research is connected through a separate orchestration path rather than the main phase loop.
- Submission validation and metric selection are still skill-specific rather than registry-driven.

These are the places where generic plugin-style behavior would be added later. The current logic should remain unchanged unless a deliberate refactor is approved.

## Scalability Principles

To support other algorithms and methods, keep the following rules:

- Preserve the current I/O contracts between skills.
- Add new estimators as new variant definitions rather than replacing the gate contract.
- Keep the scientist output format stable so downstream consumers can remain generic.
- Keep branch promotion and governance independent from the model family.
- Avoid embedding competition-specific assumptions in the orchestration layer.

## Summary

The current pipeline is already modular at the control-plane level. The main genericity constraint is not the loop structure; it is the hardcoded estimator and evaluation choices inside specific skills. The architecture should therefore evolve by widening plugin points, not by rewriting the stable DAG.

## Real Code Validation Notes

The following skill implementations are present in the codebase and were used as the source for this note:

- [zindian/skills/skill_01_integrity.py](file:///C:/Users/Adrian/Desktop/Agents/zindian-orchestrator/zindian/skills/skill_01_integrity.py)
- [zindian/skills/skill_02_intake.py](file:///C:/Users/Adrian/Desktop/Agents/zindian-orchestrator/zindian/skills/skill_02_intake.py)
- [zindian/skills/skill_07_features.py](file:///C:/Users/Adrian/Desktop/Agents/zindian-orchestrator/zindian/skills/skill_07_features.py)
- [zindian/skills/skill_08_anchor.py](file:///C:/Users/Adrian/Desktop/Agents/zindian-orchestrator/zindian/skills/skill_08_anchor.py)
- [zindian/skills/skill_11_gate.py](file:///C:/Users/Adrian/Desktop/Agents/zindian-orchestrator/zindian/skills/skill_11_gate.py)
- [zindian/skills/skill_12_metric.py](file:///C:/Users/Adrian/Desktop/Agents/zindian-orchestrator/zindian/skills/skill_12_metric.py)
- [zindian/skills/skill_15_reporter.py](file:///C:/Users/Adrian/Desktop/Agents/zindian-orchestrator/zindian/skills/skill_15_reporter.py)
- [zindian/skills/skill_16_submit.py](file:///C:/Users/Adrian/Desktop/Agents/zindian-orchestrator/zindian/skills/skill_16_submit.py)
- [zindian/skills/skill_17_governance.py](file:///C:/Users/Adrian/Desktop/Agents/zindian-orchestrator/zindian/skills/skill_17_governance.py)
- [zindian/skills/skill_18_librarian.py](file:///C:/Users/Adrian/Desktop/Agents/zindian-orchestrator/zindian/skills/skill_18_librarian.py)
- [zindian/skills/skill_19_code_miner.py](file:///C:/Users/Adrian/Desktop/Agents/zindian-orchestrator/zindian/skills/skill_19_code_miner.py)
- [zindian/skills/skill_20_scientist.py](file:///C:/Users/Adrian/Desktop/Agents/zindian-orchestrator/zindian/skills/skill_20_scientist.py)
