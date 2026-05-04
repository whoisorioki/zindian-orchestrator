# Zindian Orchestrator — Build Tasks & Checklist

The agent executing this file works top to bottom. Each task has a status: `TODO` | `IN_PROGRESS` | `DONE` | `BLOCKED`

Update this file at the start of each session after reading `SKILL_STATE.json` and `challenge_config.json`.

---

## Current Session Status

**Date**: 2026-05-04  
**Competition**: financial-inclusion-in-africa (MAE metric)  
**Phase**: phase_0_foundation  
**Submissions Used**: 0/10  
**Current Progress**: Phase 0 infrastructure ~70% complete  

---

## PHASE 0 — Foundation (Wiring + Auth + Infrastructure)

> Goal: Environment, credentials, state files, and DuckDB ledger ready. No ML yet.

- [x] Ubuntu WSL environment set up
- [x] Python 3.12 venv at `.venv/` created
- [x] Core ML packages installed (lightgbm, pandas, shap, duckdb, scikit-learn)
- [x] OpenCode installed and connected to Gemini
- [x] Zindi CLI installed
- [x] Zindi authentication confirmed (logged in)
- [x] `.env` file created with API keys (verify not in git)
- [ ] Fix Zindi CLI `select_a_challenge()` JSONDecodeError (if it reoccurs)
- [ ] Initialize DuckDB ledger: run `python scripts/init_ledger.py`
- [ ] Verify `reports/experiments.db` exists with `experiments` and `submissions` tables
- [ ] Create `specs/` directory with requirements.md, design.md, tasks.md
- [ ] Create tool-agnostic spec structure (symlinks/copies to `.github/`, `.cursor/`, `.windsurf/`, `.kiro/`)
- [ ] Verify all tool directories contain copies of AGENTS.md
- [ ] Create `CLAUDE.md` (copy of AGENTS.md)

**Phase 0 Gate**: ✅ All items done before proceeding to Phase 1

---

## PHASE 1 — Integrity + Intake (MD5 Lock + Challenge Config Population)

> Goal: Lock MD5 hash of target column. Populate challenge_config.json from Zindi API. Initialize reporter.

### Skill 01 — Integrity Audit (MD5 Hash Lock)

- [x] Write `zindian_orchestrator/skills/skill_01_integrity.py`
  - [x] Compute MD5 hash of target column from raw data
  - [x] Lock hash to `SKILL_STATE.json` as `md5_target_hash`
  - [x] Raise `MD5MismatchError` if target column modified since last run
  - [x] Log integrity report to `reports/integrity_audit.json`
- [ ] Test: Run Skill 01 on fresh financial-inclusion dataset
- [ ] Test: Verify `md5_target_hash` written to SKILL_STATE.json
- [ ] Test: Verify hash verification works on subsequent runs

### Skill 02 — Challenge Intake (Config Populator)

- [x] Refactor `zindian_orchestrator/skills/skill_02_intake_new.py` (created with full API support)
  - [x] Query Zindi API for competition metadata
  - [x] Populate `challenge_config.json` from API response
  - [x] Validate all required fields present (metric, domain, use_probabilities, etc.)
  - [x] Raise `ConfigNotPopulated` if any required field is null
  - [x] Update `SKILL_STATE.json`: `dag_phase` → "phase_1_intake"
- [ ] Test: Run Skill 02, verify challenge_config fully populated
- [ ] Test: Verify SKILL_STATE.json dag_phase updated

### Skill 15 — Reporter (DuckDB Initialization)

- [x] Write `zindian_orchestrator/skills/skill_15_reporter.py`
  - [x] Initialize DuckDB ledger at `reports/experiments.db`
  - [x] Verify schema: `experiments` table with (experiment_id, branch_name, oof_rmse, feature_count, calibration_method, gate_result, gate_reason, timestamp)
  - [x] Verify schema: `submissions` table with (submission_id, experiment_id, branch_name, submission_rank, public_score, private_score, my_rank, selected_for_final, rationale, timestamp)
  - [x] Write initial report metadata to `reports/phase_1_summary.json`
- [ ] Test: Run Skill 15, verify `reports/experiments.db` initialized correctly

**Phase 1 Gate**: ✅ MD5 locked + challenge_config verified + ledger ready before proceeding to Phase 2

---

## PHASE 2 — Anchor Baseline (EDA + LightGBM Anchor + Submit)

> Goal: Establish ground-truth baseline. Submit anchor model. Determine OOF RMSE floor.

### Skill 03 — EDA (Exploratory Data Analysis)

- [ ] Write `zindian_orchestrator/skills/skill_03_eda.py`
  - [ ] Load raw data, verify target column MD5 hash
  - [ ] Profile: missing values, dtypes, cardinality, distributions
  - [ ] Generate EDA report to `reports/eda_baseline.json`
  - [ ] Identify feature engineering opportunities
- [ ] Test: Run Skill 03, verify EDA report written

### Skill 08 — Anchor Baseline (LightGBM + Cross-Validation)

- [ ] Write `zindian_orchestrator/skills/skill_08_anchor.py`
  - [ ] Load raw features (no engineered features)
  - [ ] Stratified 5-fold CV with LightGBM
  - [ ] Calculate OOF RMSE (or appropriate metric from config)
  - [ ] Save OOF predictions to `submissions/sub_001_anchor.csv`
  - [ ] Log experiment to DuckDB ledger
  - [ ] Update `SKILL_STATE.json`: `anchor_oof_rmse` field
  - [ ] Output submission-ready CSV (format: unique_id, prediction)
- [ ] Test: Anchor trains without errors
- [ ] Test: OOF RMSE persisted to SKILL_STATE.json and ledger

### Skill 08b — Anchor Submit (Budget-Checked)

- [ ] Extend `zindian_orchestrator/skills/skill_08_anchor.py` or create `skill_08b_anchor_submit.py`
  - [ ] Check `remaining_submissions` via ZindiClient
  - [ ] Submit `submissions/sub_001_anchor.csv` to Zindi
  - [ ] Structure comment: `branch:anchor|oof_rmse:0.XXX|features:8|calib:none`
  - [ ] Poll Zindi API for `public_score` and `my_rank`
  - [ ] Update `SKILL_STATE.json`: `submissions_used_today`, `anchor_lb_score`
  - [ ] Log submission to DuckDB `submissions` table
- [ ] Test: Budget guard blocks if `remaining_submissions == 0`
- [ ] Test: Submission comment matches format spec
- [ ] Test: my_rank polled and logged

**Phase 2 Gate**: ✅ Anchor submitted + leaderboard score visible before proceeding to Phase 3

---

## PHASE 3 — Features + Calibration (Governed Feature Work + SHAP + Calibration)

> Goal: Engineer features, calibrate model, explain predictions.

### Skill 04 — Feature Engineering

- [ ] Write `zindian_orchestrator/skills/skill_04_features.py`
  - [ ] Generate candidate features (domain-aware based on challenge_config.domain)
  - [ ] Filter features by importance (SHAP or mutual information)
  - [ ] Save feature list to `reports/feature_list.json`
  - [ ] Create feature matrix for downstream skills
- [ ] Test: Features generated and persisted

### Skill 05 — Experiment Branches (Optional, can run in parallel with 09–10)

- [ ] Write `zindian_orchestrator/skills/skill_05_branches.py`
  - [ ] Create multiple branches (e.g., `feature_v1`, `feature_v2`)
  - [ ] Train model per branch
  - [ ] Compute OOF RMSE per branch
  - [ ] Log to DuckDB `experiments` table
- [ ] Test: Multiple branches logged

### Skill 09 — Calibration

- [ ] Write `zindian_orchestrator/skills/skill_09_calibration.py`
  - [ ] Apply isotonic regression on validation fold
  - [ ] Recalibrate OOF predictions
  - [ ] Compute calibrated OOF RMSE
  - [ ] Log to ledger with `calibration_method: "isotonic"`
- [ ] Test: Calibration applied and OOF RMSE updated

### Skill 10 — SHAP Analysis

- [ ] Write `zindian_orchestrator/skills/skill_10_shap.py`
  - [ ] Compute SHAP values for top 20 features
  - [ ] Generate SHAP summary plot and save to `reports/shap_analysis.json`
  - [ ] Document feature importance rationale
- [ ] Test: SHAP analysis written

**Phase 3 Gate**: ✅ Multiple calibrated branches tested + SHAP analysis complete before Phase 4

---

## PHASE 4 — Branch + Gate (Branch Selection + OOF Gate + Submission Approval)

> Goal: Test multiple branches. Gate each against anchor OOF RMSE. Approve for submission.

### Skill 11 — Gate Checker

- [ ] Write `zindian_orchestrator/skills/skill_11_gate.py`
  - [ ] Query DuckDB `experiments` table
  - [ ] For each branch: compute (anchor_oof - branch_oof) / anchor_oof
  - [ ] Gate rule: ≥0.5% improvement required
  - [ ] Log gate results to DuckDB with reason
  - [ ] Return list of branches that PASS gate
- [ ] Test: Gate correctly prunes branch with worse OOF RMSE

### Skill 16 — Critique & Approval

- [ ] Write `zindian_orchestrator/skills/skill_16_critique.py`
  - [ ] Summarize all passed branches
  - [ ] Recommend ≥2 branches for final submission (based on diversity)
  - [ ] Log rationale to `reports/critique.json`
  - [ ] Prepare for Phase 5 submission

**Phase 4 Gate**: ✅ ≥2 branches pass gate + rationale documented before Phase 5

---

## PHASE 5 — Fusion + Final Submit (Ensemble + Final Submission)

> Goal: Fuse predictions from best branches. Submit final 2 selections. Complete competition.

### Skill 13 — Fusion Ensemble

- [ ] Write `zindian_orchestrator/skills/skill_13_fusion.py`
  - [ ] Fuse ≥2 gated branches via stacking or weighted average
  - [ ] Compute final OOF RMSE
  - [ ] Apply physical domain constraints if applicable (check config.domain)
  - [ ] Apply probability thresholding only if not use_probabilities (check config)
  - [ ] Save fusion predictions to `submissions/sub_00X_fusion.csv`
- [ ] Test: Fusion ensemble created

### Skill 14 — Inference Guard (Pre-Submit Validation)

- [ ] Write `zindian_orchestrator/skills/skill_14_inference_guard.py`
  - [ ] Validate submission CSV format (matches submission_format in config)
  - [ ] Check for NaNs, infinities
  - [ ] Verify row count matches test set
  - [ ] Verify unique_id column present and unique
- [ ] Test: Inference guard passes or fails appropriately

### Skill 17 — Submission Governance (Final 2 Selections)

- [ ] Write `zindian_orchestrator/skills/skill_17_sub_governance.py`
  - [ ] Query `submissions` table for all submitted models
  - [ ] Score each by diversity + public_score
  - [ ] Select top 2 for final private judging
  - [ ] Log selections to `reports/final_selections.md` with rationale
  - [ ] Lock `SKILL_STATE.json`: `selected_submissions: [sub_id_1, sub_id_2]`
- [ ] Test: Exactly 2 submissions selected + rationale logged

**Phase 5 Gate**: ✅ Final 2 selections documented + competition completed

---

## Cross-Cutting Infrastructure

### zindian_orchestrator/state.py
- [x] Reads `SKILL_STATE.json` (started)
- [ ] **HARDEN**: Implement atomic write (write-then-rename)
- [ ] **HARDEN**: Add `increment()` method for submission tracking
- [ ] Test atomicity: concurrent read/write doesn't corrupt file

### zindian_orchestrator/config.py
- [ ] **CREATE**: Read `challenge_config.json`
- [ ] **CREATE**: Raise `ConfigNotPopulated` if required fields null
- [ ] **CREATE**: Provide `.get(key, default=None)` access
- [ ] Test: Config reads correctly, null guard works

### zindian_orchestrator/ledger.py
- [ ] **CREATE**: DuckDB wrapper
- [ ] **CREATE**: `log_experiment()` method
- [ ] **CREATE**: `log_submission()` method
- [ ] **CREATE**: Query interface
- [ ] Test: Ledger initializes, reads/writes correctly

### zindian_orchestrator/zindi_client.py
- [ ] **HARDEN**: Verify budget guard before every submit
- [ ] **HARDEN**: Structure submission comments correctly
- [ ] **HARDEN**: Poll leaderboard after submit
- [ ] Test: BudgetExhausted raised when budget is zero

---

## Testing & Validation

- [ ] **Unit tests**: Each skill testable independently
- [ ] **Integration tests**: Phase transitions work end-to-end
- [ ] **Budget tests**: Verify submission limit respected
- [ ] **Data integrity tests**: MD5 hash verification works
- [ ] **Reproducibility tests**: Notebook runs top-to-bottom on fresh data

---

## Documentation & Reporting

- [ ] `reports/phase_0_summary.json` — foundation setup summary
- [ ] `reports/phase_1_summary.json` — integrity + intake summary
- [ ] `reports/eda_baseline.json` — exploratory data analysis
- [ ] `reports/shap_analysis.json` — feature importance analysis
- [ ] `reports/experiments.db` — DuckDB ledger (all runs)
- [ ] `reports/critique.json` — gate results + branch recommendations
- [ ] `reports/final_selections.md` — rationale for 2 final submissions
- [ ] `reports/submission_log.md` — all submission attempts + scores

---

## Submission Budget Tracking

| Phase | Max/Phase | Notes |
|-------|-----------|-------|
| Anchor (Phase 2) | 1 | Establish baseline only |
| Exploration (Phase 3–4) | 5 | Gated branches only |
| Final (Phase 5) | 2 | Exact 2 selections for private |
| Reserve | 2 | Never fully exhaust daily limit (10 max) |

**Current**: 0/10 used today

---

## Next Task (For Agent at Session Start)

Read this file and `SKILL_STATE.json`.

**Current phase**: phase_0_foundation

**Current task**: Initialize DuckDB ledger (Phase 0, item "Initialize DuckDB ledger")

**Proposed next step**: 
1. Run `python scripts/init_ledger.py` to create `reports/experiments.db`
2. Verify schema with `sqlite3 reports/experiments.db ".schema"`
3. Proceed to Phase 1 (Skill 01 — MD5 hash lock)

**Approval needed**: Yes — confirm Phase 1 is ready to start before proceeding.
