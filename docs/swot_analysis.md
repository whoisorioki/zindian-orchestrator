# Zindian Orchestrator — Living SWOT Analysis & Command Book

**Competition:** June Study Jam Series: Bank Transaction Volume Forecasting Challenge  
**Active Directory:** `competitions/june-study-jam-series-transaction-volume-forecasting-challenge`  
**Last Updated:** June 15, 2026

---

## 1. Phase-by-Phase Command Reference Book

Always run these commands from the repository root. On Windows PowerShell, ensure the environment variables are set in your session first.

### Environment Preparation (PowerShell)
```powershell
# Set environment variables for the current session
$env:COMPETITION_SLUG="june-study-jam-series-transaction-volume-forecasting-challenge"
$env:PYTHONIOENCODING="utf-8"
$env:PYTHONPATH="."
```

### Zindi Monitor & Playwright Browser Setup
*Goal: Crawl Zindi rules, evaluation page, prizes, and discussion board to update state and compliance files.*
```powershell
# Install playwright browser binaries (run once at setup)
.venv\Scripts\python -m playwright install

# Run the Zindi Monitor directly
.venv\Scripts\python zindian/zindi_monitor_core.py

# Run the Zindi Monitor via the orchestrator
.venv\Scripts\python -c "import zindian.orchestrator as orc; orc.run_skill('skill_00')"
```

### Phase 1 — Ingestion & Integrity
*Goal: Lock target hashes, ingest rules, and initialize the experiment database.*
```powershell
# Run all Phase 1 skills (skill_01, skill_02, skill_15)
.venv\Scripts\python -c "import zindian.orchestrator as orc; orc.run_phase(1)"
```

### Preflight Static Scans & Verification Checks
*Goal: Validate that config files, human gate structures, and file integrity align with the SoT before runs.*
```powershell
# Run preflight enforce script (exits 0 if compliant)
.venv\Scripts\python scripts/preflight_enforce.py --competition competitions/june-study-jam-series-transaction-volume-forecasting-challenge

# Run ground-truth verification audit of active competition directories
$env:PYTHONPATH="."; .venv\Scripts\python scripts/verify_competition_state.py
```

### Phase 2 — Baseline & Anchor Generation
*Goal: Train baseline model, establish anchor local OOF metric, and upload first submission.*  
*Precondition: Set `"human_gate_1_approved": true` in `SKILL_STATE.json`.*
```powershell
# Run all Phase 2 skills (skill_03, skill_08)
.venv\Scripts\python -c "import zindian.orchestrator as orc; orc.run_phase(2)"
```

### Phase 3 — Feature Engineering & Calibration
*Goal: Create processed feature sets, compute SHAP features, and perform calibration.*  
*Precondition: Anchor baseline is successfully established.*
```powershell
# Run all Phase 3 skills (skill_04, skill_05, skill_09, skill_10)
.venv\Scripts\python -c "import zindian.orchestrator as orc; orc.run_phase(3)"
```

### Phase 4 — Branch & Gate Validation
*Goal: Run gating logic comparing variants to baseline and submit approved branches.*
```powershell
# Run all Phase 4 skills (skill_11, skill_16)
.venv\Scripts\python -c "import zindian.orchestrator as orc; orc.run_phase(4)"
```

### Phase 5 — Oracle Fusion & Submission Selection
*Goal: Blend/fuse top candidates and select the final two submissions.*
```powershell
# Run all Phase 5 skills (skill_13, skill_14, skill_17)
.venv\Scripts\python -c "import zindian.orchestrator as orc; orc.run_phase(5)"
```

### On-Demand, Testing, & Out-of-Loop Skills (Not in Sequential Phase Runs)

These skills are not executed by the sequential phase runner (`orc.run_phase()`) because they represent parallel research operations, custom dataset plugins, dynamic experimental sandboxes, or final governance checks:

* **`pytest` (Testing Suite)**
  * *Canonical Phase*: Continuous / Development.
  * *Goal*: Validate complete codebase health, seed discipline, target-leak isolation, and rule compliance.
  * *Command*: `.venv\Scripts\pytest`
* **`skill_06_cleaning` (Data Cleaning & Imputation)**
  * *Canonical Phase*: Phase 2A.
  * *Goal*: Missingness indicator generation and constant column pruning.
  * *Note*: In the current Nedbank consolidations, cleaning is executed dynamically inside the custom feature extraction plugin (`plugins/nedbank_extractor.py`).
* **`skill_07_features` (Feature Engineering Sandbox)**
  * *Canonical Phase*: Phase 2B.
  * *Goal*: Generate mathematical feature engineering combinations (polynomials, interactions, ratios, target-dependent binning) and train candidate models.
  * *Command*: `.venv\Scripts\python -m zindian.skills.skill_07_features --variant variant-06 --fetch`
* **`skill_12_metric` (Metrics Audit)**
  * *Canonical Phase*: Phase 3A.
  * *Goal*: Compute unbiased fold score variance (`ddof=1`) and recommend optimal decision thresholds.
  * *Command*: `.venv\Scripts\python -m zindian.skills.skill_12_metric`
* **`skill_21_pseudo_label` (Classification Pseudo-label Retraining)**
  * *Canonical Phase*: Phase 3B.
  * *Goal*: Semi-supervised test set labeling & model retraining loop.
  * *Command*: `.venv\Scripts\python -m zindian.skills.skill_21_pseudo_label`
* **`skill_22_reproducibility_audit` (Final Pipeline Sign-off)**
  * *Canonical Phase*: Phase 4.
  * *Goal*: Validate complete reproducibility and human gate approvals before competition close.
  * *Command*: `.venv\Scripts\python -m zindian.skills.skill_22_reproducibility_audit`
* **Deep Research Pipeline (`skill_18` → `skill_19` → `skill_20`)**
  * *Canonical Phase*: Research Sidecar (Continuous).
  * *Goal*: Literature crawl, codebase mining, and hypothesis generation.
  * *Command*: `.venv\Scripts\python -c "import zindian.orchestrator as orc; orc.run_deep_research(domain='finance')"`

---

## 2. Current SWOT Analysis (Phase 2 Complete)

### Context & Evidence Base
```
current_dag_phase                 : phase_2_anchor_confirmed
preflight_result                  : PASS
config_lock_active                : true
Phase 1 complete                  : yes
Phase 2A (formal skill_06) complete: no (bypassed/consolidated into plugin)
Phase 2B (skill_08 baseline) complete: yes
human_gate_1_approved             : true
human_gate_2 branches approved    : 0 of 0 branches
human_gate_3_approved             : false
human_gate_4_approved             : false
human_gate_5_selection            : not selected
cv_strategy_type                  : KFold
anchor_oof_score                  : 209.87340
drift_threshold                   : 0.05
submission_budget_remaining       : 30
submission_budget_daily_remaining : 5
logged_in_as                      : whoisorioki
```

### Strengths
* **Claim** — Dataset integrity and target column hashes are locked down in state.
  * **Evidence** — `SKILL_STATE.json` has `md5_target_hash` populated with `"c374679a0bfc2b125a65d89b1c269266"`.
  * **Impact** — Prevents silent data corruption or leakage in future feature engineering runs.
* **Claim** — Preflight static analysis passes without compliance blockages.
  * **Evidence** — `preflight_enforce.py` returns `PRELIGHT ENFORCE: ALL CHECKS PASSED`.
  * **Impact** — Ensures the code has no architectural violations (such as AutoML libraries or cross-skill imports) before training starts.
* **Claim** — Baseline performance anchor exists.
  * **Evidence** — `anchor_oof_score` is successfully populated with `209.87340` (RMSE).
  * **Impact** — Establishes the mathematical reference point for variant gating comparison.
* **Claim** — Tabular Phase 2A data cleaning and imputation outcomes are functionally achieved.
  * **Evidence** — Non-numeric columns are factorized and NaNs are filled in the pipeline plugin (`nedbank_extractor.py`).
  * **Impact** — Ensures standard scaler and LightGBM model inputs are fully numeric without leakage.

### Weaknesses
* **Claim** — [RESOLVED] Target-Transformation Scaling Mismatch.
  * **Evidence** — Prior run of `skill_07` produced raw target RMSE (e.g. `142.55`) due to missing `regression_metric` on `train_lightgbm_cv`, causing a scale mismatch with log1p baseline (`0.55`).
  * **Resolution** — Patched `skill_07_features.py` to pass the correct `regression_metric` and dynamically access `lgb_result.oof_rmse`. Verified OOF RMSLE matches baseline exactly (0.55454).
* **Claim** — [RESOLVED] Generic Features Grouping Logic Bug.
  * **Evidence** — Digits check used `"0" in var_id` which matched `"variant-06"` incorrectly, grouping all `variant-0X` variants under `first_half` (14 features).
  * **Resolution** — Refactored to inspect the last digit of the variant ID and explicitly mapping `variant-00` to all features.
* **Claim** — [RESOLVED] Domain-Logic Hardcoding Violation (Assumption A5).
  * **Evidence** — Environmental features (`tmax_mean_sq`, `frost_risk`) were hardcoded in the core structural feature loops of `skill_07_features.py`.
  * **Resolution** — Purged environmental variables from core code, replacing them with an abstract, configuration-driven math engine that parses polynomials, interactions, ratios, and conditions dynamically from config, with a default backward-compatible fallback dictionary for unit testing.
* **Claim** — [RESOLVED] Log-Space Metric Discrepancy.
  * **Evidence** — Fold-level prints printed original-scale RMSE (`rmse=80.26`) while the baseline target transformation optimized log-space RMSLE (`valid_0 rmse: 0.574`), confusing logs.
  * **Resolution** — Standardized `_lightgbm_shared.py` to evaluate and log `rmsle` values fold-by-fold when `use_log1p` is True.
* **Claim** — State Ledger Disconnect.
  * **Evidence** — Because the cleaning operations are consolidated entirely within the custom extraction plugin (`plugins/nedbank_extractor.py`), `SKILL_STATE.json` lacks formal metadata tracking for missingness indicators (MNAR) and constant column pruning.
  * **Impact** — This functional shortcut works, but it bypasses the system's static auditing layers.
* **Claim** — Leaderboard Blindness.
  * **Evidence** — The `anchor_lb_score` remains unpopulated.
  * **Impact** — Until a baseline submission is executed, the orchestrator's automated drift detection utility cannot evaluate the delta between local Out-of-Fold (OOF) scores and the public leaderboard score.

### Opportunities
* **Action** — Configure generic mathematical features in `challenge_config.json`.
  * **Precondition** — Core features engine abstract refactor complete.
  * **Expected gain** — Generate interaction and polynomial terms dynamically on key financial features (e.g., `AnnualGrossIncome` or `txn_count`) to identify stronger signals and exceed the baseline.
  * **Budget cost** — 0 submissions.
* **Action** — Run Phase 3 feature engineering.
  * **Precondition** — Baseline is established and git branch is locked.
  * **Expected gain** — Unlocks variant generation and gating comparison.
  * **Budget cost** — 0 submissions.
* **Action** — Submit the baseline model.
  * **Precondition** — `sub_001_anchor.csv` exists and is formatted.
  * **Expected gain** — Sets `anchor_lb_score` and verifies external API submission flow.
  * **Budget cost** — 1 submission.

### Threats
* **Claim** — Strict submission limits constrain final validation iterations.
  * **Trigger condition** — Exceeding 5 daily submissions or 30 total submissions.
  * **Severity** — High.
  * **Mitigation** — Only submit variants that demonstrate a local cross-validation score improvement exceeding the gate margin.
* **Claim** — [RESOLVED] API Drift & Namespace Shadowing.
  * **Evidence** — Shadowed local package folder `zindi/` collided with live API submissions.
  * **Resolution** — Resolved using the dynamic import wrapper in `zindi_client.py` to isolate stubs.

---

## 3. Ranked Action List

* **Priority 1** — **Submit Baseline** — Run `skill_16_submit` on the baseline predictions `sub_001_anchor.csv` to establish the Leaderboard reference.
* **Priority 2** — **Run Phase 3 (Feature Engineering & Calibration)** — Run `skill_07` variants (e.g. `variant-06`) to extract and train variant candidates.
* **Priority 3** — **Run Phase 3A (Generalisation & SHAP Audit)** — Run `skill_10` to scan variant features for target leakage.

---

## 4. Compliance Tracking: General vs. Challenge-Specific Rules

To prevent governance failures and disqualification, the orchestrator distinguishes between two types of compliance constraints:

### A. General (Platform-Wide) Zindi Rules
These are standard constraints applicable to almost every competition on the Zindi platform. They are automatically checked and appended as baseline warnings:
* **Team Stability:** No new team members can be added during the final 5 days of the competition.
* **Selection Limit:** You must manually select exactly 2 submissions on the platform before the deadline for private judging.
* **Reproducibility:** A fixed random seed must be set for all training runs.
* **Package Licensing:** Only open-source packages are permitted (no paid/commercial SaaS or closed APIs for model predictions).

### B. Challenge-Specific Constraints
These are rules unique to this specific forecasting challenge, which the Zindi Monitor scans for in page rules and discussion forums:
* **Target Count Forecasting Limits:** No leakage of future transaction details (predictions must rely exclusively on historical transaction dates prior to November 2015).
* **Feature Scope:** No usage of external financial or demographic databases beyond the provided demographic/financial parquet files (as `"allowed_external_data": false` is enforced).
* **Banned Features:** The EY Frogs spatial/coordinate ban is removed; no spatial banned feature overrides apply to this tabular financial dataset.

---

## 5. Phase-by-Phase Architecture: Inputs, Outputs, and Execution Models

The orchestrator combines **Static Control Flow** and **Dynamic Execution Units** across the 6 canonical phases of the Source of Truth:

### Static vs. Dynamic Execution Paradigm
* **Static Components**: These represent deterministic control flow, data integrity assertions, math-based gates, and compliance audits that execute in a fixed sequence. Examples include `skill_01` (integrity checks), `skill_03` (legality checks), `skill_11` (mathematical promotion gating), `skill_15` (audit reporting), and `skill_22` (reproducibility audit). These steps guarantee system safety, enforce temporal config locks, and prevent target leakage.
* **Dynamic Components**: These represent the active feature engineering, model exploration, and parameter tuning blocks. Examples include `skill_07` (variant feature extraction), `skill_08` (baseline anchor training), `skill_09` (cross-validation and variant model training), and custom plugins (like `plugins/nedbank_extractor.py`). These components adapt to the specific tabular, spatial, or temporal nature of the competition data and generate predictions that feed into static validation checks.

### Detailed Phase-by-Phase Breakdown

#### Phase 1 — Competition Fingerprint & Config Lock
* **Execution Model**: **Static**. Ensures strict sequence progression and establishes project baseline parameters.
* **Input**: Raw competition data files (`Train.csv`, `Test.csv`, `SampleSubmission.csv`), Zindi discussion forum, and rules crawled via `skill_00`.
* **Output**: Locked configuration `challenge_config.json` (defining target bounds, evaluation metrics, and the selected CV strategy), locked file hashes, initial state metadata in `SKILL_STATE.json`, and initialization of the experiment tracking database (`experiments.db`).

#### Phase 2A — Data Cleaning & Imputation
* **Execution Model**: **Static Control / Dynamic Utility**.
  * *Standard SoT Flow*: Evaluates `policy_gate()` to assert that blocked columns are removed, then runs `skill_06_cleaning` to apply missingness indicators (MNAR columns first, then MCAR median/mode filling) and drop constant columns.
  - *Current Nedbank Consolidated Flow*: Bypassed in formal orchestrator execution because the active `phase_skill_map` maps directly to `["skill_03", "skill_08"]`. Instead, the cleaning logic (categorical factorization, Age calculation from BirthDate, median filling for NaNs) is executed dynamically within the custom feature extractor plugin (`plugins/nedbank_extractor.py`).
* **Input**: Raw demographic/financial/transaction parquet files, EDA missingness specifications from `SKILL_STATE.json`.
* **Output**: Cleaned feature matrices (`features_train.csv`, `features_test.csv`) saved to the processed data folder, and (when formal) cleaning metrics logged in `SKILL_STATE.json`.

#### Phase 2B — Signal Search (Anchor Baseline & Feature Engineering)
* **Execution Model**: **Dynamic**. Runs baseline and initial feature variant training.
* **Input**: Cleaned feature matrices, locked cross-validation parameters from `challenge_config.json`, and seed configurations.
* **Output**: Anchor baseline predictions (`sub_001_anchor.csv`), Out-of-Fold (OOF) prediction arrays, locked baseline anchor performance metric (`anchor_oof_score` set to `209.87340` RMSE), and creation of the `anchor-baseline` git branch.

#### Phase 3A — Generalisation Audit (SHAP & Calibration)
* **Execution Model**: **Static / Audit**. Stress-tests model predictions and variant features to check for target leakage or overfitting.
* **Input**: Variant model predictions, target values, feature matrices, and SHAP explainers.
* **Output**: Per-fold SHAP feature importance arrays (`skill_10`), target leakage audits, calibrated probabilities (`skill_09` for classification), and optimal metric threshold grids (`skill_12`).

#### Phase 3B — Promotion & Fusion
* **Execution Model**: **Static Gating & Dynamic Fusion**. Promotes validated variant candidates and builds ensemble blends.
* **Input**: Candidate OOF scores, SHAP leakage audits, anchor baseline score, and human gate approvals (Human Gate 2 and Human Gate 3).
* **Output**: Promoted branches, blended ensemble predictions (`skill_13_oracle_fusion`), and optional pseudo-labeled training sets (`skill_21` for classification tasks).

#### Phase 4 — Governance (Formatting & Submission Guard)
* **Execution Model**: **Static Gatekeeper**. Restricts final actions based on Zindi platform regulations and sanity checks.
* **Input**: Fused/ensembled prediction arrays, target domain bounds, daily/total submission budget counters, and human gate approvals (Human Gate 4 and Human Gate 5).
* **Output**: Formatted submission CSV (`skill_14_inference`), Zindi submission API call (`skill_16_submit`), and final sign-off report with git tag assets (`skill_17_governance`, `skill_22`).

