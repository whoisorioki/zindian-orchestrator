# Zindian Orchestrator — Build Tasks & Checklist
# Source of truth for all agents. Update at every session start.
# Status: TODO | IN_PROGRESS | DONE | BLOCKED

---

## Current Session Status

**Date**: 2026-05-06
**Competition**: EY Biodiversity Challenge — Frogs (ey-biodiversity-challenge)
**Branch**: anchor-v3
**DAG Phase**: phase_3_features — Round 4 running
**Anchor LB F1**: 0.881642512 — rank 240 — submission WeXoXWi6
**Submissions today**: check SKILL_STATE.json before any submit
**Deadline**: May 24, 2026 — 18 days remaining
**Final sub selection deadline**: May 19, 2026

---

## COMPLIANCE CHECKPOINT — Read Before Any Code

PERMITTED
- Raw Latitude and Longitude as direct model inputs
- All 14 TerraClimate variables (listed in competition requirements.txt)
- Long-term climate statistics: mean, std, min, max
- Open source packages only
- Pretrained models if openly available to everyone

BANNED
- Derived spatial features — discussion 32369
  No distance calculations, spatial clusters, bins, H3 hex bins,
  admin region encodings, polynomial or interaction terms of Lat/Lon
- External datasets — GBIF, Kaggle, WorldClim unless verified permitted
- AutoML tools — H2O, AutoSklearn, TPOT etc.
- Thresholding before submission — threshold search is internal OOF only
- Latitude and Longitude in compliant submissions
  Removed from anchor-v3 onward — TerraClimate only

---

## PHASE 0 — Foundation — DONE

- [x] WSL Ubuntu environment + Python 3.12 venv at .venv/
- [x] All packages installed: lightgbm, xarray, dask, rioxarray, rasterio,
      shap, adlfs, pystac-client, planetary-computer, ipykernel, jupyter
- [x] VS Code WSL Remote extension configured, kernel registered as Zindian (WSL)
- [x] Zindi CLI installed and authenticated — user: whoisorioki
- [x] .env file created with credentials — confirmed not in git
- [x] tabula init ey-frogs executed — competition workspace scaffolded
- [x] competitions/ey-frogs/ folder structure confirmed

---

## PHASE 1 — Integrity + Intake — DONE

- [x] skill_01_integrity.py — MD5 hash locked
      md5_train_file  : 0373cba783a545f95af48b206959ab9a
      md5_test_file   : 429b1a8ec3868dc73dc94eaffe058d68
      md5_sample_sub  : 6cea7a8efd61aaea9b4cf370c94f46c8
      md5_target_hash : 358790892826d3197c30a577c893e29a
- [x] skill_02_intake.py — challenge_config.json fully populated
      metric: f1_score, use_probabilities: false, daily_limit: 10
      banned_features: derived_spatial_features, external_spatial_data
      code_review_tier: top_10, max_team_size: 3
- [x] skill_15_reporter.py — DuckDB ledger initialized at reports/experiments.db
- [x] SKILL_STATE.json dag_phase: phase_1_complete
- [x] compliance_log.md — 3 flagged discussion threads reviewed and understood

### Deep Research Loop (new)

- [x] skill_18_librarian.py — domain literature retrieval (Semantic Scholar)
- [x] skill_20_scientist.py — bounded hypothesis generation (Gemini free-tier)
- [x] skill_19_code_miner.py — machine-learning prior-art mining (Gemini + fallback)
- [x] Orchestrator wiring: run 02.4 and 02.6 before 02.5 synthesis
      Implemented via `zindian.orchestrator.run_deep_research()` and the updated skill outputs
- [x] Add schema tests for code_miner_cache.json and code_miner_patterns.json
      Covered in `tests/test_deep_research_scaffolds.py`

---

## PHASE 2 — Anchor Baseline — DONE

- [x] skill_03_legality_check.py — legality verified manually and via script
      Status: GO — TerraClimate permitted, f1_score metric confirmed,
      use_probabilities=false confirmed, derived spatial banned confirmed
      NOTE: script lives in competitions/ey-frogs/scripts/ not zindian/skills/
      TODO: move to zindian/skills/skill_03_legality.py
- [x] skill_08_anchor.py — anchor baseline trained and submitted
      Model: LightGBM, KFold n=5, features: Latitude + Longitude only
      OOF logloss: 0.4852 — OOF AUC: 0.8340
      Branch: anchor-baseline created and locked
- [x] Bug fixes applied to skill_08 this session (2026-05-06)
      Bug 1: save_submission used float64 — fixed to int32 hard labels
      Bug 2: validate_submission Check 8 inverted logic for use_probabilities=False
      Bug 3: compute_oof_predictions used Booster.predict_proba — fixed to predict()
      Bug 4: threshold search added before submission file creation

---

## PHASE 3 — Feature Engineering — IN PROGRESS

### TerraClimate Data Pipeline — DONE

- [x] fetch_terraclimate_full.py — 13 variables x 4 stats = 52 bands
      Time slice: 2011-2021 (10-year modern normal, 132 months)
      Spatial extent: SE Australia — 100% training data coverage confirmed
      Retry logic: per-band checkpointing survives Azure timeouts
      Output: competitions/ey-frogs/data/processed/TerraClimate_14band.tiff
- [x] extract_terraclimate_features.py — spiral NaN imputation
      17 coastal training points fixed via nearest valid land pixel
      4 test points fixed
      0 NaNs remaining — integrity verified
      Output: features_train.csv, features_test.csv
- [x] SHAP analysis run on variant-06 full model
      Top 5: aet_min, tmin_mean, Longitude, pet_mean, srad_mean
      Dead (near-zero): swe_mean, swe_std, swe_min, swe_max, def_min, q_min

### Skill 07 — Feature Engineering — IN PROGRESS

- [x] zindian/skills/skill_07_features.py written and operational
      Governed: reads config, writes state, generates round reports
      Variants 06-16 defined, model branching for LGB tuned and RF ensemble
- [x] skill_11_gate.py — branch gate operational
      Promotes best variant to new anchor, advances feature round
- [x] skill_16_submit.py — submission governance operational
      5-check validation, budget guard, human gate, structured comment

### Round 1 Results — anchor: 0.83396 (Lat/Lon only) — DONE

- [x] variant-01  srad only (2 features)          AUC 0.6553  PRUNE  ad-hoc
- [x] variant-02  vap only (2 features)           AUC 0.6119  PRUNE  ad-hoc
- [x] variant-03  srad + vap (3 features)         AUC 0.6982  PRUNE  ad-hoc
- [x] variant-04  LightGBM Lat/Lon (2 features)   AUC 0.8272  PRUNE  ad-hoc
- [x] variant-05  LR optimal threshold (2 feat)   AUC 0.6790  PRUNE  ad-hoc
- [x] variant-06  All 52 TC bands (54 features)   AUC 0.84387 PASS   WINNER
- [x] variant-07  Temperature only (10 features)  AUC 0.83687 PRUNE
- [x] variant-08  Water balance (26 features)     AUC 0.83887 PRUNE
- [x] variant-09  Radiation+humidity (14 feat)    AUC 0.84266 PASS
      Gate result: variant-06 promoted to anchor-v2 via skill_11_gate

### Round 2 Results — anchor: 0.84387 (All 52 TC + Lat/Lon) — DONE

- [x] variant-10  Top 10 SHAP (12 features)       AUC 0.84262 PRUNE
- [x] variant-11  Drop dead features (48 feat)    AUC 0.84303 PRUNE
- [x] variant-12  Top 20 SHAP (22 features)       AUC 0.84248 PRUNE
- [x] variant-13  Tuned LGB (54 features)         AUC 0.84387 PRUNE  BUG — delta=0
- [x] variant-14  RF ensemble (54 features)       AUC 0.84387 PRUNE  BUG — delta=0
- [x] variant-16  Top 5 SHAP (7 features)         AUC 0.83736 PRUNE
      Gate result: all pruned — no promotion
      NOTE: variant-13 and variant-14 produced identical results to variant-06
      Root cause: special model code not executing — bug unresolved

### Round 3 — Compliance Correction — DONE

- [x] Compliance audit: Lat/Lon confirmed BANNED in submission
      Non-compliant submissions identified: LrU3Hg7g, FgpCxZ5e, yDnrXdKz, eWDKfyBV
- [x] anchor-v3 established: TC only (52 bands, no Lat/Lon)
      OOF AUC: 0.84291 — LB F1: 0.881642512 — rank 240
      Submission: WeXoXWi6 — file: sub_011_anchor.csv — SELECTED

### Round 4 — IN PROGRESS

- [ ] variant-27  TC all 52 — LGB tuned (deeper, lower LR)
- [ ] variant-28  TC all 52 — Random Forest
- [ ] variant-29  TC all 52 — XGBoost
- [ ] variant-30  TC temp 8 — LGB default — temperature only
- [ ] variant-31  TC water 16 — LGB default — moisture only
- [ ] variant-32  TC stress 12 — LGB default — drought stress only
- [ ] variant-33  SHAP top 12 — LGB default — domain science features
- [ ] variant-34  TC all 52 — LGB + RF blend — ensemble
      Gate threshold: OOF AUC delta >= 0.005 vs anchor 0.84291
      All variants: NO Latitude, NO Longitude

---

## PHASE 4 — Gate + Promotion — TODO

- [ ] Run skill_11_gate after Round 4 completes
- [ ] If variant passes: promote to anchor-v4, create new branch
- [ ] If all pruned: define Round 5 hypotheses (see below)
- [ ] Submit any passing variant via skill_16_submit

### Round 5 Hypotheses (if Round 4 all pruned)

- [ ] XGBoost with dart booster on TC features
- [ ] CatBoost on TC features
- [ ] Per-fold threshold optimisation instead of global threshold
- [ ] Stacking: LGB + RF + XGB with logistic meta-learner
- [ ] WorldClim 19 bioclimatic variables — verify legality first
- [ ] SRTM elevation data — verify legality first

---

## PHASE 5 — Fusion + Final Submission — TODO

- [ ] skill_13_fusion.py — HUMAN GATED — write before May 17
      Blend variant-06 (non-compliant, reference only) and anchor-v3
      Or blend two compliant passing variants if Round 4 produces them
- [ ] skill_14_inference.py — HUMAN GATED — post-processing
- [ ] skill_17_governance.py — final 2 submission selection
      Must lock selections by May 19 (5 days before deadline)
      Current selected: WeXoXWi6 (sub_011_anchor.csv, LB 0.8816)
      Second selection: TBD — best passing variant from Round 4+

---

## KNOWN BUGS — Fix Before Next Feature Run

- [ ] BUG: variant-13 and variant-14 special model code not executing
      Both returned delta=0.0 (identical to variant-06)
      Diagnosis: grep -n "variant-13\|elif variant\|RandomForest" zindian/skills/skill_07_features.py
      Fix: verify variant_name parameter passes correctly into train_variant()
      Rerun after fix: python3 -m zindian.skills.skill_07_features --variant=variant-13

- [ ] BUG: skill_08 cosmetic fields stale
      git_branch hardcoded as anchor-baseline — should read from state
      n_features hardcoded as 2 — should count feature_cols dynamically

- [ ] TODO: Move skill_03_legality_check.py to zindian/skills/skill_03_legality.py

---

## SKILLS STATUS

| Skill | File | Status | Notes |
|---|---|---|---|
| 00 | skill_00_zindi_monitor.py | DONE | legacy alias: skill_00_discussion_monitor.py |
| 01 | skill_01_integrity.py | DONE | |
| 02 | skill_02_intake.py | DONE | |
| 03 | skill_03_legality.py | PARTIAL | lives in competitions/scripts/ not zindian/skills/ |
| 07 | skill_07_features.py | DONE | variant-13/14 bug unresolved |
| 08 | skill_08_anchor.py | DONE | bugs fixed 2026-05-06 |
| 11 | skill_11_gate.py | DONE | |
| 15 | skill_15_reporter.py | DONE | |
| 16 | skill_16_submit.py | DONE | |
| 04 | skill_04_eda.py | TODO | |
| 05 | skill_05_cv.py | TODO | |
| 06 | skill_06_cleaning.py | TODO | |
| 09 | skill_09_calibration.py | TODO | needed before Phase 5 |
| 10 | skill_10_shap.py | TODO | SHAP run ad-hoc, not governed |
| 12 | skill_12_metric.py | TODO | |
| 13 | skill_13_fusion.py | TODO | HUMAN GATED — needed by May 17 |
| 14 | skill_14_inference.py | TODO | HUMAN GATED — needed by May 17 |
| 17 | skill_17_governance.py | TODO | needed by May 19 |

---

## NOTEBOOKS STATUS

All notebooks have stale column names (occurrenceStatus vs actual Occurrence Status).
None have been run and frozen. Not critical path — fix after Round 4.

| Notebook | Status | Notes |
|---|---|---|
| 01_integrity_audit.ipynb | STALE | wrong column names |
| 02_eda_anchor.ipynb | STALE | wrong column names |
| 03_anchor_baseline.ipynb | STALE | wrong column names |
| 04_baseline.ipynb | TODO | not created |
| 05_features.ipynb | TODO | not created |
| 06_calibration.ipynb | TODO | not needed yet |

---

## SUBMISSION LEDGER

| Sub ID | File | LB F1 | Rank | Compliant | Selected |
|---|---|---|---|---|---|
| LrU3Hg7g | sub_001_anchor.csv | 0.0 | — | NO | NO |
| XAU8mhWs | sub_003_anchor.csv | 0.0 | — | NO | NO |
| FgpCxZ5e | sub_004_anchor_f1_fixed.csv | 0.8767 | 291 | NO | NO |
| yDnrXdKz | variant-06_submission.csv | 0.8911 | 228 | NO | NO |
| eWDKfyBV | variant-25_submission.csv | 0.8824 | — | NO | NO |
| SVzenJsj | sub_009_anchor.csv | 0.0 | — | NO | NO |
| Bw7woyXB | sub_010_anchor.csv | 0.0 | — | NO | NO |
| WeXoXWi6 | sub_011_anchor.csv | 0.8816 | 240 | YES | YES |

Total used: 11
Must select 2 final by May 19
Second selection: TBD

---

## COMPETITION STANDING

Current rank  : 240
Best LB F1    : 0.8816 (compliant)
Top LB F1     : 0.9576
Gap to close  : 0.0760
Days remaining: 18

---

## SESSION START PROMPT FOR NEXT AGENT

Read competitions/ey-frogs/SKILL_STATE.json and challenge_config.json.

Tell me:
- Current dag_phase and git branch
- Anchor LB score and rank (should be 0.8816, rank 240)
- Submissions remaining today
- Which Round 4 variants have completed
- Whether any variant passed the gate this round

Then run: grep -n "variant-27\|variant-28\|variant-29\|variant-30" zindian/skills/skill_07_features.py
Confirm all Round 4 variants are defined.

Then fix variant-13/14 bug before running any Round 4 variants.

Enter Plan mode. Do not write any code until I approve the plan.
