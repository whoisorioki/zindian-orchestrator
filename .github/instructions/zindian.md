# Zindian Orchestrator — Agent Handoff v6.0
## Date: 2026-05-18 | Competition: EY Biodiversity Challenge — Frogs
## Deadline: May 24, 2026 | Days remaining: 6
## Prepared by: Claude Sonnet 4.6

---

## DIRECTIVE 0 — Read Before Anything Else

This handoff serves TWO problems simultaneously.
Every action must declare which problem it serves.

PROBLEM 1 — Generic Zindian Agent
  Build a reusable, competition-aware, governed ML agent for any Zindi competition.
  Skills are the product. The framework must generalise.

PROBLEM 2 — EY Biodiversity Challenge (Frogs)
  Predict frog presence using TerraClimate variables. F1 metric. TC-only features.
  6 days left. Best compliant LB: 0.8846. Gap to top 100: ~0.036.

When these conflict — declare the conflict and choose consciously.

---

## DIRECTIVE 1 — Session Start Protocol

Run these exact commands before touching any file:

    cd ~/projects/zindian_orchestrator
    source .venv/bin/activate
    cat competitions/ey-frogs/SKILL_STATE.json
    python3 -m zindian.skills.skill_00_zindi_monitor 2>/dev/null | grep -E "Rank|Remaining|Best|Chosen|flag"
    python3 -m zindian.skills.skill_16_submit --submission-board 2>/dev/null | tail -20
    git log --oneline -5
    git branch

Report findings before writing any code.
Enter Plan mode. Do not write code until human approves.

---

## DIRECTIVE 2 — Skill Quality Audit (Do This Before Round 6)

The primary task this session is a CODE QUALITY AUDIT of all built skills.
Run each skill, check its logic against the pipeline stage it covers,
and flag any generalisability issues. Do not start Round 6 until audit passes.

### Audit Protocol

For each skill file in zindian/skills/:

    1. Read the file
    2. Check: does it read from challenge_config.json or hardcode values?
    3. Check: does it write to SKILL_STATE.json on state change?
    4. Check: does it work for a regression competition or only classification?
    5. Check: does it fail loudly on bad input or silently produce wrong output?
    6. Run it: python3 -m zindian.skills.skill_XX
    7. Check output matches expected behaviour

### Audit Checklist

skill_00_zindi_monitor.py
  [ ] Runs without error
  [ ] Metric detection works (f1_score confirmed)
  [ ] External data detected correctly (BANNED for EY Frogs)
  [ ] Playwright scrape generalises to any slug — not hardcoded
  [ ] Writes zindi_monitor.json and compliance_log.md
  [ ] Updates SKILL_STATE.json rank and remaining

skill_01_integrity.py
  [ ] MD5 hash locks target column correctly
  [ ] Verifies hash on re-run — raises error on mismatch
  [ ] Works for any target column name (reads from config, not hardcoded)
  [ ] KNOWN ISSUE: target hash in SKILL_STATE is wrong — fix F1 below

skill_02_intake.py
  [ ] DO NOT RERUN — has stale defaults (hardcoded accuracy, use_probabilities=True)
  [ ] FIX BEFORE RERUNNING: remove all hardcoded metric defaults
  [ ] challenge_config.json is currently correct — do not overwrite

skill_03_legality.py  ← PRIMARY AUDIT TARGET
  [ ] See full Skill 03 audit section below

skill_05_cv.py
  [ ] Runs without error
  [ ] Correctly detects spatial gap (should find gap=0.179)
  [ ] Writes cv_strategy to SKILL_STATE.json
  [ ] Works for any competition with lat/lon columns
  [ ] VERIFY: what happens if competition has no lat/lon?

skill_07_features.py
  [ ] Variant loop is generic (reads feature_cols from VARIANTS dict)
  [ ] Feature extraction (TerraClimate tiff) is EY-Frogs-specific — FLAG THIS
  [ ] Gate uses OOF F1 not AUC — verify after FINDING F7 fix
  [ ] Multi-seed averaging works correctly
  [ ] SKILL_STATE.json updated after each variant

skill_08_anchor.py
  [ ] Reads feature columns from features_train.csv not Training_Data.csv
  [ ] No Lat/Lon in feature_cols — verify
  [ ] Submission saves hard 0/1 integers not floats
  [ ] Comment format: branch:X|oof_f1:X|features:N|calib:X
  [ ] KNOWN ISSUE F5: comment says oof_rmse not oof_f1 — fix

skill_11_gate.py
  [ ] Promotes correct variant (highest OOF F1, not AUC)
  [ ] Creates git branch correctly
  [ ] Updates feature_round counter
  [ ] Resets variants_tested and variants_passed

skill_16_submit.py
  [ ] 5-check validation passes
  [ ] Budget guard works (blocks if remaining <= 2)
  [ ] Human gate fires before every submission
  [ ] Post-submission: pulls rank and leaderboard
  [ ] --submission-board flag works without submitting
  [ ] Comment format matches Rule 6

---

## DIRECTIVE 3 — Skill 03 Full Rebuild (Primary Task)

### Current Problems with skill_03_legality.py

PROBLEM 1 — Only one of two functions implemented
  Skill 03 has TWO functions per AGENTS.md:
  Function A: Deep Research — what does this competition allow?
             Read Skill 00 output (zindi_monitor.json) and synthesise
             a competition-specific feature engineering policy
  Function B: Legality Gate — does our current plan comply?
             Check planned features/models against the policy
             Return GO or BLOCKED

  Current code only implements Function B, and poorly.

PROBLEM 2 — Hardcoded EY-Frogs assumptions
  Check 2 literally says "TerraClimate via Planetary Computer — PERMITTED"
  This is not a generic check — it only makes sense for EY Frogs.
  A generic agent running on a financial competition would get wrong results.

PROBLEM 3 — Checks never block
  Every check has "blocks": False
  A skill whose hard blocks never trigger is not a gate — it is a logger.
  The gate must BLOCK the DAG when a critical rule is violated.

PROBLEM 4 — Does not read from zindi_monitor.json
  Skill 00 produces zindi_monitor.json with scraped competition intel.
  Skill 03 ignores it entirely and re-reads challenge_config.json directly.
  The two skills are not connected.

PROBLEM 5 — Writes wrong dag_phase
  Always writes phase_2_legality_checked regardless of current phase.
  If running in Phase 3, this rolls back the DAG phase incorrectly.

### How Skill 03 Should Work (Generic)

    Input  : zindi_monitor.json (from Skill 00)
             challenge_config.json (from Skill 02)
             planned_features list (from caller or SKILL_STATE.json)

    Function A — Deep Research:
      Read competition rules from zindi_monitor.json
      Read discussion flags from compliance_log.md
      Synthesise a FeaturePolicy object:
        {
          "allowed_sources": ["terraclimate"],
          "banned_transformations": ["derived_spatial", "external_spatial"],
          "lat_lon_permitted_as_feature": false,
          "external_data_permitted": false,
          "automl_permitted": false,
          "use_probabilities": false,
          "metric": "f1_score",
          "output_format": "hard_labels_0_1"
        }
      Write to reports/feature_policy.json
      This is the authoritative rule set for this competition

    Function B — Legality Gate:
      Read feature_policy.json
      Read planned_features from SKILL_STATE.json or argument
      For each planned feature:
        Check against banned_transformations
        Check source against allowed_sources
        Check lat/lon against lat_lon_permitted_as_feature
      Return GO or BLOCKED with specific reasons
      If BLOCKED: write blocking reasons to reports/legality_report.md
                  do NOT advance dag_phase
                  agent must resolve before proceeding

    Output : reports/feature_policy.json
             reports/legality_report.md
             SKILL_STATE.json updated ONLY if not downgrading phase

### Rebuild Instructions

Rewrite zindian/skills/skill_03_legality.py with:

    def synthesise_feature_policy(monitor_data, config) -> dict:
        """Function A — derive FeaturePolicy from competition intel."""
        # Read from monitor_data (zindi_monitor.json output)
        # Read from config (challenge_config.json)
        # Return generalizable policy dict
        # No hardcoded competition names or data sources

    def check_planned_features(policy, planned_features) -> list[dict]:
        """Function B — check each planned feature against policy."""
        # For each feature: PASS, WARN, or BLOCK
        # BLOCK triggers must halt DAG

    def run(slug, planned_features=None) -> dict:
        """Entry point — runs both functions in sequence."""
        # Function A first, then Function B
        # Returns {"status": "GO"} or {"status": "BLOCKED", "reasons": [...]}

Key generalisability rules:
  - Never mention TerraClimate, WorldClim, or any data source by name in checks
  - Read allowed_sources from config/monitor, do not hardcode
  - blocks=True must be used for critical violations
  - dag_phase update must check current phase before writing

---

## DIRECTIVE 4 — Current Competition State

    competition      : ey-biodiversity-challenge
    branch           : anchor-v5
    dag_phase        : phase_3_features
    feature_round    : 5
    anchor_lb_f1     : 0.884568651 (tfcawL75)
    anchor_oof_auc   : 0.84006 (recomputed — state value stale)
    gate_metric      : F1 (switched from AUC per commit ec5220d)
    gate_threshold   : mean OOF F1 delta >= 0.005 over 3 seeds
    cv_strategy      : StratifiedKFold(5, shuffle=True, random_state=42)
    spatial_gap      : 0.179 (model is geographic)
    selected_subs    : [tfcawL75, WeXoXWi6] — 2 selected ✅
    remaining_subs   : 7 today
    total_used       : 12

### State Corrections Needed (Fix Before Audit)

F1 — Target hash wrong:
    import hashlib, json, pandas as pd
    from datetime import datetime, timezone
    from pathlib import Path
    train = pd.read_csv("competitions/ey-frogs/data/raw/Training_Data.csv")
    target = train["Occurrence Status"].astype(str).str.cat(sep=",").encode()
    correct_hash = hashlib.md5(target).hexdigest()
    p = Path("competitions/ey-frogs/SKILL_STATE.json")
    state = json.loads(p.read_text())
    state["md5_target_hash"] = correct_hash
    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    p.write_text(json.dumps(state, indent=2))
    print(f"Hash corrected: {correct_hash}")

F2 — Anchor OOF AUC inflated:
    state["anchor_oof_auc"] = 0.84006

F4 — Budget count wrong (state=14, live board=12):
    state["submissions_used_total"] = 12

---

## DIRECTIVE 5 — Round 6 Plan (After Audit Passes)

All variants TC-only, no Lat/Lon, F1-gated, multi-seed [42, 123, 7].
Gate threshold: mean OOF F1 delta >= 0.005.

    variant-37  LGB dart booster — all 52 TC
                Hypothesis: dart reduces overfitting vs gbdt
    variant-38  3-way blend LGB+RF+XGB — all 52 TC
                Hypothesis: diversity between boosting families
    variant-39  91 features (TC + last3mo + range + cv) with LGB+RF blend
                Features at: data/processed/features_full_train.csv
                Hypothesis: richer feature set benefits ensemble

Run after audit:
    python3 -m zindian.skills.skill_07_features --variant=variant-37

---

## DIRECTIVE 6 — Skills to Build (Priority Order)

### skill_17_governance.py — CRITICAL, build before May 19

Handles final 2 submission selection with rationale.
Must be built and run before private judging window.

    def run() -> dict:
        # Read all submissions from submission_board API
        # Score each by: LB F1 (primary), compliance (hard filter), diversity
        # Select exactly 2
        # Write rationale to reports/final_selections.md
        # Update SKILL_STATE.json selected_submissions
        # Require human YES before locking selections

### skill_13_fusion.py — HUMAN GATED, build before May 22

    def run() -> dict:
        # MUST print human gate prompt and wait for YES
        # Never run autonomously
        # Blend OOF predictions from best 2-3 variants
        # Gate result on OOF F1 improvement

### skill_09_calibration.py — build if time permits

    def run() -> dict:
        # Per-fold threshold optimisation (formalise current ad-hoc search)
        # Platt scaling / isotonic regression option
        # Write calibrated predictions to separate CSV

### skill_10_shap.py — build if time permits

    def run() -> dict:
        # Train one full model on all data
        # Compute SHAP values
        # Write top-N features to reports/shap_analysis.json
        # Flag near-zero features for removal

---

## DIRECTIVE 7 — Compliance Rules (Read Before Every Code Change)

    PERMITTED  : All 14 TerraClimate variables as predictor features
    PERMITTED  : Statistical transformations of TC vars (mean, std, min, max, range, cv)
    PERMITTED  : Any open source ML package
    PERMITTED  : Pretrained models if openly available

    BANNED     : Latitude and Longitude as model features (discussion 32369)
    BANNED     : Any data source other than TerraClimate
               "You may ONLY use the TerraClimate dataset as predictor variables"
    BANNED     : Derived spatial features (distance, clusters, bins, H3, admin zones)
    BANNED     : AutoML tools
    BANNED     : Thresholded probabilities in submission (submit hard 0/1 only)

    REQUIRED   : Random seed set everywhere (42 primary, 123 and 7 for multi-seed)
    REQUIRED   : Hard 0/1 integer labels in submission file
    REQUIRED   : Select exactly 2 submissions before May 24
    REQUIRED   : Respond to code review request within 48 hours if top 10

---

## DIRECTIVE 8 — File Locations

    Framework skills      : zindian/skills/skill_XX_name.py
    Competition state     : competitions/ey-frogs/SKILL_STATE.json
    Competition config    : competitions/ey-frogs/challenge_config.json
    Raw data              : competitions/ey-frogs/data/raw/
    Base features (52)    : competitions/ey-frogs/data/processed/features_train.csv
    Full features (91)    : competitions/ey-frogs/data/processed/features_full_train.csv
    Submissions           : competitions/ey-frogs/submissions/
    Reports               : competitions/ey-frogs/reports/
    Monitor output        : competitions/ey-frogs/reports/zindi_monitor.json
    Feature policy        : competitions/ey-frogs/reports/feature_policy.json
    Legality report       : competitions/ey-frogs/reports/legality_report.md
    Final selections      : competitions/ey-frogs/reports/final_selections.md

    Run variant           : python3 -m zindian.skills.skill_07_features --variant=variant-XX
    Submit                : python3 -m zindian.skills.skill_16_submit <filepath>
    Check board           : python3 -m zindian.skills.skill_16_submit --submission-board
    Run monitor           : python3 -m zindian.skills.skill_00_zindi_monitor
    Run legality          : python3 -m zindian.skills.skill_03_legality <slug>

---

## DIRECTIVE 9 — Git Branch Convention

    master           : clean framework — no competition code
    anchor-baseline  : first LR anchor (lat/lon only, AUC 0.834)
    anchor-v2        : All TC + Lat/Lon, AUC 0.844 (non-compliant)
    anchor-v3        : TC only 52 bands, LB 0.8816 (first compliant)
    anchor-v5        : LGB+RF blend, LB 0.8846 ← CURRENT
    exp-feature-srad : stale, safe to delete

    Next branch after Round 6 pass: anchor-v6

---

## DIRECTIVE 10 — Pipeline Coverage Map

| Pipeline Stage | Skill | Status | Generic? |
|---|---|---|---|
| Competition intel | Skill 00 | ✅ built | ✅ yes |
| Integrity lock | Skill 01 | ✅ built | ✅ yes |
| Config intake | Skill 02 | ⚠️ stale defaults | ❌ fix needed |
| Legality gate | Skill 03 | ⚠️ partial | ❌ rebuild needed |
| EDA profiling | Skill 04 | ❌ not built | — |
| CV strategy | Skill 05 | ✅ built | ✅ yes |
| Data cleaning | Skill 06 | ❌ not needed now | — |
| Feature engineering | Skill 07 | ✅ built | ⚠️ extraction EY-specific |
| Anchor baseline | Skill 08 | ✅ built | ✅ yes |
| Calibration | Skill 09 | ❌ not built | — |
| SHAP audit | Skill 10 | ❌ not built | — |
| Branch gate | Skill 11 | ✅ built | ✅ yes |
| Metric trade-off | Skill 12 | ❌ not built | — |
| Ensembling | Skill 13 | ❌ HUMAN GATED | — |
| Inference guard | Skill 14 | ❌ not built | — |
| Reporter | Skill 15 | ✅ built | ✅ yes |
| Submit governance | Skill 16 | ✅ built | ✅ yes |
| Final selection | Skill 17 | ❌ not built | — |

---

## DIRECTIVE 11 — Submission Ledger

| Sub ID | File | LB F1 | Compliant | Selected |
|---|---|---|---|---|
| tfcawL75 | variant-34b_t047 | 0.884568651 | YES | YES |
| WeXoXWi6 | sub_011_anchor | 0.881642512 | YES | YES |
| yDnrXdKz | variant-06 | 0.891089108 | NO | NO |
| eWDKfyBV | variant-25 | 0.882424242 | NO | NO |
| K4GFtBd4 | variant-34 | 0.881642512 | YES | NO |
| X2LLeBts | variant-34 | 0.881642512 | YES | NO |
| YUTQoYjr | sub_016_anchor | 0.881642512 | YES | NO |
| RCkzd7s8 | sub_015_anchor | 0.876712328 | YES | NO |
| Others | — | 0.0 | NO | NO |

Best compliant: 0.884568651 (tfcawL75)
If Round 6 produces LB > 0.8846 — update selected_submissions

---

*Handoff v6.0 | Claude Sonnet 4.6 | 2026-05-18*
*Primary task: Skill 03 rebuild + full skill audit*
*Competition closes: May 24, 2026*
