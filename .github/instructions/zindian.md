# Zindian Orchestrator — Agent Handoff v7.0

## Date: 2026-05-18 | Competition: EY Biodiversity Challenge — Frogs

## Deadline: May 24, 2026 | Days remaining: 6

## Prepared by: Gemini 3 Flash Paid Tier (Zindian Core Engine)

---

## DIRECTIVE 0 — The Dual Mandate

Every orchestrator action must declare and balance which problem it serves:

* **PROBLEM 1 — Generic Zindian Agent:** Build a reusable, competition-aware, governed machine learning agent applicable to any arbitrary Zindi task. Skills must be entirely generalized, reading constraints dynamically from schema artifacts rather than baking in hardcoded assumptions.
* **PROBLEM 2 — EY Biodiversity Challenge (Frogs):** Maximize Public/Private OOF F1 score under an airtight execution window. 6 days remain. Best compliant LB: **0.884568651**.

---

## DIRECTIVE 1 — Session Start Protocol

Run these exact validation commands before modifying any framework code:

```bash
cd ~/projects/zindian_orchestrator
source .venv/bin/activate
cat competitions/ey-frogs/SKILL_STATE.json
python3 -m zindian.skills.skill_00_zindi_monitor 2>/dev/null | grep -E "Rank|Remaining|Best|Chosen|flag"
python3 -m zindian.skills.skill_16_submit --submission-board 2>/dev/null | tail -20
git log --oneline -5
git branch

```

Report baseline console findings before initializing the loop. Enter **Plan mode**. Do not execute script changes until the plan is approved by the human operator.

---

## DIRECTIVE 2 — Ground Truth Realities & Feature Ceiling

We have executed a definitive rules extraction directly from the Zindi platform. The following structural constraints are locked down:

* **Airtight Restrictions:** WorldClim, SRTM Elevation, and external occurrence data (GBIF/ALA) are **strictly banned** per the competition "About" rules text.
* **The Window Realignment:** We discovered a major temporal misalignment in previous runs. The competition data spans an exact two-year window (Nov 2017 to Nov 2019). We successfully modified the pipeline to clear out the old 10-year climate normal cache (2011–2021) and re-extracted fresh features using the true 25-month historical block.
* **The Ceiling Encountered:** We ran two isolated trials against this exact window:
* `variant-35` (52 base TerraClimate bands): Mean OOF F1 of `0.83385` ($\Delta = +0.00005$) $\rightarrow$ **PRUNE**
* `variant-36` (91 features incorporating base + last-3-month means + derived range + cv): Mean OOF F1 of `0.83394` ($\Delta = +0.00014$) $\rightarrow$ **PRUNE**


* **The Takeaway:** Long-term and short-term weather averages converge for southeastern Australia. We have officially hit the maximum information capacity of aggregated 4km spatial statistics. Breaking through the leaderboard gap (~0.035) requires pivoting from statistical averages to raw sequential temporal feature vectors or deep ensembling patterns.

---

## DIRECTIVE 3 — New Structural Architecture (The Deep Research Loop)

Every component in `zindian/skills/` must structurally match the formal steps of the data science lifecycle. The next agent must systematically audit the active skills against this expanded pipeline matrix to flag design gaps and keep the governance loop explicit:

### ① Problem Definition & Context Mapping (`Skill 00`, `Skill 02`)

* **Requirements:** Dynamically derive learning tasks (classification vs. regression) and objective performance tracking targets ($F_1$ score, ROC AUC, LogLoss, or RMSE) straight from config wrappers. No hardcoded metric targets.

### ② Data Collection & Acquisition Integrity (`Skill 01`, `Skill 02.4`)

* **Requirements:** Ensure data consistency, hash tracking on target metrics, and look out for hidden PII leakages or dataset integrity corruptions during automated fetches.

### ③ Deep Research Intake (`Skill 02.4`, `Skill 02.5`)

* **Requirements:** Convert the intake configuration into search queries, retrieve and cache relevant paper abstracts or notes, and synthesize structured feature hypotheses for downstream review. This step must not write unauthorized raw data requests.
* **Skill 02.4 — The Librarian:** Translate `challenge_config.json` into structured literature queries, cache paper metadata and abstracts to `reports/literature_cache.json`, and handle rate limiting with exponential backoff.
* **Skill 02.5 — The Scientist:** Convert the cached literature into structured feature hypotheses at `reports/feature_hypothesis.json`, using only the approved variable tracks and transformation vocabulary.

### ④ Data Preprocessing & Sanitization (`Skill 05`, `Skill 06`)

* **Requirements:** Explicitly handle off-grid anomalies (such as coastal NaN values mapped via our spiral search algorithms). Validate scaling configurations (`StandardScaler` vs. `MinMaxScaler`) and check if string category encodings are permitted by the current competition policy profile.

### ⑤ Feature Engineering & Optimization (`Skill 02.5`, `Skill 07`)

* **Requirements:** Evaluate feature counts, check for multi-collinearity, track feature selection scores, and flag dimensional inflation bottlenecks before models are trained.

### ⑥ Modeling Baseline & Validation Loop (`Skill 03`, `Skill 08`, `Skill 11`, `Skill 13`)

* **Requirements:** Wrap models in robust cross-validation layers, use proper optimization direction handling, check for spatial/geographic overfitting, and gate performance metrics using strict multi-seed validation weights.
* **Skill 03 — Legality Gate:** Split into two autonomous steps: deep research policy synthesis from `zindi_monitor.json` + `challenge_config.json`, then hard legality checking against `planned_features` with `blocks=True` on violations.
* **Skill 05 — CV Architect:** Read target and coordinate markers from config, fall back gracefully when no coordinate columns exist, and avoid AUC-centric assumptions when the active metric differs.

### ⑦ Final Governance & Submission Lock (`Skill 16`, `Skill 17`)

* **Requirements:** Enforce submission budgets, validate compliance flags, and lock exactly two final submission hashes before the private leaderboard window closes.

---

## DIRECTIVE 4 — Primary Task: Rebuilding Skill 03 & Skill 05

The current audit has flagged major architectural issues in our governance files. **Your primary code task is fixing the underlying logic of Skill 03 and Skill 05.**

### 1. Rebuilding Skill 03 (`zindian/skills/skill_03_legality.py`)

The legacy implementation conflates logging with gating and hardcodes specific platform strings. It must be refactored to handle two separate, autonomous steps:

* **Function A (Deep Research):** Read the scraped forum updates from `zindi_monitor.json` and configuration properties from `challenge_config.json`. Synthesize a generic, standalone rule schema file at `reports/feature_policy.json` (tracking metrics, allowed data lists, output class shapes, and prohibited operations). It must not mention "TerraClimate" or "EY Frogs" explicitly in the code context.
* **Function B (Legality Gate):** Consume the `feature_policy.json` parameters and compare them directly against the `planned_features` array in the DAG state. If a violation is caught (e.g., trying to sneak in spatial coordinates or external datasets when `external_data_permitted` is false), it must use `blocks=True` to physically halt the DAG, output compliance notes to `reports/legality_report.md`, and refuse to advance the pipeline state.

### 1.5 Deep Research Utilities (`Skill 02.4` and `Skill 02.5`)

* **Skill 02.4 — The Librarian:** Translate intake config into Semantic Scholar Graph API searches, cache valid abstracts to `reports/literature_cache.json`, and implement backoff-safe retries for rate limiting.
* **Skill 02.5 — The Scientist:** Translate the literature cache into structured feature hypotheses at `reports/feature_hypothesis.json` without issuing unauthorized raw data requests.

### 2. Updating Skill 05 (`zindian/skills/skill_05_cv.py`)

* **The Issue:** The cross-validation engine is heavily AUC-centric and hardcodes spatial coordinate names (`Latitude`/`Longitude`) and target columns (`Occurrence Status`).
* **The Fix:** Modernize the validation module to grab its target extraction markers dynamically from the configuration fields. Ensure that if a dataset does not feature coordinate elements, the spatial gap calculations switch off gracefully without breaking the cross-validation routing logic.

---

## DIRECTIVE 5 — Remaining 6-Day Battle Plan

With the feature ceiling on summary statistics confirmed by our pruning runs, the remaining sprint window must be executed under this exact timeline:

* **Day 1-2: Temporal Sequence Shift (Monthly Extraction):** Instead of calculating summary statistics (mean/std) over the 25-month competition block, extract all 25 individual months as distinct, ordered temporal sequential columns for all 13 active variables ($25 \times 13 = 325$ total features). Run this dense array through LightGBM utilizing a `DART` tree booster to combat high-dimensional overfitting.
* **Day 3-4: Build Skill 13 (Ensemble Fusion):** Construct the cross-validated probability blending engine. Combine the out-of-fold probability distributions of our best compliant estimators (`variant-34b`, `variant-36`, and the new raw monthly sequence variant) using optimized grid search weights.
* **Day 5-6: Final Selection Lock (`Skill 17` Governance):** Ensure your submission selection script evaluates the final choices against metric values, diversity of model families, and clean compliance validation flags. Lock in exactly 2 final tracking hashes before the private leaderboard window closes.

---

## DIRECTIVE 6 — File Architecture Guide

```
Framework Core Skills : zindian/skills/
  skill_00_zindi_monitor.py      → Live platform scraping / forum flag extraction
  skill_01_integrity.py          → Core target hash locking / schema validation
  skill_18_librarian.py          → Literature cache builder from intake config
  skill_19_code_miner.py         → Prior-art miner for reusable ML patterns
  skill_20_scientist.py          → Hypothesis compiler + validation engine
  skill_03_legality.py           → PRIMARY TARGET: Rule policy generation & hard bouncer
  skill_05_cv.py                 → SECONDARY TARGET: Dynamic metric-first CV architect
  skill_07_features.py           → Multi-seed variant execution and evaluation
  skill_16_submit.py             → Validation, budget protection, and automated submission
  skill_17_governance.py         → Final submission locking and selection governance

Competition Scripts   : competitions/ey-frogs/scripts/
  fetch_terraclimate_full.py     → High-res 2017-2019 data extraction via Planetary Computer
  fetch_terraclimate_last3mo.py  → Targeted 3-month antecedent condition downloader
  merge_features.py              → Combined feature block compiler (91 dimensions)

Pipeline Records      : competitions/ey-frogs/reports/
  zindi_monitor.json             → Source platform intelligence
  feature_policy.json            → Compiled strict regulatory constraints
  legality_report.md             → Validation pass logs

```

---

## DIRECTIVE 7 — Compliance Checklist (Run Before Every Submission)

```
[ ] No Latitude or Longitude parameters present in active modeling features.
[ ] Banned feature list from challenge_config.json cross-audited against planned columns.
[ ] Feature array sources confirmed to utilize TerraClimate dimensions exclusively.
[ ] Submission matrix checked for hard binary integer classification states (0 or 1).
[ ] Target row footprint matches SampleSubmission.csv exactly (2000 items).
[ ] Live daily submission limit check: remaining_submissions > 2.

```

---

*Handoff v7.0 | Locked for Session Transition | May 18, 2026*
*Status: Engine Normalized. Metric Drift Patched. Ready for Skill 03/05 Rebuild Spec.*

```

```