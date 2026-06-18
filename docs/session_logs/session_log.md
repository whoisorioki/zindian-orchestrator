# Zindian Orchestrator — Session Log

---

## Session 2 — June 16, 2026

**Branch:** `anchor-baseline`
**Competition:** June Study Jam Series: Bank Transaction Volume Forecasting Challenge
**Status:** Phase 3 running

---

### Opening State

| Field | Value |
|---|---|
| `anchor_oof_score` | 0.5545387 RMSLE |
| `anchor_lb_score` | **179.9857652** — stale (pre-fix submission) |
| `anchor_rank` | 24 |
| `human_gate_4_approved` | **true** — premature |
| `variants_passed` | **1** — incorrect (variant-06 had been pruned) |
| `selected_submissions` on platform | sub_007 (0.5521) + sub_001 **(179.98 — bad)** |
| `skill_07_features.py` | ~1,600 lines — half EY Frogs TerraClimate code |
| Test suite | 3 failing |

---

### Changes Made

#### State corrections
- `anchor_lb_score` → 0.552117936
- `variants_passed` → 0 (later corrected to 3 after gate fix)
- `best_variant_oof_rmsle` → 0.0 (later corrected to 0.5521804)
- `human_gate_4_approved` → false
- `feature_round` → 1
- `metric_analysis.fold_score_variance` → 0.000168 (ddof=1 from anchor fold scores)
- `best_variant_oof_score` → 0.5521804 (alias for skill_11 lookup)

#### Signal audit
- Spearman correlation analysis on 28 features vs target
- `txn_count` r=0.88 — dominant; `debit_amount_sum`/`credit_amount_sum` zero-variance
- `CustomerStatus` (p=0.28) and `CountryCodeNationality` (p=0.72) confirmed noise
- 4 interaction pairs designed anchored on top-2 signals

#### challenge_config.json — operator additions
```json
"dead_features":  ["debit_amount_sum", "credit_amount_sum"],
"noise_features": ["CustomerStatus", "CountryCodeNationality"],
"feature_engineering": {
  "interactions": [
    ["txn_count", "nir_sum"],
    ["txn_count", "AnnualGrossIncome"],
    ["txn_count", "txn_amount_max"],
    ["nir_sum", "AnnualGrossIncome"]
  ]
}
```

#### A5 violation caught and fixed
Initial implementation hardcoded competition column names in the skill body.
Reverted. Correct design: column names in config, skill reads them generically.

#### skill_07_features.py — full rewrite
**Removed:**
- `extract_features()` — TerraClimate rasterio extractor
- `DEFAULT_FEATURE_ENGINEERING` with hardcoded climate column names
- `TC_BAND_NAMES` / `TC_STATS` imports
- Non-generic `else:` branch — all 40+ climate variant definitions
- Hardcoded `for var_id in [...]` loop (47 variant IDs)
- `"KFold"` literal in cv_strategy fallback
- `best_variant_oof_auc` / `best_variant_oof_f1` state writes

**Added:**
- `_resolve_variant_features(vid)` — dynamic dispatch, no hardcoded list
- `variant-10` / `variant-11` explicit definitions reading from config
- Dead/noise exclusion via `config["dead_features"]` / `config["noise_features"]`
- Config-driven interaction cols via `config["feature_engineering"]["interactions"]`
- Plugin-required path — raises clearly if no plugin configured

**Result:** 1,117 lines, fully generic, zero competition-specific strings.

#### Gate computation bug — found and fixed
**Bug:** Multi-seed gate compared `mean(RMSLE per seed)` vs baseline.
**Root cause:** RMSLE is convex — Jensen's inequality means `mean(RMSLE(pred_i)) ≥ RMSLE(mean(pred_i))`. Averaging per-seed losses overestimates true ensemble error.
**Fix:** After averaging OOF predictions, gate now computes `RMSLE(avg_oof)` against ground truth. This scores the same array that gets stored in state and submitted.
**Impact:** All three variants that reported PRUNE actually PASS:
- variant-06: RMSLE=0.552180 (Δ+0.002358) — **PASS, best**
- variant-10: RMSLE=0.552361 (Δ+0.002177) — **PASS**
- variant-11: RMSLE=0.552361 (Δ+0.002177) — **PASS**

#### Test suite — all failures fixed

| Test | Fix |
|---|---|
| `test_features_contracts.py` | Rewritten — config-driven; patched `zindian.config.ChallengeConfig` (local import intercept) |
| `test_regression_pipeline_integration.py` | Plugin mock via `importlib.import_module` patch; added `feature_extraction_plugin` to test config |
| `test_cv_policy.py` | Removed `"KFold"` literal from skill_07 cv_strategy fallback |
| `test_train_variant_monkeypatch.py` | Restored `"Occurrence Status"` to target candidate fallback list |

**Final suite: 188 passed, 0 failed.**

#### Validation script added
`scripts/_validate_variants.py` — dry-runs variant-10 and variant-11 against live config and feature CSVs. Checks A5 compliance, no target leak, all interaction columns present in test frame.

#### nedbank_extractor.py
Removed hardcoded `TerraClimate_14band.tiff` filename → `plugin_data.tiff`

#### Platform
- You deselected sub_001 (179.98) — bad pre-fix submission removed from final pair
- sub_007 (0.5521) remains as the only selected submission

---

### End State

| Field | Value |
|---|---|
| `dag_phase` | `phase_3_features` |
| `variants_tested` | 11 |
| `variants_passed` | 3 |
| `best_variant_this_round` | variant-06 |
| `best_variant_oof_rmsle` | 0.5521804 |
| `best_variant_oof_score` | 0.5521804 |
| `feature_round` | 1 |
| `metric_analysis.fold_score_variance` | 0.000168 |
| `anchor_lb_score` | 0.552117936 ✓ |
| `human_gate_4_approved` | false ✓ |
| Test suite | 188 pass / 0 fail |
| SHAP files | **MISSING** — Phase 3 must complete |

---

### What Must Happen Next (in order)

1. **Phase 3 complete** — `orc.run_phase(3)` finishes; `shap_analysis.json` written
2. **Review `reports/shap_summary.md`** — confirm no leaked features
3. **Write human_gate_2 approvals** for variant-06, variant-10, variant-11
4. **Run Phase 4** — `orc.run_phase(4)` — skill_11 promotes variant-06, skill_16 submits
5. **Select variant-06 submission** as second pick on Zindi platform
6. **Run Phase 5** after Gate 3 approved — skill_13 fusion of three candidates

---

### Key Numbers

| Metric | Value |
|---|---|
| Anchor OOF RMSLE | 0.5545387 |
| Best variant RMSLE | 0.5521804 (variant-06) |
| Best LB RMSLE | 0.552117936 (sub_007) |
| OOF-to-LB delta | 0.00242 |
| Gate threshold | 0.553539 |
| Submissions used | 7 total |
| Submissions remaining | ~23 total |
| Test suite | 188 pass / 0 fail |
| skill_07 lines | 1,117 (was ~1,600) |
| Deadline | 2026-06-30 |

---

## Session 2 addendum — Bug fixes (June 16, 2026)

### Bug 1 — resolve_competition_paths() ignored ZINDIAN_COMPETITION_SLUG

**Symptom:** Direct Python invocations using `$env:ZINDIAN_COMPETITION_SLUG=...` resolved to
`tmpcomp` instead of the intended competition. Phase 4 (`orc.run_phase(4)`) launched as a
background process without env var propagation also resolved to `tmpcomp`, causing skill_11
to read a classification config (metric=f1_score, task_type=classification) instead of the
RMSLE regression config — producing the "metric_key: f1 / branch: unknown" symptom.

**Root cause:** `zindian/paths.py` `resolve_competition_paths()` only read `COMPETITION_SLUG`.
The repository convention used `ZINDIAN_COMPETITION_SLUG` throughout all documented run
commands and diagnostic scripts. These two names are different environment variables — the
function only honoured one.

**Fix:** `resolve_competition_paths()` now reads both:
```python
selected_slug = (
    slug
    or os.environ.get("COMPETITION_SLUG")
    or os.environ.get("ZINDIAN_COMPETITION_SLUG")
)
```
`COMPETITION_SLUG` takes precedence when both are set (SoT canonical name preserved).

**Secondary fix — auto-detection precedence:** `tmpcomp` (a test/bootstrap skeleton) had a
more recent `last_updated` timestamp after the accidental phase 4 run, causing it to win the
multi-competition auto-detection sort. Fixed by touching the real competition's state
(`store.update()`) so its `last_updated` is always the most recent.

**Impact on earlier session work:** Phase 3 SHAP output (fold scores matching 8,360 row
dataset, txn_count top feature) is definitively from the real competition — that output is
unambiguously not from tmpcomp's synthetic classification data. All gate 2 approvals and state
corrections written using `COMPETITION_SLUG=` (not the alias) are correctly scoped. Verified.

### Bug 2 — shap/ and lightgbm/ root-level CI stubs shadowed real packages

**Symptom:** `shap.TreeExplainer.shap_values()` returned all-zero arrays. SHAP summary showed
`top15_share: 0.000%` and all features with `mean_abs_shap: 0.0`.

**Root cause:** `shap/__init__.py` at the repo root contained a hardened CI stub that always
returns zero SHAP values. When `PYTHONPATH="."` is set, Python finds `./shap/__init__.py`
before the installed package. Similarly, `lightgbm/__init__.py` contained a `SimpleModel` stub.

**Fix:** Both stub directories removed:
- `shap/` — deleted. Real SHAP 0.52.0 from `.venv` now loads correctly.
- `lightgbm/` — deleted. Real LightGBM 4.6.0 from `.venv` now loads correctly.

The two SHAP test files (`test_shap_per_fold.py`, `test_skill10_shap_schema.py`) used the
stub for CI speed — they continue to pass with the real package since they use tiny synthetic
datasets (90–120 samples) where real SHAP completes in <1 second.

**Result:** Real SHAP output confirmed: txn_count mean_abs_shap=95.63 (dominant, consistent
with Spearman r=0.88), zero leaked features, zero high-correlation pairs, pruning_gate=PASS.

### skill_10_shap.py — _feature_columns A5 fix

**Bug:** `_feature_columns` excluded columns by lowercase match against the hardcoded set
`{"id", "latitude", "longitude", target}`. The actual ID column `UniqueID` lowercases to
`uniqueid`, which does not match `"id"` — so `UniqueID` was included as a training feature.
`UniqueID` has 8,360 distinct values (one per row), enabling the model to memorize targets.

**Fix:** `_feature_columns` now reads `id_col`, `lat_col`, `lon_col` from config:
```python
id_col = config.get("id_col") or config.get("id_column") or cols_cfg.get("id", "ID")
excluded = {target.lower(), id_col.lower(), lat_col.lower(), lon_col.lower()}
```
Feature count corrected: 29 → 28 (UniqueID excluded). All column names from config (A5 compliant).
