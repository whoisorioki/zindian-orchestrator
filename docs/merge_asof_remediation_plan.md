# `merge_asof` Row-Scrambling Remediation Plan

**Status:** PLAN — awaiting approval. No pipeline files mutated.
**Date:** 2026-09-27
**Competition:** `climate-risk-health-prediction-challenge`
**Branch:** `anchor-baseline`
**Predecessor doc:** [`competition_state_audit_climate_risk_health_2026-09-20.md`](competition_state_audit_climate_risk_health_2026-09-20.md) (§12 Round 3)

---

## 0. Scope

A defect class in which a date-joined exogenous feature is **silently
row-scrambled** relative to the labels it is attached to. Confirmed to have
corrupted 14 branches. This plan covers the shared fix, the two culprit
plugins, full branch regeneration, and the structural guards that prevent
recurrence.

**Out of scope:** pipeline architecture changes, new dependencies, anything
requiring an SoT patch without one.

---

## 1. Round 4 Findings — train/test data flow + CV provenance

### 1.1 The row-order contract is positional and unguarded

Traced end to end. Train and Test are **not** reindexed against each other
anywhere — the only coupling is positional, through
`data/processed/features_{train,test}_{branch}.csv`.

| Producer | Behaviour | Verdict |
|---|---|---|
| `plugins/tabular_extractor.py` | Reads Train/Test, drops `ID` + banned cols, writes both. No sort, no merge. | **CLEAN** — order preserved |
| `plugins/macro_stress_extractor.py` | Sorts **both** train and test by date → `merge_asof` → positional `.loc[original.index]` restore | **BUGGY** (config default, committed) |
| `plugins/advanced_spatial_temporal_extractor.py` | Same sort → asof → positional restore pattern | **BUGGY** (untracked) |

Both buggy plugins scramble **test** as well as train, consistent with the
observed 28-coordinate test shift.

### 1.2 The "honest ID on dishonest predictions" mechanism — now located in code

`zindian/skills/skill_07_features.py` L2274–2286:

```python
raw_tr = pd.read_csv(... Train.csv) if id_col not in train_feat else None
train_id_series = train_feat[id_col] if id_col in train_feat else raw_tr[id_col]
oof_df = pd.DataFrame({id_col: train_id_series, "oof_prob": np.asarray(result["oof_probs"])})
```

Because the buggy plugins **drop the ID column**, `id_col not in train_feat`
is `True`, so skill_07 falls back to raw `Train.csv` IDs and zips them
**positionally** onto predictions derived from a scrambled matrix. The same
pattern appears at L1682–1696.

This is the precise mechanism behind the Round-3 principle that
*output-file ID correctness ≠ upstream correctness*. It is not a separate
bug — it is how the corruption becomes **invisible** to downstream ID checks.

**Consequence:** no ID-based assertion on any artifact can ever catch this
class of defect. Only an assertion that re-derives features from raw inputs
can.

### 1.3 Where the output *is* safe

`zindian/skills/skill_14_inference.py` L317–318 reindexes test

### 1.4 Test-set leakage in the shared trainer

`zindian/skills/_lightgbm_shared.py` fits the label encoder on
`pd.concat([train_vals, test_vals])`. This is a genuine (separate)
train/test leakage finding: the encoder sees test label values. Should be
fit on train only and applied via `.transform()` on test.

### 1.5 CV provenance — hypothesis largely REFUTED, narrowed to two branches

I suspected the anchor OOF had been computed under a silently-swapped
fallback splitter. **That is not the case**, and I am recording the refutation
explicitly so the hypothesis is not re-litigated:

- `zindian/state.py` L60–71 externalizes `cv_split_indices` to
  `scores/cv_split_indices.json` and leaves a **dict pointer** in state.
- `SkillStateStore.read()` L132–144 **re-hydrates** that pointer back into a
  list. Verified on disk: raw state value is
  `{'cv_splits_file': 'scores/cv_split_indices.json', 'count': 5}`.
- `skill_08` L207 `load_explicit_cv_splits(state)` therefore **succeeds**
  in the DAG path.
- Persisted splits have **non-uniform folds** (fold 0 = 1510 train / 995
  val). `KFold` would be uniform (1604/401). This proves genuine
  `BufferedSpatialCV` splits were in effect for the anchor and all base
  branches. **The anchor OOF is spatially honest.**

Two real defects remain:

**(a) Silent fallback in the two untracked refit scripts** — confirmed by
code reading. Both point at the wrong path:

```python
cv_split_path = DATA_DIR / "cv_split_indices.json"   # scripts/...:79, :111
```

`DATA_DIR` is `data/processed/`, where **no such file exists** (verified).
The real file is `competitions/<slug>/scores/cv_split_indices.json`. So
`refit_macro_stress_geofence.py` fell back to `KFold(5, shuffle=True,
random_state=SEED)` and `refit_residual_dlnm_specialist.py` to
`make_cv_splitter({"type": "kfold"})` — **random folds while the OOF record
carries a spatial `cv_strategy_id`.** This is a true provenance lie,
confined to `macro-stress-geofence-refit` and `residual-dlnm-specialist`.

**(b) Latent trap in `zindian/cv.py` L60–64** — `make_cv_splitter` has
branches only for `stratified` / `group` / default. There is **no
`BufferedSpatialCV` branch**, so any caller reaching it with
`type="BufferedSpatialCV"` silently receives shuffled `KFold`. Currently
unreachable in the DAG (explicit splits load first, skill_08 L207), but it
is a live footgun and directly contradicts the config.

### 1.6 Blocking syntax error

`zindian/skills/skill_07_features.py` L1317–1318 contains an uncommitted
`SyntaxError` inside a call expression. This breaks preflight **and** all 5
collection-dependent test files (0 tests currently pass). Must be fixed
first — it is the gate on every verification step in this plan.

---

## 2. Implementation plan

### Phase 0 — Unblock the test suite
1. Fix the `SyntaxError` at `skill_07_features.py` L1317–1318.
2. Run `pytest tests/ -q`; record the true baseline pass/fail count
   *before* any other change, so later regressions are attributable.

### Phase 1 — Shared ID-asserting join utility
New module `zindian/joins.py`:

```python
def sorted_asof_join(df, ext_df, *, id_col, date_col, by_cols=(),
                     tolerance=None):
    """Left-join exogenous `ext_df` onto `df` by nearest-past `date_col`,
    optionally within `by_cols` groups.

    Guarantees the returned frame is row-for-row identical in order and
    length to `df`, with `df`'s ID column intact. Raises on any violation.
    """
```

Internal guarantees, all hard-fail:
- snapshot `raw_ids = df[id_col].copy()` and `n = len(df)` before any work
- assert `ext_df` is date-sorted on `date_col`
- assert output length `== n`
- **ID round-trip assertion:**
  `assert out[id_col].reset_index(drop=True).equals(raw_ids.reset_index(drop=True))`
- assert no NaN in the newly joined key column unless `tolerance` was given

**Why ID round-trip and not `.loc` restore:** `merge_asof` returns a
non-unique index whenever the left key has ties. `.loc[RangeIndex]` then
selects *the first matching rows of the wrong ordering*, scrambling collision
neighbourhoods. Carrying the ID through the join makes ordering irrelevant —
`set_index(id_col).reindex(raw_ids)` is order-independent and was proven
0/0/0/0 against the real `Train.csv` in Round 3.

**Never** restore by position again.

### Phase 2 — Duplicate-date regression test
`tests/test_sorted_asof_join.py`:
- construct a left frame with **tied** `date_col` values and colliding
  `by_cols` keys (mirroring the real 56 duplicate `(deathdate, lat, lon)` keys)
- assert the buggy positional pattern reproduces the scramble (sanity check
  that the fixture is adversarial enough to catch it)
- assert `sorted_asof_join` returns byte-identical `id_col` and order
- assert a deliberately mismatched input raises, not silently returns
- assert downstream: labels zipped onto joined features still match

probabilities by ID:

```python
prob_df = prob_df.set_index(id_col).reindex(sample[id_col]).reset_index()
```

and L287 builds the output from `sample[id_col]`. So **final submission row
order is ID-correct.** The test-side scramble is a *model input* defect
(wrong feature per row), not a *row-order* defect. It cannot be fixed by
re-ordering — it requires regenerating features.

### Phase 3 — Refactor both culprit plugins
`plugins/macro_stress_extractor.py` and
`plugins/advanced_spatial_temporal_extractor.py`:
- replace the sort → `merge_asof` → positional-restore blocks with
  `sorted_asof_join(...)`
- **keep the ID column** in the returned frame (see §1.2 and decision §3.2)
- while in these files, remove the A5 violations they contain: hardcoded
  `"age"`, the `65`/`5` thresholds, and the `"avg_temperature"` /
  `"precipitation"` / `"wbgt_approx"` / `"spei_12m"` fallbacks. All must be
  read from `challenge_config.json`.

### Phase 4 — Stop the ID re-attachment in skill_07
At `skill_07_features.py` L2274–2286 and L1682–1696, **delete** the
`raw_tr`/`raw_te` positional ID fallback. If a feature file lacks `id_col`,
raise a hard error naming the responsible plugin — never silently re-attach
raw IDs to foreign predictions. This is the single change that makes the
defect class detectable at all.

### Phase 5 — ID-alignment guards on all OOF consumers
Every consumer of an OOF vector (`skill_11_gate`, `skill_12_metric`,
`skill_13_ensemble`, `oracle_fusion_core`, `skill_21_pseudo_label`) must
verify the vector length equals the *current* training-set length and that
the associated `cv_strategy_id` resolves against the active strategy
(§1.5(a) shows this check is currently bypassable). Add the length assertion
at the consumption boundary, not only at write time.

### Phase 6 — Single CV-split source
- Fix both refit scripts to read the real path
  (`SkillStateStore(paths.state_path).read()["cv_split_indices"]`, or
  `scores/cv_split_indices.json`), and **fail hard** if absent rather than
  falling back to `KFold`.
- Add a `BufferedSpatialCV` branch to `zindian/cv.py:make_cv_splitter` that
  raises `NotImplementedError` instead of silently degrading to `KFold`.
- Both `macro-stress-geofence-refit` and `residual-dlnm-specialist` must be
  re-run after the fix, and their OOF provenance re-verified.

### Phase 7 — Regenerate the 14 corrupted branches
Order matters — upstream first:
1. Re-run both plugins' feature extraction for all 14 branches.
2. Verify each with an **ID-based** restore against real `Train.csv`:
   expect **0/0/0/0** (coord match / label match / test / count).
3. Re-anchor: re-run the anchor OOF, then re-gate.
4. `features_train_ensemble.csv` is **load-bearing** — `oracle_fusion_core.py`
   L386–394 reads it for `y_true` (L714–728), which drives member verification
   scoring *and* collinearity pruning. It must be regenerated before
   `sub_008` or any fusion governance is re-evaluated. Round 3 found `sub_008`
   exposed through exactly this 3-layer chain (scrambled member vectors →
   verification/pruning scored against 168 shifted labels → blend locked into
   slot 2).
5. Re-run fusion governance, then re-submit only after Gate 4 re-approval.

**Clean and unaffected (verified 0/0):** `anchor-baseline`, `cohort-only`,
`temporal-anomaly-only`, `ai4eac-longmemory-cohorts`, `climate-longmemory-only`,
`shap_audit`, `sub_005`. Do not regenerate these.

### Phase 8 — Governance reconciliation
- The fusion exclusion list is **stale**: it was frozen against 8 submissions,
  but 8 newer live submissions now exist.
- `residual-dlnm-specialist` is currently excluded and must be
  un-excluded and re-gated after the Phase 6 provenance fix. Note: it declares
  `feature_extraction_plugin`, which the DAG honours only for ad-hoc refits
  (skill_07 L1890 resolves the plugin solely from
  `config.get("feature_extraction_plugin")`), so the declaration alone did not
  cause the corruption and does not protect against it.
- Harden `composite_metric`: it defaults to `f1_origin="oof"` and **0 of 12
  call sites pass provenance**, so the LB-contamination guard is effectively
  inert.

### Phase 9 — Verification gate
Do not declare this closed until:
- `pytest tests/ -q` passes (baseline + new tests from Phase 2)
- all 14 regenerated branches report 0/0/0/0 on the ID-based check
- `preflight_enforce.py` passes, including the A7 OOF-completeness check
- `skill_22` reproducibility audit agrees with the new fingerprints
- fusion governance lock is regenerated from current live submissions

---

## 3. Decisions I need confirmed before starting

1. **Regeneration blast radius.** Phase 7 re-runs features for 14 branches
   and re-anchors. That invalidates the OOF baseline, which invalidates gate
   margins, which may force re-review at human gates. Confirm you accept
   invalidating the current OOF baseline, or prefer a staged rollout
   (plugins + tests first, regeneration after you review).
2. **Plugin ID retention (Phase 3).** Keeping `id_col` in plugin output
   requires the corresponding change in skill_07 Phase 4. Do you want the ID
   carried all the way through, or dropped in the plugin with the assertion
   enforced before the drop?
3. **Two untouched findings** that I will *not* fix without a separate
   go-ahead: the `_lightgbm_shared.py` concat encoder leakage (§1.4) and the
   `make_cv_splitter` `BufferedSpatialCV` gap (§1.5(b), proposed as
   fail-loud in Phase 6). Both are real; both widen scope.

