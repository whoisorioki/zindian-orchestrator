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
affected 14 branches. This plan covers the shared fix, the two culprit
plugins, full branch regeneration, and the structural guards that prevent
recurrence.

**Out of scope:** pipeline architecture changes, new dependencies, anything
requiring an SoT patch without one.

---

## 0.1 Review response — corrections to the previous draft

A reviewer challenged the draft on seven points. Six were upheld; the
findings below supersede the earlier text. Where the draft was wrong, it is
corrected here rather than quietly dropped.

| # | Reviewer challenge | Verdict | Correction |
|---|---|---|---|
| 1 | Stated mechanism (non-unique index) does not hold together | **Upheld — draft was wrong** | §1.1 gives the measured mechanism |
| 2 | Are rows internally permuted, or are features detached from labels? | **Upheld — materially narrows the damage** | §1.2 — damage is *positional*, not internal |
| 3 | L1 (anchor OOF value set differs) unexplained by the root cause | **Upheld — still open** | §1.3, now an explicit blocker |
| 4 | L2 is the same bug seen from another side | **Upheld** | §1.4 |
| 5 | L3 narrative overstated | **Upheld** | §1.5, rewritten |
| 6 | `cv_split_indices` pointer / loader question | **Upheld as a real latent bug, but not currently firing** | §1.6 |
| 7 | mtime-based cleanliness is a hypothesis | **Upheld** | §1.7 |

The reviewer's central methodological point is adopted as a standing rule:
**every claim must name the measurement that establishes it.** Where a
mechanism is inferred rather than measured, this document says so.

---

## 1. Round 4 Findings — train/test data flow + CV provenance

### 1.1 The mechanism — measured, and the draft's stated cause was wrong

**Correction to the previous draft.** That draft claimed `merge_asof`
returns a *non-unique index* and that `.loc` then resolves several rows per
label. This is wrong on two counts, and the reviewer was right to reject it:

- `.loc` with duplicate labels *changes the length* of the result. The draft
  simultaneously asserted the frame length was preserved. Both cannot hold.
- `pd.merge_asof` returns a fresh `RangeIndex`, which is unique by
  construction. Measured directly on real data:
  `merged index unique? True`, `len(merged) == 3146`.

**The actual mechanism, established by experiment.** Reproduced end to end
against the real `Train.csv` (3146 rows) and
`macro_stress_proxies.parquet` (305,360 rows):

```
kind=quicksort   (plugin's default, plugins/macro_stress_extractor.py:69)
   merged index equals sorted index (order)?  False
   plugin restore (.loc[original.index]) -> rows with wrong ID: 387

kind=mergesort   (stable)
   merged index equals sorted index (order)?  True
   plugin restore -> rows with wrong ID: 0
```

So the trigger is that **`pd.merge_asof` does not preserve the left frame's
index when the left frame is not already in join-key order.** With the
default unstable `sort_values`, the sorted permutation is applied to the
data but the returned index does not correspond back to it, so the
subsequent positional `.loc[original.index]` picks the wrong rows.

Two consequences the draft got backwards:

- It is **not** tie-driven. A stable sort (`mergesort`) fixes this data
  precisely because it preserves the *pre-existing* row order — ties were
  never the issue. The draft's "56 duplicate keys" framing was a
  coincidence of this dataset, not the cause.
- A stable sort is **not** a fix. It happened to work here; it is not a
  guarantee. `merge_asof` makes no ordering promise, so relying on it is
  exactly the fragility that caused this. The fix must not depend on sort
  kind at all.

`plugins/macro_stress_extractor.py` L69–71 and L94–95, verbatim:

```python
train_sorted = train.sort_values(by=date_col).copy()   # L69  unstable
test_sorted  = test.sort_values(by=date_col).copy()    # L70  unstable
ext_sorted   = ext_df.sort_values(by=date_col).copy()  # L71  unstable
...
train_merged = train_merged.loc[train.index].copy()    # L94  positional
test_merged  = test_merged.loc[test.index].copy()      # L95  positional
```

Both plugins carry this pattern; both scramble train *and* test.

### 1.2 The damage is positional, not internal — this narrows the blast radius

**This is the reviewer's most consequential correction, and it is
established by direct measurement**, not inference. Joining the scrambled
feature file back to raw on ID:

```
features_train_macro-environmental-specialist.csv
  rows: 3146
  rows where is_climate_sensitive != raw: 0
  rows where latitude            != raw: 0
```

**Whole rows were permuted. Within any single affected file, features and
labels remain correctly paired with each other.**

So the draft's blanket "every number is untrustworthy" was **too strong for
per-branch OOF and correct for submissions.** What actually breaks:

- **Per-branch OOF within one branch: largely intact.** The model learned
  from correctly-paired (features, label) rows. Its internal fit is sound.
- **Positional everything else: broken.** The corruption only manifests when
  something *outside* the file is joined by position — CV split indices,
  pairing with the anchor OOF vector, and `test_probs` receiving IDs from
  raw `Test.csv` by position at `skill_07_features.py` L1693.

This is why single-branch scores looked healthy while cross-branch
comparisons and leaderboard results did not. It also means Phase 7
regeneration is mostly about **restoring cross-artifact alignment**, not
about retraining from a poisoned matrix — which is good news for
reproducibility and bad news for anyone who assumed the models themselves
need rethinking.

Corollary the reviewer raised and this plan adopts: for the 387 permuted
rows, the **spatial buffer guarantee was also voided**, because a row's
coordinates travelled away from the location it was assigned to in the CV
split. Buffer integrity and row integrity failed together.

### 1.3 L1 — the anchor OOF was overwritten. Still unexplained. BLOCKER.

**Upheld in full.** This is not explained by the scramble and must be
resolved before any regeneration, or it will simply recur.

Measured facts:

- `scores/branch_anchor-baseline_oof.json` mtime is **2026-10-02 11:46:21**,
  i.e. rewritten today. Nineteen sibling OOF files share that same
  timestamp, so a single bulk run rewrote the whole `scores/` directory.
- `branch_collapsed-catboost_oof.json` retains **2026-09-03** — it was *not*
  touched by that run. The rewrite was selective.
- The current anchor vector has a different value set from the one in the
  September audit, correlates only ~0.80 with it, and scores AUC 0.766
  against its own feature file.

**A row scramble cannot change prediction *values*.** Something regenerated
a different model, or applied a different fold assignment, at 11:46. Until
that is identified, the anchor baseline is not a stable reference and any
regeneration run would be measured against a moving target.

Required before Phase 7: identify what wrote `scores/` at 11:46, determine
whether an operator-approved run or an unintended side effect produced the
current anchor record, and record the finding in this document.

### 1.4 Why ID assertions alone cannot catch this — and where output is safe

Because the corruption is a *whole-row permutation*, every file-level ID
check passes: the set of IDs is complete and each row is internally
consistent. The defect is only visible when an artifact is compared against
**another artifact by position**.

That is what `skill_07_features.py` L2274–2286 (and L1682–1696) does when a
feature file lacks the ID column:

```python
raw_tr = pd.read_csv(... Train.csv) if id_col not in train_feat else None
train_id_series = train_feat[id_col] if id_col in train_feat else raw_tr[id_col]
oof_df = pd.DataFrame({id_col: train_id_series, "oof_prob": np.asarray(result["oof_probs"])})
```

The plugins drop the ID column, so this falls back to raw IDs zipped
**positionally** onto predictions derived from a permuted matrix. This is
how the corruption becomes invisible, not how it is caused.

One output path is genuinely safe: `skill_14_inference.py` L317–318 reindexes
test probabilities by ID against `SampleSubmission`, and L287 builds output
from `sample[id_col]`. So **final submission row order is correct**; the
test-side defect is a wrong-feature-per-row defect, not a row-order defect.
It cannot be fixed by re-ordering — it requires regenerating features.

### 1.5 L2 is the same bug, not a second corruption

**Reviewer upheld.** 168 label shifts out of 3146 is the same 5.34%
mismatch seen from the test side. L2 is fully explained by §1.1.

The length-only asserts remain a real, separate defect — they cannot detect
a permutation at all — but L2 is not evidence of additional corruption.

### 1.6 L3 — false provenance, but the leakage narrative was overstated

**Reviewer upheld; the draft's framing was wrong.**

The solid finding: `scripts/refit_macro_stress_geofence.py:79` and
`scripts/refit_residual_dlnm_specialist.py:111` both read
`DATA_DIR / "cv_split_indices.json"` = `data/processed/cv_split_indices.json`,
which **does not exist** (verified by directory listing). Both therefore fell
back — to shuffled `KFold(5)` and `make_cv_splitter({"type": "kfold"})` — while
stamping records with `cv_strategy_id="config:BufferedSpatialCV"`.

That is **false provenance**, and it is real: those two branches' OOF was
computed under random folds but labelled spatial, so it is not comparable to
the anchor and cannot be pooled with spatial branches.

The overstated part: the draft implied `sub_018` gained something illegitimate
from buffered CV. That does not follow. Only 2 of 55 locations are isolated
from training data, so buffered spatial CV is **pessimistic** relative to
this test set for *every* model, anchor included. A wide OOF/LB gap is
expected and is not by itself evidence of leakage.

Also: the OOF 0.7872 figure for `sub_018` is a **reconstruction**, produced
after the anchor OOF it consumed had been overwritten (§1.3). It is not
usable as evidence in either direction and should not be cited again.

`zindian/skills/_lightgbm_shared.py` fits the label encoder on
`pd.concat([train_vals, test_vals])`. This is a genuine (separate)
train/test leakage finding: the encoder sees test label values. Should be
fit on train only and applied via `.transform()` on test.

### 1.7 The `cv_split_indices` pointer — real latent bug, currently NOT firing

**Reviewer's biggest open question, now resolved by measurement.**

The raw state file holds a dict pointer, not a list:

```
cv_split_indices -> {'cv_splits_file': 'scores/cv_split_indices.json', 'count': 5}
```

And `zindian/cv.py:114` requires a list:

```python
raw_splits = state.get("cv_split_indices")
if not isinstance(raw_splits, list) or not raw_splits:
    return None          # <-- silent None on a dict pointer
```

So `load_explicit_cv_splits` **returns `None` if handed raw JSON state.**
That is a genuine footgun.

In the DAG path it does not currently fire. Verified end to end:

```
via SkillStateStore.read(): list, len 5
load_explicit_cv_splits -> 5 splits
  fold0: train=1510 val=995      fold1: train=960  val=514
  fold2: train=1967 val=563      fold3: train=644  val=559
  fold4: train=1597 val=515
```

`SkillStateStore.read()` (L132–144) rehydrates the pointer before the loader
sees it. Fold sizes are uneven, which proves genuine `BufferedSpatialCV`
splits rather than `KFold` (uniform 1604/401). **The anchor OOF is spatially
honest.** The refit scripts bypass rehydration because they read JSON
directly.

The loader should still fail loudly rather than return `None` on a dict — a
silent `None` is indistinguishable from "no splits persisted," which is
precisely how the refit scripts ended up on random folds.

### 1.8 Two additional findings surfaced by this round

**`BufferedSpatialCV` is a routing label, not a class.** No such class
exists anywhere in the repo — only the string. `skill_05_cv.py:121` implements
it as `build_spatial_splits()`: KMeans geographic blocks + `GroupKFold` +
`_apply_spatial_buffer()`. This matters because `zindian/cv.py:make_cv_splitter`
has no branch for it and **silently degrades to shuffled `KFold`** for any
caller reaching it with that config. The DAG avoids this only because
explicit splits load first.

**The buffer removes far more training data than the draft assumed.** Rows
excluded from training per fold: `[641, 1672, 616, 1943, 1034]`. The union of
all training folds is 2832 of 3146, so **314 rows are never in any training
fold** — their OOF prediction comes from models that never saw them. The sum
of validation folds is exactly 3146, so every row is validated exactly once
(no overlap, no duplicates). This is expected behaviour for buffered CV, not
a bug, but OOF is pessimistic by construction and the 314 never-trained rows
deserve explicit reporting rather than silent inclusion.

### 1.9 mtime-based cleanliness is a hypothesis — measurement is the standard

Upheld. `climate-interactions` was called clean because it was untouched by
the buggy run. That is inference from timestamps, and the ID-join check it
lacks cannot detect it either: comparing only columns carried from raw will
show zero mismatches even when *macro-derived* columns are attached to the
wrong rows. The only sound standard is re-deriving features from raw inputs
and comparing by ID. Folded into Phase 7.

**(b) Latent trap in `zindian/cv.py` L60–64** — `make_cv_splitter` has
branches only for `stratified` / `group` / default. There is **no
`BufferedSpatialCV` branch**, so any caller reaching it with
`type="BufferedSpatialCV"` silently receives shuffled `KFold`. Currently
unreachable in the DAG (explicit splits load first, skill_08 L207), but it
is a live footgun and directly contradicts the config.

### 1.10 Blocking syntax error — RESOLVED

`zindian/skills/skill_07_features.py` L1317–1318 contained an uncommitted
`SyntaxError` inside a call expression, breaking preflight and 5
collection-dependent test files (0 tests passing). **Fixed in Phase 0.**
Verified: HEAD was clean; the error existed only in the working tree.
Baseline restored to **404 passed, 6 skipped**.

### 1.11 Test-set leakage in the shared trainer (separate, deferred)

`zindian/skills/_lightgbm_shared.py` fits the label encoder on
`pd.concat([train_vals, test_vals])`. This is a genuine train/test leakage
finding — the encoder sees test label values. Should be fit on train only
and applied via `.transform()` on test. **Deferred pending explicit
go-ahead**, as it is unrelated to this defect class and changes model
inputs.

---

## 2. Implementation plan

### Phase 0 — Unblock the test suite — **DONE**
1. ~~Fix the `SyntaxError` at `skill_07_features.py` L1317–1318.~~ Done.
2. ~~Record the true baseline.~~ Recorded: **404 passed, 6 skipped**.

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

### Phase 2 — Regression test — **DONE**
`tests/test_sorted_asof_join.py` — **12 tests, all passing.**

The fixture uses an adversarial **unsorted** left frame (the reviewer's
correction — ties alone do not reproduce the bug; the trigger is that the
left frame is not already in join-key order). It carries a guard test that
**proves the old pattern still fails**, so the suite cannot pass against the
real defect:

```
old positional pattern, quicksort -> 6 scrambled rows
sorted_asof_join                 -> 0 scrambled rows
```

Cases covered: order-independent restore on `id_col`; exact round-trip;
unmatched-row guard raises; identity hard-fail; test-frame parity with
train.

> Note: the reviewer's warning that a test encoding the *described*
> mechanism could pass against the real bug is why the quicksort-guard test
> is mandatory rather than incidental. It is the regression anchor.

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

## 2A. Required Source-of-Truth amendments

Per the working rule — *a gap must be patched in the SoT before it is
resolved in code* — this defect class is **not** currently covered by any SoT
clause. The SoT mandates the OOF *schema* but says nothing about **row
identity across artifacts**. These amendments target **v2.10**.

| # | Amendment | SoT location | Why needed |
|---|---|---|---|
| S-1 | **Row-Identity Contract**: every artifact carrying a per-row model input/output must carry `id_col`, and joins between artifacts must be by ID, never by position | new §2 **Principle 7** | The OOF Contract (Principle 3) covers schema, not identity. This is the contract whose absence allowed the defect |
| S-2 | **Label provenance rule**: labels must be sourced from raw `Train.csv` joined by `id_col`. A feature file may never supply the target | §4 Phase 2B (skill_07) + Principle 7 | Reviewer's missing-point #1. `ensemble` showed why: a label read from a permuted feature file is a silent target leak |
| S-3 | **Silent-fallback ban**: any `cv_split_indices` consumer must raise on a malformed/pointer value rather than return `None` and proceed on a different splitter | §2 Principle 3 | §1.7 — a silent `None` is indistinguishable from "no splits persisted," which is how the refit scripts got random folds under a spatial tag |
| S-4 | **`BufferedSpatialCV` fail-loud**: `make_cv_splitter` must raise `NotImplementedError` on an unhandled `cv_strategy.type`, never degrade to shuffled `KFold` | §2 Principle 3 | §1.8 — the degrade is currently silent and contradicts config |
| S-5 | **Provenance honesty**: a script that cannot load the configured splits may not write a record tagged with that `cv_strategy_id` | §6 Reproducibility Contract | §1.6 — false provenance on 2 branches |
| S-6 | **Cross-artifact alignment check** as a preflight ENFORCE check (A9): for each branch, assert feature rows and OOF vector are aligned by `id_col`, not only by length | §3 Preflight ENFORCE | Existing A7 checks *count*. A permutation passes a count check. This is the check that would have caught it |
| S-7 | **§7 Known Gaps entry** for the anchor-OOF overwrite (§1.3) while unexplained | §7 | Track it; do not let it vanish |

### 2A.1 Where the SoT is currently wrong or silent

**Silent** — no clause requires ID-based alignment (§1.2 proves a length-only
check passes). S-1, S-2, S-6 close this.

**Silent** — nothing forbids a feature file from supplying a label. S-2 closes.

**Silent** — §2 Principle 3 specifies *which* splits, never that a failure to
load them must be loud. S-3, S-5 close.

**Arguably contradicted** — §4 documents `BufferedSpatialCV` as a first-class
`cv_strategy.type`, but no such class exists and the shared factory degrades
it silently. S-4 makes the mismatch explicit.

**Incomplete** — §7 does not record the §1.3 overwrite. S-7 tracks it.

### 2A.2 Sequencing

S-1 and S-2 are **blocking** for Phases 4–5: they are the contracts that
justify removing the positional ID fallback and adding consumer guards. Per
the working rule, code must not be changed ahead of its SoT clause, so the
SoT patch lands **immediately before** Phase 4, not at the end.

S-3, S-4, S-5 land with Phase 6. S-6 lands with the Phase 9 verification gate.

## 2B. Test plan

**Baseline: 404 passed, 6 skipped.** No phase may reduce that.

| Phase | Test file | Asserts | Gate |
|---|---|---|---|
| 2 *(done)* | `test_sorted_asof_join.py` | `sorted_asof_join` restores by `id_col`; **old quicksort pattern still scrambles** (6 rows) — regression anchor | 12 pass |
| 3 | `test_macro_stress_extractor.py` *(new)* | Plugin output row order == input row order; `id_col` retained; no positional `.loc[original.index]` remains (**source scan**, so a reintroduced literal is caught without executing) | pre-commit |
| 3 | `test_a5_compliance.py` *(extend)* | No hardcoded `"age"`, `65`/`5`, or macro column fallbacks remain in either plugin | pre-commit |
| 4 | `test_skill07_id_contract.py` *(new)* | A feature file lacking `id_col` **raises**, naming the responsible plugin — instead of silently re-attaching raw IDs | pre-commit |
| 5 | `test_oof_id_alignment.py` *(new)* | skill_11/12/13, `oracle_fusion_core`, skill_21 reject an OOF vector whose length matches but whose `id_col` order differs — **the permutation case a length check misses** | pre-commit |
| 6 | `test_cv_split_loader_contract.py` *(new)* | `load_explicit_cv_splits` raises on a dict pointer instead of returning `None`; `make_cv_splitter` raises `NotImplementedError` on `BufferedSpatialCV` | pre-commit |
| 6 | `test_refit_script_provenance.py` *(new)* | Both refit scripts load splits from the real `scores/` path, and **refuse to write** an OOF record when splits are unavailable | pre-commit |
| 7 | `test_feature_regeneration_alignment.py` *(new)* | For each of the 14 branches: raw-carried columns and the label match raw **by ID**, and macro-derived columns match a fresh re-derivation **by ID**. Replaces the mtime heuristic (§1.9) | post-regeneration |
| 9 | preflight A9 | Cross-artifact ID alignment per branch | closure |

### 2B.1 Test design notes

**The permutation test is the important one.** These defects are invisible
to length assertions. `test_oof_id_alignment.py` must construct two frames of
*identical length* whose `id_col` orders differ, and assert the guard rejects
them. If it passes on a length check, it is worthless.

**Regression anchoring.** Per the reviewer's point 1, every new guard test
should first be confirmed to **fail against the pre-fix code**. Where that is
impractical (plugin execution is slow), the source-scan test covers the
literal's absence and the unit test covers the behavioural guard.

**No new dependencies.** All tests use `pandas` / `numpy` / `pytest`, already
in `requirements.txt`.

### 2B.2 Full-suite cadence

Run `pytest tests/ -q` after **every** phase and report the delta against
404/6. A drop is a blocker for the following phase, not a footnote.

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
   go-ahead: the `_lightgbm_shared.py` concat encoder leakage (§1.11) and the
   `make_cv_splitter` `BufferedSpatialCV` gap (§1.8, now covered by S-4 in the
   SoT patch). Both are real; both widen scope.
4. **NEW — L1 investigation (§1.3).** Resolving what rewrote `scores/` at
   11:46 today is on the critical path for Phase 7; without it the anchor
   reference is unstable. Trace it now (git reflog, shell history, ledger
   timestamps), or proceed with Phases 3–6 and treat L1 as a hard blocker at
   Phase 7?
5. **NEW — final-pick decision.** The leaderboard facts in the September audit
   are 12 days stale, and the top two submissions differ by ~0.001, which is
   inside public-LB noise at n≈1000. Any final blend decision should be
   deferred until after regeneration rather than made on stale numbers. Tori's
   blends also inherit whatever branches they were built from, and that
   provenance is unknown.

---

## 4. Notes carried forward from the reviewer

**Staleness.** The audit is dated Sep 20; today is Oct 2. Board facts need
re-querying before any leaderboard-dependent decision.

**Commit discipline (reviewer's point 5).** Phase 0's `SyntaxError` fix and
the two untracked refit scripts should be committed on a clean branch before
any regeneration run, so it is reproducible from a known commit rather than a
dirty tree.

**Independent of this defect class**, to be tracked separately: the fusion
exclusion gaps, the bypassable `ScoreProvenance` defaults (0 of 12 call sites
pass provenance), and a stale governance lock.

