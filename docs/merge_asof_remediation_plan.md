# `merge_asof` Row-Scrambling Remediation Plan

**Status:** IN PROGRESS — Phases 0, 1, 2 and the test-safety work complete; Phase 3 awaiting review.
**Date:** 2026-10-02 (rev 2 — after the first two review rounds; see §0.1)
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
| 1 | Stated mechanism (non-unique index) | **Upheld** | §1.1 — that claim *and* my first "correction" were both wrong |
| 2 | Are rows internally permuted, or features detached from labels? | **Upheld** | §1.2 — damage is *positional*, not internal |
| 3 | L1 unexplained by the root cause | **Upheld, twice** | §1.3 — still OPEN; now proven *not* a permutation |
| 4 | L2 is the same bug from another side | **Upheld** | §1.5 |
| 5 | L3 narrative overstated | **Upheld** | §1.6, rewritten |
| 6 | `cv_split_indices` pointer / loader | **Upheld as latent, not firing** | §1.7 |
| 7 | mtime-based cleanliness is a hypothesis | **Upheld** | §1.9, resolved by measurement in Phase 7 |
| 8 | "Not tie-driven" / mergesort works by luck | **Upheld** | §1.1 — both files are pre-sorted by date; it *is* tie-driven |
| 9 | Gate baseline on a different basis than candidates | **Upheld — verified 0.008555** | §1.3a, Phase 7 step 0 |
| 10 | Test suite mutates the live competition | **Upheld — confirmed and fixed** | committed `42097d5` |
| 11 | ID guard cannot work — records carry no IDs | **Upheld** | Phase 5 now leads with a schema change |
| 12 | Labels still come from a feature file in fusion | **Upheld** | new Phase 6a |
| 13 | Clean list doesn't match the audit | **Upheld** | resolved by ID-join measurement |
| 14 | Test-side claim too soft | **Upheld** | §1.4 — restated as ~28 rows (2.7%) |
| 15 | Encoder leak is category vocabulary, not labels | **Upheld** | §1.11 — low severity, deferred |
| 16 | "314 never-trained rows" is about fold quality | **Upheld** | §1.8 reworded |
| 17 | Commit 851428a carried a behaviour change | **Upheld** | already pushed, not rewritten; follow-up `acd260d` |

**Standing rule adopted:** every claim names the measurement that
establishes it. Where a mechanism is inferred rather than measured, the
document says so.

**Standing operational rule adopted:** the test suite must never write to a
live competition tree. Enforced at the write choke point and verified by
before/after md5.

---

## 1. Round 4 Findings — train/test data flow + CV provenance

### 1.1 The mechanism — measured, and both earlier drafts misattributed it

**Two corrections.** The first draft claimed `merge_asof` returns a
*non-unique index* and `.loc` resolves several rows per label. That is
contradictory (`.loc` on duplicate labels *changes length*, and the draft
also asserted length was preserved) and false in fact: `merge_asof` returns
a fresh `RangeIndex`, measured `unique? True`, `len == 3146`.

The second draft then over-corrected to "not tie-driven" and "mergesort works
by luck." **Both of those are also wrong.** The decisive measurement:

```
Train.csv deathdate : sorted=True, 1,609 rows in tied dates
Test.csv  deathdate : sorted=True,   229 rows in tied dates
```

Both raw files are **already sorted by `deathdate`**. That is the key fact
the second draft missed, and it makes the mechanism fully coherent:

1. The plugin sorts anyway (L69–71, default **unstable** quicksort).
2. `merge_asof` receives a key-sorted left frame, so it never raises — it
   returns a **fresh `0..n-1` index and discards the original labels**.
3. `.loc[train.index]` then requests labels `0..n-1` of a frame that is
   **already in sorted order** — it is a **no-op**. The output is permuted
   whenever the sort moved any row.

So it **is** tie-driven: quicksort reorders only rows with *equal*
`deathdate` (the sort key is the date alone, not the triple), which is why
damage appears as short isolated windows with a longest run of 6, and why
`mergesort` returns 0 — **not by luck, but because the file is pre-sorted
and a stable sort therefore leaves it untouched.**

```
kind=quicksort (plugin default) : 387 rows with wrong ID
kind=mergesort (stable)         :   0 rows with wrong ID
```

Corroborating: 1,609 train rows sit in tied dates, and Test.csv is also
pre-sorted with 229 tied rows, consistent with the ~28 test rows affected.

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

**The fix is unaffected** — restoring on `id_col` is correct regardless of
sort kind. What changes is only the explanation, and it matters for the
regression test: the fixture must use an **unsorted** left frame with **tied
dates**, because a sorted input reproduces nothing.

### 1.1a Independent confirmation — the pre-sorted-file prediction

The mechanism predicts that damage only occurs where the sort can move rows.
Both files being pre-sorted by date is consistent with everything measured;
had `Test.csv` been unsorted, the test-side damage would have been far larger
than the ~28 rows observed. Recorded as a checkable prediction, not an
assumption.

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

### 1.3 L1 — the anchor vector is a different model, not a permutation. OPEN.

**Upheld: L1 is NOT closed.** Two earlier claims here were wrong and are
retracted.

**Retracted claim 1 — "the Sep 3 10:34 event changed the values."** It did
not. The September audit already measured AUC 0.766 on Sep 20, so the
value change predates that. The Sep 3 burst (13:34:20–13:34:25, all
`scores/` files within 5 s) has the same signature as a `SkillStateStore`
write, not a model run: it records when `scores/` was last **re-serialised**,
not when values were produced.

**Retracted claim 2 — "L1 might be a row-order change."** Settled by one
measurement against the locked snapshot:

```
state vector vs data/raw/oof_anchor.csv (Sep 1, 18:46)
  len 3146 vs 3146
  SORTED EQUAL : False
  max sorted diff : 0.11930030388084334
```

A permutation preserves the multiset of values. It does not. **L1 is a
different model**, and the entire row-permutation explanation is ruled out
for this question.

**What is established, and what is not:**

- The ledger brackets the change. The last anchor row is
  `0.651694 / phase_4_inference_complete / 2026-09-02 21:33`; there is
  **no anchor training event after that**, yet the current vector differs.
- No ledger rows exist after Sep 4.
- The unexplained quantity is the **0.651694 anchor row**. Nothing in the
  audit produces ~0.65 for this anchor. The audit also found `0.6533` in
  `last_submission_comment` for sub_018 — two near-0.65 values on
  different branches point to **a scoring path run over misaligned pairs**,
  which is the best current lead and sits inside the bracket.
- The Oct 2 events are re-serialisation only: `branch_anchor-baseline_oof.json`
  md5 is unchanged across test runs, and the vector values are identical to
  those read hours earlier.

**L1 remains open, narrowed to:** which write placed a different vector into
the state key between Sep 1 18:46 and Sep 2 21:33, and whether the 0.6517
row and the 0.6533 submission comment share a misaligned scoring path.

Required before Phase 7: read the full columns of the `0.651694` and `0.6533`
records, and identify the write. The competition tree is **gitignored**
(`.gitignore:37`), so there is no version history to diff — the locked
snapshot at `~/snapshots/climate-2026-10-02` is the only pristine copy and
was taken *after* earlier test runs had already re-serialised state, so it
preserves current state, not September's.

### 1.3a The gate baseline is on the wrong basis — VERIFIED, 0.0086 bias

`851428a` moved candidate scoring to a **fixed 0.5 threshold**, but the
anchor baseline in state was produced by the **old threshold sweep**. The
two are not comparable, and every candidate inherits the bias. Measured on
the ID-aligned Sep-1 vector:

```
anchor @0.5            : F1=0.802937  AUC=0.813342  composite=0.8070989
stored anchor_oof_score: 0.8156538   (old, swept basis)
stored anchor_oof_f1   : 0.8171950   (swept F1, not the 0.5 F1)
BIAS against candidates: 0.008555
```

So a variant must beat **0.8157** to pass, while on the same 0.5 basis the
anchor is **0.8071**. Every candidate is penalised by ~0.0086. Saying
"numbers either side of the commit are not comparable" is not sufficient
while the baseline itself sits on the superseded basis.

**Recomputing the anchor baseline at 0.5 on the ID-aligned vector is a
Phase 7 prerequisite** (see Phase 7 step 0).

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

One output path is ID-safe in *row order*: `skill_14_inference.py` L317–318
reindexes test probabilities by ID against `SampleSubmission`, and L287 builds
output from `sample[id_col]`, so the submission's row order is correct.

**But that is a magnitude, not a clean bill of health.** The IDs on skill_07's
`test_probs` were attached **by position** from raw `Test.csv` onto
predictions derived from permuted features (L1693), so skill_14 faithfully
reindexes *wrong pairings*. Measured exposure: **~28 of 1,030 test rows
(≈2.7%)** carry another row's features. Small — but it is a wrong-feature
defect on real submitted rows, not a formatting issue, and it is fixed only by
regenerating features, never by re-ordering.

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

**The 5 km buffer is heavy — fold quality is uneven.** Rows excluded from
training per fold: `[641, 1672, 616, 1943, 1034]`; the union of all training
folds is 2,832 of 3,146, so 314 rows never appear in any training fold. The sum
of validation folds is exactly 3,146, so every row is validated exactly once
(no overlap, no duplicates) — this is expected buffered-CV behaviour, not a
bug, and it is **not** about OOF rows being unseen (every row is unseen by its
own fold). The real point: the buffer removes on the order of 10% of rows
from every fold's training set, and **fold 3 trains on only 644 of 3,146
rows**, so per-fold estimates are not equally well-supported. That matters
when reading the anchor OOF and any Nadeau-Bengio fold-variance correction,
which should carry the fold sizes.

### 1.9 mtime-based cleanliness is a hypothesis — measurement is the standard

Upheld. `climate-interactions` was called clean because it was untouched by
the buggy run. That is inference from timestamps, and the ID-join check it
lacks cannot detect it either: comparing only columns carried from raw will
show zero mismatches even when *macro-derived* columns are attached to the
wrong rows. The only sound standard is re-deriving features from raw inputs
and comparing by ID. **Resolved by measurement** — see Phase 7, where
`climate-interactions` and `seasonal-deathdate` are confirmed clean by ID
join (0/0) and `shap_audit` is found to have no feature file at all.

### 1.10 Blocking syntax error — RESOLVED

`zindian/skills/skill_07_features.py` L1317–1318 contained an uncommitted
`SyntaxError` inside a call expression, breaking preflight and 5
collection-dependent test files (0 tests passing). **Fixed in Phase 0.**
Verified: HEAD was clean; the error existed only in the working tree.
Baseline restored to **404 passed, 6 skipped**.

### 1.11 Encoder fit on train+test category vocabulary (separate, deferred)

`zindian/skills/_lightgbm_shared.py` fits the label encoder on
`pd.concat([train_vals, test_vals])`. This is a genuine, but **minor**,
train/test coupling: the encoder's *category vocabulary* is derived from both
frames. Note the severity precisely — the test set carries **no labels**, so
this is not label leakage; it is vocabulary derived from test inputs. Should
be fit on train only and applied via `.transform()` on test. **Deferred
pending explicit go-ahead**, as it is unrelated to this defect class and
changes model inputs.

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

**Why ID round-trip and not `.loc` restore:** `merge_asof` does not preserve
the left frame's row-to-label mapping — it returns a fresh `0..n-1` index
discarding the original labels. A positional `.loc[RangeIndex]` restore then
selects by *position* against that fresh index, which scrambles rows whenever
the preceding sort moved any row (measured: 387 rows, driven by tied dates).
Carrying the ID through the join makes ordering irrelevant —
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

### Phase 5 — ID-order guards (schema change first)

**A length check cannot detect this defect class.** A permutation preserves
length exactly, so a same-length misalignment passes every count-based
assertion. Worse, the artifacts themselves carry no IDs today:
`scores/branch_*_oof.json` is a bare list of 3,146 floats with no identifier
column, so a consumer has nothing to compare against.

1. **Schema change (`write_oof_record`, SoT S-1/S-6).** Persist an
   `id_order` list alongside `scores`, or at minimum an `id_order_hash`
   (sha256 of the comma-joined IDs) plus `id_count`. Without this the guard
   is decoration. Backfill for existing records on next write.
2. **Consumer guards** — `skill_11_gate`, `skill_12_metric`,
   `skill_13_ensemble`, `oracle_fusion_core`, `skill_21_pseudo_label`:
   recompute the expected hash from the *current* training IDs and compare.
   A hash mismatch is a hard failure naming the branch. A length check is
   kept only as a cheap pre-filter, never as the guard itself.
3. Keep the `cv_strategy_id` resolution check (§1.7) — currently bypassable.

**Test requirement:** build two frames of *identical length* whose `id_col`
orders differ, and assert the guard **rejects** them. A test that passes on
length alone is worthless.

### Phase 6 — Single CV-split source
- Fix both refit scripts to read the real path
  (`SkillStateStore(paths.state_path).read()["cv_split_indices"]`, or
  `scores/cv_split_indices.json`), and **fail hard** if absent rather than
  falling back to `KFold`.
- Add a `BufferedSpatialCV` branch to `zindian/cv.py:make_cv_splitter` that
  raises `NotImplementedError` instead of silently degrading to `KFold`.
- Both `macro-stress-geofence-refit` and `residual-dlnm-specialist` must be
  re-run after the fix, and their OOF provenance re-verified.

### Phase 6a — Fusion must read labels from raw, not a feature file

`oracle_fusion_core.py` currently sources `y_true` from
`features_train_ensemble.csv` (L386–394 → L714–728), which drives member
verification scoring **and** collinearity pruning. Even after regeneration
that dependency remains. Change it to read `y_true` from raw `Train.csv`
joined by `id_col`, and stop opening the feature file for labels entirely.

This is S-2's code half. Regenerating the file (Phase 7 step 4) does **not**
discharge it.

### Phase 7 — Regenerate the 14 corrupted branches

**Step 0 — prerequisite: recompute the anchor baseline on the 0.5 basis.**
The stored `anchor_oof_score = 0.8156538` was produced by the old threshold
sweep; candidates are now scored at fixed 0.5. Measured on the ID-aligned
vector, the anchor at 0.5 is **0.8070989** — a **0.008555** penalty applied to
every candidate. Re-derive the anchor on the ID-aligned vector at 0.5 and
write that as the gate baseline *before* any candidate is re-gated. Until
this is done, Phase 7's re-gate inherits the bias.

Order matters — upstream first:
1. Re-run both plugins' feature extraction for all 14 branches.
2. Verify each with an **ID-based** restore against real `Train.csv`:
   expect **0/0/0/0** (coord match / label match / test / count).
3. Re-anchor: re-run the anchor OOF on the 0.5 basis (step 0), then re-gate.
4. `features_train_ensemble.csv` is **load-bearing** — `oracle_fusion_core.py`
   L386–394 reads it for `y_true` (L714–728), which drives member verification
   scoring *and* collinearity pruning. It must be regenerated before
   `sub_008` or any fusion governance is re-evaluated. Round 3 found `sub_008`
   exposed through exactly this 3-layer chain (scrambled member vectors →
   verification/pruning scored against 168 shifted labels → blend locked into
   slot 2). Phase 6a then removes the label dependency entirely.
5. Re-run fusion governance, then re-submit only after Gate 4 re-approval.

**Clean, confirmed by ID-join against raw (not by mtime):**
`anchor-baseline`, `ai4eac-longmemory-cohorts`, `climate-interactions`,
`climate-longmemory-only`, `cohort-only`, `seasonal-deathdate`,
`temporal-anomaly-only` — all 0 label and 0 coordinate mismatches under an
ID join. `shap_audit` has **no feature file** and is not a regeneration
target at all. Do not regenerate these.

Re-derived in a scratch dir outside the competition tree (2026-10-02):

```
climate-interactions        n=3146  lab_mis=0    coord_mis=0    [ID]
seasonal-deathdate          n=3146  lab_mis=0    coord_mis=0    [ID]
shap_audit                  MISSING
macro-stress-geofence       n=3146  lab_mis=168  coord_mis=368  [pos]
catboost-climate-interactions n=3146 lab_mis=168 coord_mis=368  [pos]
```

This resolves the reviewer's point 6: the earlier draft had dropped
`climate-interactions` and `seasonal-deathdate` from the clean set on mtime
inference and added `shap_audit` unmeasured. Measurement puts both back in
the clean set, and `shap_audit` is not a feature-file branch.

### Phase 8 — Governance reconciliation
- The fusion exclusion list is **stale**: it was frozen against 8 submissions,
  but 8 newer live submissions now exist.
- `residual-dlnm-specialist` is **NOT** in `fusion_excluded_branches` — this
  corrects an earlier draft in this document, which wrongly claimed it was
  excluded. Verified: the list is
  `[ensemble, calibration_ensemble, calibration_anchor-baseline, pseudo_label_augmented]`.
  The exposure is the opposite of "excluded": the branch sits **inside** the
  fusion pool while correlating 0.7268 with the anchor and resting on
  scrambled features. It must be explicitly excluded, then re-gated after
  Phase 6, rather than "un-excluded" as previously written.
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

## 3. Decisions resolved and outstanding

**Resolved in this round:**

- **Rollout** — staged, as directed. Plugins, guards and tests land first and
  are reviewed before any regeneration. Phase 3 is next.
- **Plugin ID retention** — carry `id_col` all the way through plugin output.
  The training paths already exclude `id_col` from the feature set, so keeping
  it is safe, and dropping it early would make S-1 and the Phase 4 hard error
  unreachable.
- **Snapshot before Phase 7** — required, and done. `scores/` and
  `data/processed` are checksummed into a read-only copy
  (`~/snapshots/climate-2026-10-02`, 509 files, manifest verified with 0
  failures, `chmod -R a-w`), because L1 shows overwrites destroy evidence.
  The three untracked source files are copied to
  `~/snapshots/untracked-2026-10-02`.
- **Commit history** — `851428a` is on `origin/anchor-baseline`, so it is not
  rewritten; `acd260d` documents the behaviour change instead.

**Still outstanding:**

1. **L1 (§1.3).** The anchor vector is provably a *different model*, not a
   permutation (sorted values differ by up to 0.1193). Best lead: the
   unexplained `0.651694` anchor ledger row alongside the `0.6533` submission
   comment — two near-0.65 values suggesting a scoring path over misaligned
   pairs. Read the full columns of both records. **Does not block Phases
   3-6**; blocks Phase 7.
2. **Second snapshot copy off-box.** The untracked sources are now copied, but
   both snapshots live on the same disk. A copy to another disk or machine is
   outstanding.
3. **Re-verify audit findings sourced from test-written files.** The mutation
   set includes the phase summaries, diagnostics, session log, feature policy
   and the ledger, and earlier suite runs did the same. Any finding drawn from
   those — e.g. the `phase_4` summary metadata showing an F1 value under
   `anchor_oof_score` — must be re-verified against something tests do not
   write, or against the snapshot's oldest surviving copy.
4. **Phase 9 `skill_22` audit must run against the frozen snapshot**, never the
   live tree: `test_skill22_audit.py` was one of the four mutating tests, so
   any earlier skill_22 agreement result was produced against a directory the
   suite was rewriting.

---

## 4. Notes carried forward

**Staleness.** The audit is dated Sep 20; today is Oct 2. Board facts need
re-querying before any leaderboard-dependent decision, and the top two
submissions differ by ~0.001 - inside public-LB noise at n~1000. Tori's
blends also inherit unknown branch provenance.

**Commit discipline.** The tree still carries an untracked plugin, two
untracked refit scripts, and a modified `skill_08_anchor.py`. They are copied
into the snapshot, but the branch should be cleaned before Phase 7 so the
regeneration is reproducible from a known commit.

**Independent of this defect class:** the fusion exclusion gaps, the
bypassable `ScoreProvenance` defaults (0 of 12 call sites pass provenance),
and a stale governance lock.
