# Intake & Intel Truthfulness — Findings and RFC

**Status:** RECOMMENDATION / FINDINGS (no code changed)
**Date:** 2026-09-01
**Version aligned with:** docs/source_of_truth.md v2.8
**Subject competition:** `climate-risk-health-prediction-challenge`
**Method:** Live verification of `https://zindi.africa/competitions/climate-risk-health-prediction-challenge`
(main + `/data` pages) via plain HTTP and a Playwright headless scrape, the public
`https://api.zindi.africa/v1/competitions/{slug}` endpoint, plus direct inspection of the
skill modules, `challenge_config.json`, `SKILL_STATE.json`, `reports/zindi_monitor.json`, and
the competition's raw data files. Code behaviour is ground truth for "what currently happens".

---

## 0. Implementation Status (2026-09-01)

Fixes A/B/D/E/F from §8 are **implemented** in this session (operator-approved via the answers to §9). Status per item:

| Item | Status | Verification |
|---|---|---|---|
| A — target resolution via data dictionary + `submission_target_columns` | ✅ DONE (`skill_02_intake.py`) | unit tests `tests/test_intake_target_resolution.py`; live `zindian phase 1` now completes with `target_col="is_climate_sensitive"` + `submission_target_columns=["TargetF1","TargetRAUC"]` |
| B — external-data truthfulness | ✅ DONE (`zindi_monitor_core.py`) | live competition page carries a **genuine ban-vs-permission conflict** (generic full-rules "may use only the datasets provided" vs Data tab "encouraged to use publicly available climate datasets"— exactly the discussion flag "Rules tab vs Data tab conflict" at `resolved_by_organizer: false`). Monitor now prints `External data: UNKNOWN` + a loud operator warning instead of silently BANNED. |
| C — dual-output contract (binary + probability) | ⏳ DEFERRED — needs an SoT contract decision; `submission_target_columns` is recorded as the first step | — |
| D — wire the config temporal-lock guard | ✅ DONE (`zindi_monitor_core._write_config_intel`) | post-Phase-1 monitor runs now write `community_signals` only (with a frozen-config note) |
| E — `allow_teams` / `max_number_of_participants_per_team` mapping | ✅ DONE (`skill_02_intake.py`) | live config now `team_allowed: true` |
| F — phase-completion hygiene + strict Phase-2A gate | ✅ DONE (`orchestrator.py`) | `phase_*_complete` now written only on zero-error phases; failed/unauthorised runs cannot leave a passable flag (the Phase-2A gate now keys on the orchestrator-set flag, not the skill_01 dag_phase string) |

Also fixed: `_resolve_external_banned` may no longer silently default to banned — ambiguous/conflict page text now returns `None`, with the conflict surfaced via an `external_banned_conflict` marker and loud warnings in monitor output / compliance log. Challenged-config `allowed_external_data` remains a schema-required bool (conservative default until the operator resolves the conflict).

Phase 1 of the subject competition crashed in `skill_04` and `skill_05` because
`challenge_config.json["target_col"]` was set to `"TargetF1"` — a **submission column** —
while the actual training target in `Train.csv` is `is_climate_sensitive`. The intake skill
infers `target_col` from `SampleSubmission.csv` without ever cross-checking the training
columns or reading `data_dictionary.csv` (no code in `zindian/` or `scripts/` reads the data
dictionary at all).

A live scrape also shows two **rule fields written to config that contradict the actual
competition page**:

1. `allowed_external_data: false` — the Data tab explicitly says external climate data is
   **encouraged/permitted** (ERA5, CHIRPS, NDVI, SRTM…), subject to public-accessibility and
   transparency rules.
2. `use_probabilities: false` — the evaluation requires a **probability column** (`TargetRAUC`)
   *and* a binary column (`TargetF1`) thresholded at 0.5 (no custom threshold allowed).

Additional framework defects were found while investigating (dead temporal-lock guard in the
monitor's config writer, stale phase-completion flags surviving a failed Phase 1, and a phase
gate that can pass on stale flags).

---

## 2. Reproduction / symptom summary

Commands run (in this order, terminal transcript):

```
zindian preflight --competition climate-risk-health-prediction-challenge
zindian phase 1 --competition climate-risk-health-prediction-challenge
zindian monitor --competition climate-risk-health-prediction-challenge
```

Observed:

```
skill_04: ERROR — Skill skill_04 failed: Target column 'TargetF1' not present in training data
skill_05: ERROR — Skill skill_05 failed: "Target column 'TargetF1' not found in raw train or features."
```

The monitor printed `Source : playwright+config_override` — which the operator correctly
identified as a suspicious attribution (playwright "should work alone").

The preflight prompt surfaced twice because `zindian phase` runs `scripts/preflight_enforce.py`
as a gate before the phase (cli.py L386–407) *in addition to* the explicitly invoked
`zindian preflight`. That is expected behaviour.

---

## 3. Ground-truth facts (verified 2026-09-01)

### 3.1 Data files

- `Train.csv` columns (13): `ID, zone, gender, deathdate, age, avg_temperature,
  max_temperature, min_temperature, precipitation, latitude, longitude, location,
  is_climate_sensitive`. 3,146 rows.
- `is_climate_sensitive` distribution: 2047 × 1 / 1099 × 0, no nulls. **This is the training
  target** (named as such in `data_dictionary.csv`).
- `SampleSubmission.csv` columns: `ID, TargetF1, TargetRAUC` (1030 rows of zeros in the sample).
  `TargetF1` = binary prediction, `TargetRAUC` = predicted probability.
- `climate_features.csv` exists (pre-downloaded SRTM/ERA5/CHIRPS/NDVI features for `ID`s) but is
  **never loaded by any pipeline skill**.

### 3.2 Competition page (playwright scrape, main page)

```
Your model must output:
  a binary target indicating whether the death is climate-sensitive or not
  a probability indicating the likelihood that the death is climate-sensitive
Setting a probability threshold is strictly forbidden. Your binary target should be based on
the default threshold of 0.5.
The submission file should follow this format:  ID TargetF1 TargetRAUC
... multi-metric evaluation ... two error metrics:
  F1-Score (60%): ... precision and recall ... imbalance datasets
  ROC-AUC (40%): ... ranks climate-sensitive cases above non-climate-sensitive ...
Submission Limits: 10 submissions per day, 300 submissions overall.
Team size: Max team size of 4
Public-Private Split: ... approximately 30% of the test dataset ... private ... 70%
Code Review: Top 10 on the private leaderboard ... 48 hours
```

### 3.3 Competition page (`/data` tab)

```
Participants are encouraged to use publicly available climate datasets to enrich the training
data. Permitted sources include temperature reanalysis products such as ERA5, rainfall
datasets such as CHIRPS, and satellite-derived environmental indicators such as NDVI or land
surface temperature.
All external datasets used must be publicly accessible. Participants are encouraged to share
links to any external data sources on the competition forum ...
Some climate features have been downloaded and added to get you started ...
A dictionary of these features is given in downloaded_climate_features_data_dictionary.csv
```

### 3.4 Public API (`GET /v1/competitions/{slug}`)

- `error_metric: "multi"` (API exposes no `metric`/`daily_limit`/`total_limit`/
  `allowed_external_data`/`automl_permitted` keys for this competition)
- `allow_teams: True`, `max_number_of_participants_per_team: 4`
- `has_benchmark: True`, `benchmark_submission_public_score: 0.763182072`
- `data_type: ["Structured", "Satellite"]`, `rubric_type: "multi"`
- `end_time: 2026-10-18T21:59:00.000Z`

### 3.5 Operator-provided confirmation (pasted from live pages, 2026-09-01)

The operator independently pasted the main-page and Data-tab text into this investigation.
It matches the Playwright scrape in §3.2/§3.3 verbatim, and adds the explicit **About / Data
tab** formulation that is the strongest evidence for the external-data finding:

```
About: Participants are encouraged to use publicly available climate datasets to enrich
the training data. Permitted sources include temperature reanalysis products such as ERA5,
rainfall datasets such as CHIRPS, and satellite-derived environmental indicators such as
NDVI or land surface temperature.
All external datasets used must be publicly accessible. Participants are encouraged to
share links to any external data sources on the competition forum to support transparency
and reproducibility.
Some climate features have been downloaded and added to get you started. However you are
welcome to download yours. A dictionary of these features is given in
downloaded_climate_features_data_dictionary.csv
```

Two independent captures of the live pages therefore agree on every claim in this document.

---

## 4. Config truthfulness audit (config vs ground truth)

| # | `challenge_config.json` value | Ground truth | Verdict | Root-cause chain |
|---|---|---|---|---|
| 1 | `target_col: "TargetF1"` | Train target is `is_climate_sensitive` | **WRONG** | skill_02 L780–814 infers first non-ID SampleSubmission column; never validated against train; `data_dictionary.csv` never read |
| 2 | `allowed_external_data: false` | Data tab encourages/permitted external climate data | **WRONG** | template default `false` (L26) inherited via merge (skill_02 L740–746); skill_02 forces `False` when API silent (L758–762); monitor `_resolve_external_banned` defaults banned (zindi_monitor_core L97–113) and its allow-phrases don't match "encouraged to use publicly available" |
| 3 | `use_probabilities: false` | Dual output required: binary @0.5 **and** probability | **MISLEADING** | single bool cannot express dual-output; skill_02 metric-based derivation L144–151 maps unknown `multi` → no; propagated by monitor L414–415 |
| 4 | `team_allowed: false` | API `allow_teams: True`, page "team of up to four" | **WRONG** | skill_02 L160 reads `team_allowed` while the API key is `allow_teams`; template default `false` survives (L22) |
| 5 | `metric: "multi"`, direction `maximize` | 0.6×F1 + 0.4×ROC-AUC, both maximize | OK (lossy) | literal from API `error_metric`; semantics not modeled |
| 6 | `daily_limit: 10`, `total_limit: 300` | Page "10 per day, 300 overall" | OK | monitor scrape → `_write_config_intel` |
| 7 | `public_split_pct: 30`, `private_split_pct: 70` | Page "approximately 30% / 70%" | OK | monitor |
| 8 | `submission_budget {300,10,0}` / `remaining_submissions` | matches live count (10 today) | OK | monitor leaderboard read |

**Audit conclusion:** the two highest-impact falsehoods are items 1 (breaks Phase 1) and 2
(breaks a compliance-dependent decision). Items 3 and 4 are contract-shape problems that
would degrade later phases if Phase 1 were fixed naively.

---

## 5. Why `Source: playwright+config_override` appears (mechanism)

`zindian/zindi_monitor_core.py`:

- `fetch_competition_intel()` (L356–468): playwright scrape of main + `/data` pages, then —
  when a `config` object is passed (always; `run()` L1324) — **overrides** nearly every rule
  field with whatever `challenge_config.json` contains (L412–434), then labels the result
  `playwright+config_override` (L434).
- Intended semantics: config (from skill_02/API) is "competition-specific" and the scraped
  boilerplate is "generic". In practice the precedence is **inverted** for fields the API
  returns as null: the template defaults (`false`/banned) then **mask** the directly-scraped
  page truth, which is exactly what happened with `external_banned`.

The label is an honest audit trail; the design intent behind the override is the problem.

---

## 6. Additional framework defects found during investigation

1. **Dead config temporal-lock guard** — `_write_config_intel` computes
   `_ = dag_phase in ("uninitialized", "phase_1_integrity_locked")` (L1210–1212) and discards
   the result; it then writes `daily_limit`/`total_limit`/etc. to `challenge_config.json` on
   every monitor invocation regardless of DAG phase, contradicting the documented
   "only `community_signals` post-Phase-1" rule.
2. **Stale phase-completion flags** — after the failed Phase 1 run, `SKILL_STATE.json` still
   carries `phase_1_complete: true` and `phase_2a_complete: true` (echoes of an earlier,
   partial session in `reports/summaries/2a_*`), while `dag_phase` correctly reads
   `phase_1_incomplete`. The orchestrator gate (orchestrator.py L855–863) uses
   `if not state.get("phase_1_complete") and dag_phase != "phase_1_complete"` — the stale
   flag makes phase 2A passable despite a failed Phase 1.
3. **`climate_features.csv` ignored** — a pre-downloaded, competition-provided feature file
   (SRTM/ERA5/CHIRPS/NDVI aggregates) is never consumed by skill_02/skill_06/skill_07; the
   workspace loses access to a sanctioned external-data asset by omission.
4. **`metric: "multi"` semantics lossy** — no skill models the composite `0.6·F1 + 0.4·ROC-AUC`
   with the mandatory 0.5 binary threshold plus raw probability output.
5. **External-data conflict detected but resolved silently conservative** — the discussion
   scan flagged "External data: Rules tab vs Data tab conflict" (`community_signals` #4,
   `resolved_by_organizer: false`), yet `_resolve_external_banned` defaulted to banned and the
   config override cemented `external_banned=True`. A genuinely conflicting rule signal was
   surfaced for humans but then overridden by the same conservative default that caused the
   falsehood — the conflict should instead be an unresolved-operator-confirm state.

---

## 7. Root-cause chain for the Phase-1 crash

1. API provides no `target_col` → `skill_02` falls back to SampleSubmission inference
   (L780–814): first non-ID column → `TargetF1`.
2. `target_col` is written into `challenge_config.json` (L809).
3. `skill_04.detect_target()` reads config `target_col`; `skill_05_cv._resolve_target_col()`
   (L46–50) reads it too and later raises when `TargetF1` is absent from `features_train.csv`
   and raw `Train.csv` (L408–420).
4. Neither skill consults `data_dictionary.csv`, nor the balance/type of `is_climate_sensitive`.

---

## 8. RFC — proposed changes (not yet implemented)

Minimal scope proposed for review. Every item is source-located so it can be implemented
with or without an SoT change; items marked **[SoT]** need an SoT contract decision first.

### A. [SoT] Target resolution spread-sheet (unblocks Phase 1) — `skill_02_intake.py` L780–814
- After inferring `target_col` from SampleSubmission, validate the column exists in
  `Train.csv` (or `features_train.csv`).
- If absent, scan `data_dictionary.csv` for a row whose description marks the target
  (e.g. description containing "target"), and use that column for `target_col`.
- Record the submission layout separately, e.g.
  `submission_target_columns: ["TargetF1", "TargetRAUC"]` and
  `binary_threshold: 0.5` (fixed).
- SoT consequence: `target_col` currently means "the column skills read for training/eval".
  This RFC proposes a distinct `submission_target_columns` concept — needs an SoT contract
  addition before coding downstream.

### B. External-data truthfulness — `zindi_monitor_core.py` L97–113, L324–353; `skill_02_intake.py` L758–762
- `_resolve_external_banned`: add allow-phrases that match the verified permissive language
  ("encouraged to use publicly available", "permitted sources include",
  "you are welcome to download"), and run the same text check against the `/data` page body
  (the About text in §3.5 is the strongest match).
- Stop defaulting to banned when no phrase matches; leave an explicit "unknown → operator
  confirm" state instead of a silent `True`.
- When a discussion is flagged with an external-data conflict (e.g. "Rules tab vs Data tab
  conflict") and `resolved_by_organizer == false`, do **not** silently resolve it to banned —
  surface it as an unresolved operator-confirm state.
- `skill_02`: do not force `allowed_external_data=False` when the API is silent; inherit the
  monitor/page verdict or leave null for the preflight prompt.

### C. [SoT] Dual-output contract (binary + probability) — `config.py`, `skill_14_inference.py`,
   `skill_16_submit.py`
- Model submission output as `{binary: <col>, probability: <col>, threshold: 0.5}` for
  dual-column classification competitions, instead of the single `use_probabilities` bool.
- skill_14 emits both columns; skill_16 validates both against SampleSubmission.

### D. Wire the temporal-lock guard — `zindi_monitor_core.py` L1210–1212
- Use the computed `dag_phase` value to gate non-`community_signals` intel writes, or remove
  the dead line and make the policy explicit.

### E. `allow_teams` mapping — `skill_02_intake.py` L160
- Map API `allow_teams` → `team_allowed`.

### F. Phase-gate/flag hygiene — `zindian/orchestrator.py` L855–863; `skill_01_integrity.py` L73–81
- Reset `phase_1_complete`/`phase_2a_complete` to `false` when a phase re-run errors, and
  tighten the gate so it cannot pass on stale flags.

---

## 9. Open questions for the operator

**Operator answers (2026-09-01):**
1. Unauthorised Phase 2A run: *"The agent ran it without my authority"* → fix F implements phase-completion hygiene so a failed/unauthorised run cannot leave a passable `phase_*_complete` flag.
2. Target source: *"the data dictionary helps and it should help identify the target, unlike the way it has happen that target roc auc is the target"* → fix A implemented: `skill_02` validates `target_col` against the training columns and resolves the real target from `data_dictionary.csv` when the candidate is a submission-only column.

3. `climate_features.csv`: *"ye we will use but let us resolve the first issues"* → approved for later use; deferred until fixes A/B/D/E/F are baked and fix C (dual-output contract) is SoT-contracted.

Remaining decisions (C, climate_features ingestion) are tracked in §8 above.

1. Was an earlier session (13:19–13:44 UTC+03) allowed to run Phase 2A on this competition
   before Phase 1 succeeded? (The stale `phase_2a_complete: true` and `reports/summaries/2a_*`
   suggest yes — that is exactly the hazard item F targets.)
2. For fix A: is `is_climate_sensitive` (from `data_dictionary.csv`) the agreed canonical
   training target to hardwire the *fallback logic* toward (the column name itself must still
   be read dynamically, never hardcoded)?
3. Should the pre-downloaded `climate_features.csv` be surfaced to Phase 2B as a sanctioned
   feature source, given the page explicitly provides it?