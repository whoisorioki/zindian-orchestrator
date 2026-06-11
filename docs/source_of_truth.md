# Zindian Orchestrator — Source of Truth Document

**Version:** 2.0.1-Canonical
**Status:** Living Document — Approved for Production Automation
**Scope:** Zindi tabular competitions (standard, spatial, temporal, grouped)
**Last updated:** v2.0 → v2.0.1 — 2 label fixes applied
(Section 4 Phase 1 gate reference string advanced to v2.0.1;
Section 8 skill_21 DoD Guard Condition 4 synced to effective_variance_threshold label)

---

## Table of Contents

1. Assumptions
2. Core Architectural Principles
3. Preflight Validation
4. Phase Architecture
5. Research Sidecar
6. Reproducibility Contract
7. Is This Reinforcement Learning?
8. Definition of Done — Master Checklist

---

## 1. Assumptions

These are explicit assumptions the entire architecture rests on.
If any are violated, the affected sections must be revisited
before any competition run.

**A1 — Single competition at a time.**
The orchestrator manages one active competition.
`challenge_config.json` and `SKILL_STATE.json` are scoped to one
competition directory. Parallel competition support is out of scope.

**A2 — Tabular data only.**
All skills assume structured, tabular input. Image, text, audio,
or graph data are not handled. Spatial data is treated as tabular
with lat/lon columns and a group identifier.

**A3 — Zindi platform conventions.**
Submission format, leaderboard polling, discussion board clarifications,
and submission budget limits follow Zindi conventions. Skills that
interface with the platform (`skill_00`, `skill_16`) are
Zindi-specific.

**A4 — Supervised learning only.**
The target column is always known and present in training data.
Unsupervised tasks are out of scope, except `skill_21_pseudo_label`
which is semi-supervised and conditional.

**A5 — No hardcoded competition-specific values anywhere.**
No skill hardcodes competition-specific values. This explicitly
includes: column names, target names, metric names, coordinate
column names, dataset names, platform names, and competition
identifiers. Every string that varies between competitions must
be read from `challenge_config.json` via a config accessor.
No string literals for any of the above are permitted in any
skill body.

**A6 — `SKILL_STATE.json` is the single source of truth for
execution state.**
No skill holds internal state between runs. All execution state
is written to and read from `SKILL_STATE.json`.

**A7 — The OOF contract is universal.**
Every skill that generates or evaluates OOF scores uses the CV
strategy written by `skill_05_cv` to config. No skill defines
its own CV object. Every OOF output carries a `cv_strategy_id`
tag. The orchestrator validates this tag before passing scores
to any evaluating skill.

**A8 — Spatial signals are group signals.**
Lat/lon columns and station/site/location identifiers are treated
as group structure. No special spatial CV path exists —
`GroupKFold` handles all group structures uniformly. When only
a spatial signal is present and no explicit group column exists,
the spatial location block identifier from
`config["spatial_signal"]["group_col"]` is used as the group
column for `GroupKFold`.

**A9 — The research sidecar is non-blocking at every consumption
point.**
Sidecar skills write recommendations to `SKILL_STATE.json`.
Skills that read sidecar output treat it as optional enrichment.
The correct pattern at every consumption point is:

```python
sidecar_recommendations = SKILL_STATE.get(
    "sidecar_recommendations", default=[]
)
if not sidecar_recommendations:
    log("No sidecar recommendations — proceeding from fingerprint")
else:
    log(f"Sidecar recommendations consumed: "
        f"{len(sidecar_recommendations)} items")
```

A sidecar failure never halts the main pipeline. No skill ever
blocks on or raises an exception for missing sidecar keys.

**A10 — Python environment is stable and reproducible.**
A pinned environment lock is required and must be committed to the repository.
The canonical workflow uses an unpinned `requirements.in` plus `pip-compile`
from `pip-tools` to generate a pinned `requirements.txt` which is committed.
Example workflow (developer):

```bash
# install the compiler (once)
pip install --upgrade pip-tools

# produce a pinned requirements.txt from requirements.in
pip-compile requirements.in --output-file requirements.txt

# install in a fresh venv
pip install -r requirements.txt
```

`skill_22` verifies the presence of a committed `requirements.txt` and that
it was generated from a present `requirements.in` file. This pattern enables
reproducible, reviewable pinning while keeping the top-level intent in
`requirements.in`.

---

## 2. Core Architectural Principles

### Principle 1 — Three-Lens Decision Philosophy

At every phase, the orchestrator evaluates every decision through
three simultaneous lenses:

- **General lens** — What does the problem family, metric, and
  literature tell us?
- **Specific lens** — What does this dataset actually contain?
- **Generalisation lens** — Will this decision hold under
  distribution shift?

No phase is purely one lens. Every skill contributes to at
least one.

---

### Principle 2 — Config Boundary and State Immutability

`challenge_config.json` stores the structural competition
contract. `SKILL_STATE.json` records all execution tracking
state.

Writing to `challenge_config.json` is governed by a strict
**temporal boundary rule**:

```
PHASE 1 — MUTABLE WINDOW
  skill_01  writes: file_hashes
  skill_02  writes: all fingerprint fields, seed,
                    submission_budget, target_domain_bounds
  skill_03  writes: policy_filters (via policy_writer())
  skill_05  writes: cv_strategy block

  skill_04  does NOT write to challenge_config.json.
            It writes EDA outputs exclusively to
            SKILL_STATE.json. It participates in Phase 1
            as a profiling step whose outputs inform
            skill_05_cv before the config lock closes.

POST-PHASE 1 LOCK
  challenge_config.json becomes strictly read-only the
  moment the Phase 1 gate checklist passes.

  No core skill may write to challenge_config.json after
  this point. Any attempted write by a non-permitted
  skill is a hard error — written to SKILL_STATE.json
  and the pipeline halts.

  SOLE EXCEPTION: skill_00 may write asynchronously to
  the community_signals array at any time. No other
  field is writable by skill_00.
```

Skills are stateless functions. They read context, do work,
write outputs. The orchestrator is the only entity that reads
both files and decides what runs next.

---

### Principle 3 — The OOF Contract

```
skill_05_cv writes one CV strategy object to
challenge_config.json
    ↓
Every OOF-generating skill reads that object
(skills 07, 08, 09, 21)
Tags every OOF output with cv_strategy_id
    ↓
Orchestrator validates cv_strategy_id before passing
scores to evaluating skills
    ↓
Every OOF-evaluating skill reads the same strategy
(skills 10, 11, 12)
    ↓
skill_22 verifies the full contract before
reproducibility sign-off
```

Breaking this contract in any skill invalidates all
cross-branch score comparisons. A contract violation is a
hard halt — not a warning.

---

### Principle 4 — Dependency Chain Enforcement

Phases execute in strict cascading sequence. The orchestrator
enforces complete resolution of Phase 1 before Phase 2A,
Phase 2A before Phase 2B, and Phase 3A before Phase 3B.
This is a structural system dependency, not a sequential
preference.

---

### Principle 5 — Feedback Loops Over Blind Iteration

The orchestrator does not generate variants blindly. Every
rejected branch produces a written diagnosis. Every gate
failure feeds back into the next variant generation cycle.
The sidecar produces grounded, dataset-specific
recommendations, not generic advice.

---

### Principle 6 — Human Gates

Five human gates exist in the pipeline. Each requires an
explicit approval key in `SKILL_STATE.json` before the
orchestrator proceeds. These keys are never written by any
skill or by the orchestrator. They are written only by a
human operator.

When any gate key is absent, the orchestrator halts, surfaces
a human-readable prompt, and waits. It does not time out,
retry, or bypass under any condition.

This is the primary mechanism enforcing Zindi's AutoML
prohibition. No model selection, fusion, or submission
proceeds without explicit human confirmation.

Gate keys required:

```
human_gate_1_approved
    After anchor evaluation, before variant generation loop.

human_gate_2_{branch}_approved
    Per promoted branch, before candidate pool entry.

human_gate_3_approved
    Before skill_13 oracle fusion runs.

human_gate_4_approved
    Before skill_14 inference formatting runs.

human_gate_5_selection
    Final private leaderboard submission pair confirmed
    before competition close.
```

**CV Strategy Override Mechanism:**

If the operator reviewing Gate 1 finds the anchor
fold scores implausibly low or inconsistent with
expected metric range for this competition:

    [D] CHALLENGE CV STRATEGY — request comparison run

    [D] is only surfaced when the auto-selected strategy
    is TimeSeriesSplit or GroupKFold — the structurally
    constrained strategies that carry a genuine risk of
    misconfiguration. If the auto-selection already landed
    on KFold or StratifiedKFold (the default fallbacks),
    [D] is suppressed entirely. Running a comparison
    anchor against an identical or equivalent distribution
    split is computationally redundant and wastes pipeline
    resources without providing a meaningful alternative.

    Gate 1 prompt options by auto-selected strategy:
        TimeSeriesSplit or GroupKFold selected:
            [A] APPROVE  — accept auto-selected strategy
            [B] REJECT   — reject anchor, regenerate
            [C] CHALLENGE — override anchor inputs
            [D] CHALLENGE CV STRATEGY — comparison run
        KFold or StratifiedKFold selected:
            [A] APPROVE  — accept auto-selected strategy
            [B] REJECT   — reject anchor, regenerate
            [C] CHALLENGE — override anchor inputs
            ([D] suppressed — no meaningful alternative)

On selection of [C]:
    [C] permits the operator to rerun the anchor model
    with modified hyperparameters or a different model
    family. It does NOT permit modification of any field
    in challenge_config.json — the temporal lock holds.
    It does NOT permit changing the CV strategy — that
    is [D]'s domain. It does NOT permit adding or removing
    features beyond what skill_07 has already generated.

    Allowed modifications under [C]:
        Model hyperparameters (learning rate, depth,
          regularisation, n_estimators, etc.)
        Model family (e.g. XGBoost instead of LightGBM)
        Feature subset drawn from the existing
          skill_07 output — no new engineering

    Disallowed under [C] — hard prohibitions:
        Any write to challenge_config.json
        Any change to the CV strategy object
        Any new feature engineering outside skill_07
        Any change to seed or reproducibility settings

    Execution flow:
        Operator specifies modifications in writing
        Orchestrator reruns skill_08 with modified config
        New anchor OOF score written to SKILL_STATE.json
          under key: anchor_oof_score_challenged
        Original anchor_oof_score preserved unchanged
        Both scores surfaced to operator
        Operator selects preferred anchor with rationale
        If challenged anchor selected:
            anchor_oof_score = anchor_oof_score_challenged
            Write to SKILL_STATE.json:
            {
              "anchor_challenge": {
                "active": true,
                "modification": "<operator description>",
                "original_oof": 0.0,
                "challenged_oof": 0.0,
                "rationale": "<operator text>",
                "approved_by": "human_gate_1",
                "timestamp": ""
              }
            }
            All subsequent gate comparisons use the
            challenged anchor score as baseline
        If original anchor retained:
            anchor_challenge.active = false
            Pipeline proceeds with original anchor

On selection of [D]:
    Orchestrator runs a second anchor using
    StratifiedKFold (for classification) or
    KFold (for regression) on the same model and features
    Surfaces both OOF scores to the operator

    Operator selects preferred strategy with
    written rationale

On override selection:
    Write to SKILL_STATE.json only — NOT to
    challenge_config.json (config lock must not
    be broken):
    {
      "cv_strategy_override": {
        "active": true,
        "original_strategy": "<auto-selected>",
        "override_strategy": "<operator-selected>",
        "original_oof": 0.0,
        "override_oof": 0.0,
        "rationale": "<operator text>",
        "approved_by": "human_gate_1",
        "timestamp": ""
      }
    }

    All downstream OOF-generating skills check
    SKILL_STATE for cv_strategy_override.active using
    the canonical safe read pattern:
        override_active = SKILL_STATE.get(
            "cv_strategy_override", {}
        ).get("active", False)
    This pattern is mandatory at every access point.
    Direct key access (SKILL_STATE["cv_strategy_override"])
    causes a KeyError on any run without an override and
    must never be used.
    If override_active == true:
        use override_strategy from SKILL_STATE instead
        of cv_strategy from challenge_config.json
    cv_strategy_id tag on all OOF outputs reflects
    the active strategy (override or config)

    skill_22 must record override in reproducibility
    sign-off and history log

---

## 3. Preflight Validation

Preflight runs at every session start before any skill
executes. It is read-only — it makes no changes to config,
state, or data.

Preflight operates in two modes to prevent initialization
deadlocks.

---

### INIT Mode

**Triggered when:** `challenge_config.json` does not exist
or is completely unpopulated (Session 1 of a new competition).

**Purpose:** Allow the full Phase 1 skill sequence to run
and populate `challenge_config.json` from scratch.

**Skills permitted to run during INIT mode:**
`skill_01`, `skill_02`, `skill_03`, `skill_04`, `skill_05`,
`skill_15` — the complete Phase 1 sequence and nothing else.

**Checks performed in INIT mode:**

```
[ ] Competition workspace directory exists and is writable
[ ] Raw data files present in expected location
[ ] No conflicting SKILL_STATE.json from a prior run
[ ] Environment lock file present (requirements.txt)
[ ] No AutoML library imports in any skill body (static scan)
[ ] No cross-skill imports present (static scan)
```

**Checks skipped in INIT mode:**
All config completeness checks, seed checks, OOF contract
checks, file hash checks, policy filter checks, and human
gate status checks. These cannot pass before Phase 1 runs.

**Output:** `reports/preflight_INIT_{timestamp}.json`

**Proceeds to:** Phase 1 execution — `skill_01` through
`skill_15`.

---

### ENFORCE Mode

**Triggered when:** `challenge_config.json` exists and is
populated (all sessions after Session 1).

**Purpose:** Full validation before any skill runs.

**Checks performed in ENFORCE mode:**

```
Config completeness:
    All required fields present in challenge_config.json
    cv_strategy block present with all required subfields
    shap_leak_threshold, variance_gate_threshold,
      gate_margin set
    submission_budget.total, .daily, .used present
    reproducibility.seed present and set
    target_domain_bounds present (null allowed for
      classification)
    community_signals array present
    policy_filters array present
    use_probabilities present and set
    metric_direction present and set

State integrity:
    SKILL_STATE.json is valid JSON
    file_hashes in config match current raw data MD5s

OOF contract:
    All existing OOF scores carry cv_strategy_id tags
    Active strategy resolution (checked in this order):
        override_active = SKILL_STATE.get(
            "cv_strategy_override", {}
        ).get("active", False)
        1. If override_active == true: validate all tags
           against override_strategy from SKILL_STATE
        2. Else: validate all tags against cv_strategy
           block from challenge_config.json
    Tags that match the active strategy: PASS
    Tags that match config but not active override: FAIL
    No skill defines a CV object internally (static scan)

Architecture integrity:
    No skill_X module appears in any skill_Y import block
    No hardcoded competition strings in any skill body
    config temporal lock active — no post-Phase-1 writes
      by non-permitted skills

Zindi compliance:
    No AutoML library imports detected in any skill body
    Raw probability format confirmed in last submission
      if applicable
    Seed set and written to config
    Submission budget remaining > 0

Human gates:
    Status of all five gate keys reported
```

**Output:** `reports/preflight_ENFORCE_{timestamp}.json`

**Confirmation written to:**
`SKILL_STATE.json["preflight_confirmed"]`

---

### Preflight Prompt Surfaced to Operator

```
╔═══════════════════════════════════════════════════════════╗
║        ZINDIAN ORCHESTRATOR — SESSION PREFLIGHT           ║
║        Competition : {competition_id}                     ║
║        Mode        : {INIT | ENFORCE}                     ║
║        Date        : {timestamp}                          ║
╚═══════════════════════════════════════════════════════════╝

[CONFIG]
  competition_id       : {competition_id}
  task_type            : {task_type}
  metric               : {metric}
  metric_direction     : {maximize | minimize}
  use_probabilities    : {true | false}
  target_col           : {target_col}
  seed                 : {seed | NOT SET — must fix}
  submission_budget    : {remaining} remaining ({daily} today)
  cv_strategy          : {type} — {selection_reason}
  cv_override active   : {YES — override_strategy | no}
  active strategy      : {override_strategy | config cv_strategy}
  target_domain_bounds : {min, max | NOT INITIALIZED}
  external_data        : {external_data_allowed}
  automl_permitted     : FALSE (Zindi rule — hard prohibition)

[INTEGRITY]
  file hashes          : {PASS | FAIL — list mismatches}
  SKILL_STATE.json     : {valid | invalid}
  environment lock     : {present | MISSING}
  config lock          : {active | NOT LOCKED — Phase 1
                          incomplete}

[OOF CONTRACT]
  active strategy      : {override_strategy if override
                          active | config cv_strategy}
  cv_strategy_id tagged: {all tagged | N violations}
  single CV object     : {confirmed | VIOLATION — list skills}

[POLICY]
  policy_filters       : {N columns blocked}
  leaked_features      : {empty | N flagged}
  banned column check  : {PASS | FAIL}

[SIDECAR]
  skill_00             : {running | not started}
  skill_18 last run    : {timestamp | not yet run}
  skill_19 last run    : {timestamp | not yet run}
  skill_20 last run    : {timestamp | not yet run}
  unresolved hypotheses: {N pending}

[HUMAN GATES]
  Gate 1 — anchor review       : {approved | pending}
  Gate 2 — branches reviewed   : {N approved | pending}
  Gate 3 — fusion              : {approved | pending}
  Gate 4 — inference           : {approved | pending}
  Gate 5 — final selection     : {selected | not selected}

[ZINDI COMPLIANCE]
  automl usage detected: {none | WARNING — list}
  raw probabilities    : {confirmed | NOT CONFIRMED}
  seed reproducibility : {confirmed | NOT SET}
  submission selection : {2 selected | NOT YET SELECTED}
  code review ready    : {yes | NO}

─────────────────────────────────────────────────────────────
PREFLIGHT RESULT: {PASS | FAIL}

Failures must be resolved before proceeding.
Warnings require explicit acknowledgement.

  [1] PROCEED  — all checks pass
  [2] ABORT    — do not start this session
  [3] OVERRIDE — proceed despite warnings
                 (requires written reason)

Reason if OVERRIDE: ____________________________________
╚═══════════════════════════════════════════════════════════╝
```

---

## 4. Phase Architecture

### Phase 1 — Competition Fingerprint + Config Lock

**Skills:**
`skill_01` → `skill_02` → `skill_03` → `skill_04` →
`skill_05` → `skill_15`

**Purpose:** Extract raw metadata, profile structural signals,
generate the universal validation plan, lock data boundaries,
and write the CV strategy. Config locks after `skill_05`
writes.

---

**`challenge_config.json` — Required layout after Phase 1:**

```json
{
  "competition_id": "",
  "task_type": "classification | regression | ranking",
  "metric": "logloss | auc | rmse | mae | f1 | custom",
  "metric_direction": "maximize | minimize",
  "use_probabilities": true,
  "target_col": "",
  "target_domain_bounds": {
    "min": null,
    "max": null
  },
  "target_distribution":
    "balanced | imbalanced | continuous_normal | continuous_skewed",
  "minority_ratio": null,
  "data_shape": {
    "n_train": 0,
    "n_test": 0,
    "n_cols": 0
  },
  "temporal_signal": {
    "present": false,
    "col": null
  },
  "group_signal": {
    "present": false,
    "col": null,
    "type": null
  },
  "spatial_signal": {
    "present": false,
    "lat_col": null,
    "lon_col": null,
    "group_col": null
  },
  "missingness_level": "low | moderate | high",
  "external_data_allowed": false,
  "submission_budget": {
    "total": 0,
    "daily": 0,
    "used": 0
  },
  "community_signals": [],
  "file_hashes": {},
  "policy_filters": [],
  "phase_skill_map": {},
  "reproducibility": {
    "seed": 42
  },
  "shap_leak_threshold": 3.0,
  "variance_gate_threshold": 0.01,
  "gate_margin": 0.001,
  "cv_strategy": {
    "type": "",
    "n_splits": 5,
    "shuffle": false,
    "random_state": 42,
    "group_col": null,
    "stratify_col": null,
    "selection_reason": ""
  }
}
```

---

**`skill_03_legality` — Internal structural breakout:**

`skill_03` is implemented as two explicitly separated,
independently testable functions.

```
policy_writer():
    Reads : challenge_config.json,
            community_signals from skill_00
    Writes: reports/feature_policy.json
    Fields: allowed_features, blocked_features,
            block_reasons
    Side effects: none — pure writer, no gating

policy_gate():
    Reads : reports/feature_policy.json,
            current feature matrix column list
    Asserts: no blocked column present in feature matrix
    On violation:
        Write to SKILL_STATE.json
        Halt pipeline
    Side effects: state write and halt only
```

`policy_writer()` runs in Phase 1.
`policy_gate()` runs as the first action of Phase 2A.

---

**`skill_04_eda` — Writes to `SKILL_STATE.json` only:**

```json
{
  "eda": {
    "mnar_columns": [],
    "mcar_columns": [],
    "outlier_columns": [],
    "target_skew": 0.0,
    "target_std": 0.0,
    "group_structure_confirmed": false,
    "temporal_index_confirmed": false
  }
}
```

`target_std` is the standard deviation of the target column
computed on the full training set. It is used by `skill_11`
and `skill_12` to normalise `gate_margin` and
`variance_gate_threshold` for regression tasks where raw
thresholds are scale-sensitive. Written during Phase 1 before
the config lock closes.

`skill_04` does not write to `challenge_config.json`.
Its outputs inform `skill_05_cv` before the config lock
closes.

---

**`skill_05_cv` — Full decision tree:**

```
Intra-phase dependency:
    skill_04 must complete and write EDA outputs to
    SKILL_STATE.json before skill_05_cv reads them.
    The orchestrator enforces this ordering within
    Phase 1 before the config lock closes.

Step 1 — Temporal check:
  If SKILL_STATE["eda"]["temporal_index_confirmed"] == true:
      cv_strategy = TimeSeriesSplit
      shuffle     = false
      Reason      : look-ahead bias prevention
      → Go to write

Step 2 — Group / Spatial check:
  Else if SKILL_STATE["eda"]["group_structure_confirmed"]
          == true
       OR config["spatial_signal"]["present"] == true:

      cv_strategy = GroupKFold

      Group col resolution:
          If config["group_signal"]["present"] == true:
              group_col = config["group_signal"]["col"]
          Else if config["spatial_signal"]["present"]
                  == true:
              group_col =
                config["spatial_signal"]["group_col"]
          (spatial_signal.group_col is written by
           skill_02 from the competition's location
           identifier column)

      Reason: group leakage prevention
      → Go to write

Step 3 — Imbalance check (classification only):
  Else if config["task_type"] == "classification"
       AND config["minority_ratio"] < 0.15:
      cv_strategy = StratifiedKFold
      Reason      : minority class fold stability
      → Go to write

Step 4 — Default:
  Else:
      cv_strategy  = KFold
      shuffle      = true
      random_state = config["reproducibility"]["seed"]
      Reason       : standard regression or balanced
                     classification fallback
      → Go to write

Write:
  All cases: n_splits = 5 (configurable in config)
  Write cv_strategy block to challenge_config.json
  Write selection_reason
  Config locks immediately after this write
```

All column names and metric identifiers read from config.
No string literals permitted in `skill_05` body.

---

**Three-lens check — Phase 1:**

- **General:** Is task type, metric, and CV strategy correctly
  identified for this problem family?
- **Specific:** Do `skill_04` EDA outputs confirm the signals
  detected by `skill_02`? Does CV strategy reflect actual
  dataset structure?
- **Generalisation:** Are file hashes locked? Are policy
  filters written? Is CV strategy committed to config before
  any model work begins?

---

**Phase 1 → Phase 2A gate:**

```
[ ] challenge_config.json matches v2.0.1 schema — all
    fields present and non-null where required
[ ] task_type, metric, target_col confirmed
[ ] metric_direction written and set
[ ] use_probabilities written and set
[ ] target_domain_bounds written if task_type == regression
[ ] file_hashes locked and written by skill_01
[ ] policy_filters written by skill_03 policy_writer()
[ ] reports/feature_policy.json present, non-empty,
    and valid JSON
[ ] feature_policy.json contains required keys:
    allowed_features, blocked_features, block_reasons
[ ] blocked_features contains at minimum all columns
    listed in challenge_config["policy_filters"]
[ ] skill_04 EDA outputs present in SKILL_STATE.json
    — verified BEFORE skill_05 runs
[ ] If task_type == regression:
    target_std present in SKILL_STATE["eda"]["target_std"]
    — required for effective_gate_margin and
    effective_variance_threshold normalisation in skill_11
[ ] skill_05 cv_strategy written with selection_reason
    — only valid after skill_04 outputs confirmed
[ ] spatial_signal.group_col populated if spatial_signal
    present and group_signal absent
[ ] challenge_config.json temporal lock active —
    confirmed read-only
[ ] seed written to config
[ ] skill_15 has logged all Phase 1 write events
```

---

### Phase 2A — Data Cleaning

**Skills:** `policy_gate()` → `skill_06`

**Purpose:** Enforce feature exclusions. Apply
missingness-aware cleaning under immutable config.
Config is read-only from this point.

---

**`policy_gate()` runs first:**

```
Reads: reports/feature_policy.json
Asserts: all blocked columns absent from feature matrix
On violation:
    Write violation entry to SKILL_STATE.json
    Halt — do not proceed to skill_06
Proceeds to skill_06 only if gate passes
```

---

**`skill_06_cleaning` — Imputation pipeline:**

```
For each column in SKILL_STATE["eda"]["mnar_columns"]:
    Step 1: Create binary indicator
            col_name + "_is_missing"
    Step 2: Fill missing positions with 0
    ORDER IS MANDATORY — indicator before fill.
    Filling before indicator creation destroys the
    missingness signal permanently.

For each column in SKILL_STATE["eda"]["mcar_columns"]:
    If numeric   : fill with column median
    If categorical: fill with column mode

Constant columns:
    Drop unconditionally from feature space
```

---

**Three-lens check — Phase 2A:**

- **General:** Are cleaning rules applied without manual
  intervention or data distortion?
- **Specific:** Does imputation match the missingness profile
  found by `skill_04`?
- **Generalisation:** Are MNAR indicators generated before
  fills? Does dropping constant columns avoid
  over-parameterisation?

---

**Phase 2A → Phase 2B gate:**

```
[ ] policy_gate() passed — all blocked columns absent
    from feature matrix
[ ] skill_06 cleaning complete
[ ] MNAR indicator columns generated before any fill —
    order verified
[ ] MCAR columns filled with median/mode
[ ] Constant columns dropped
[ ] Cleaning outputs written to SKILL_STATE.json
```

---

### Phase 2B — Signal Search

**Skills:** `skill_08` → `skill_07`

**Purpose:** Establish the anchor baseline. Search for
signal-improving feature variants. All execution reads
Phase 1 outputs from config.

**Human Gate 1 triggers after `skill_08_anchor` completes.**

---

**`skill_08_anchor` contract:**

- Reads `cv_strategy` from `challenge_config.json` —
  never defines its own
- Uses `config["reproducibility"]["seed"]` for all training
- Writes anchor OOF score, branch name, model config,
  and `cv_strategy_id` tag to `SKILL_STATE.json`
- Anchor score is the immutable comparison point for all
  subsequent gating

---

**`skill_07_features` — Engineering rules engine:**

```
If config["temporal_signal"]["present"] == true:
    → Lag features, rolling means, time-since features

If config["spatial_signal"]["present"] == true:
    → Haversine distance to spatial centroid
        Structural feature — uses coordinates only, not
        target values. Compute on full dataset at any time.
        No two-mode contract applies.
    → Nearest-neighbour distance arrays
        Structural feature — uses coordinates only.
        Compute on full dataset at any time.
        No two-mode contract applies.
    → Spatial lag of target
        TARGET-DEPENDENT feature. Two-mode contract:
            During CV validation passes:
                Computed using training fold targets only —
                never using validation fold targets
            During final model training for test inference:
                Computed using full training set targets
        Omitting the inference mode causes a column
        mismatch crash at skill_14.

If config["target_distribution"] == "continuous_skewed"
   AND config["task_type"] == "regression":
    → Log1p transform on target during training pipelines

If config["missingness_level"] == "high":
    → Interaction terms between MNAR indicator columns
      and top SHAP features from anchor

If config["group_signal"]["present"] == true:
    → Group aggregations (mean, std, count)
        Aggregations of the TARGET column are
        TARGET-DEPENDENT. Two-mode contract:
            During CV validation passes:
                Computed using training fold rows only —
                never using validation fold rows
            During final model training for test inference:
                Computed using all training rows
        Omitting the inference mode causes a column
        mismatch crash at skill_14.
        Aggregations of non-target columns (e.g. group
        size counts, group feature means) are structural —
        no two-mode contract required for those.

Sidecar enrichment (non-blocking — see A9):
    Read SKILL_STATE.get("sidecar_recommendations",
                          default=[])
    If present: enrich variant generation
    If absent : proceed from fingerprint alone
```

---

**Three-lens check — Phase 2B:**

- **General:** Does the anchor model match the metric?
  (e.g. LightGBM with `eval_metric: auc` for AUC
  competitions)
- **Specific:** Are feature variants informed by what
  `skill_04` found in this dataset?
- **Generalisation:** Do TARGET-DEPENDENT features
  (spatial lag of target, group aggregations of target)
  follow the two-mode contract — fold-restricted during
  CV, full-training-set targets during final inference?
  Structural features (Haversine, nearest-neighbour,
  non-target group counts) do not require two-mode
  treatment. Missing the inference mode on target-
  dependent features crashes skill_14. Missing the fold
  restriction silently inflates OOF scores.

---

**Phase 2B → Phase 3A gate:**

```
[ ] Human Gate 1 approved
[ ] Anchor OOF score present and cv_strategy_id tagged
    in SKILL_STATE.json
[ ] At least one feature variant OOF score present
    and tagged
[ ] All OOF scores generated using CV strategy from
    config — no internal CV objects anywhere
```

---

### Phase 3A — Generalisation Audit

**Skills:** `skill_10` → `skill_09` → `skill_12`

**Purpose:** Stress-test everything built in Phase 2B.
No promotion decision is made before this phase completes.

---

**`skill_10_shap` — SHAP computation contract:**

```
For each CV fold:
    Train model on training fold rows
    Compute SHAP values on validation fold OOF
    predictions only
    Store per-fold SHAP arrays

After all folds:
    Aggregate: mean |SHAP| across all folds per feature
    Apply threshold comparison to aggregated values only

FULL-TRAIN SHAP IS STRICTLY PROHIBITED.
Computing SHAP on full-train predictions introduces the
target into the computation and makes leak detection
unreliable.
```

---

**`skill_10_shap` — Active gate logic:**

```
For each feature variant branch:

    If feature count < 2:
        Relative SHAP leak audit is skipped — the ratio
        comparison (top / second_highest) is undefined
        with fewer than two features.
        Branch is evaluated on fold variance gate alone.
        Branch is NOT automatically promoted — all other
        skill_11_gate conditions still apply. Only the
        SHAP ratio check is skipped, not the gate itself.
        Write to SKILL_STATE.json:
            shap_audit_skipped_reason: "single_feature"
        Proceed to skill_11 gating.

    Else:
        If any feature mean(|SHAP|) >
           config["shap_leak_threshold"] × second_highest:

            Flag branch as probable target leak
            Write to SKILL_STATE.json:
                leaked_features: [branch_name]
            Block branch from skill_11_gate promotion list
            Issue drop-and-regenerate directive to skill_07
```

---

**`skill_09_calibration` contract:**

- Uses identical CV folds as `skill_08` — never a fresh
  split
- Classification tasks only
- Writes calibrated OOF predictions tagged with
  `cv_strategy_id` to `SKILL_STATE.json`

---

**`skill_12_metric` outputs:**

```json
{
  "metric_analysis": {
    "fold_scores": [],
    "fold_score_variance": 0.0,
    "recommended_threshold": 0.5,
    "oof_vs_lb_delta": null
  }
}
```

`fold_score_variance` is computed as unbiased sample variance
with ddof=1 (n-1 denominator). CV fold scores are a sample of
possible data splits, not the full population. For n=5 folds,
population variance (ddof=0) underestimates by a factor of
5/4 = 1.25 — a meaningful difference at the
`variance_gate_threshold: 0.01` boundary.

`fold_score_variance` written here is the raw unbiased sample
variance (ddof=1). Normalisation by `target_std` for regression
tasks occurs at `skill_11` gate consumption time, not here.
`skill_12` writes the raw value; `skill_11` computes
`effective_variance_threshold = variance_gate_threshold *
(target_std ** 2)` when `task_type == regression`.

For classification tasks, the raw `variance_gate_threshold`
is used directly at `skill_11` — bounded metrics need no
scale correction and `skill_12` output is consumed as-is.

High fold score variance signals the model is not
generalising uniformly across the distribution.

---

**Three-lens check — Phase 3A:**

- **General:** Is calibration appropriate for this metric?
  (Matters for logloss, less so for AUC)
- **Specific:** Are fold scores consistent, or is one fold
  an outlier indicating distribution shift?
- **Generalisation:** Is the SHAP audit passing? Is fold
  variance within `variance_gate_threshold`?

---

**Phase 3A → Phase 3B gate:**

```
[ ] SHAP audit complete for all candidate branches
[ ] leaked_features evaluated and written for all branches
[ ] Fold score variance computed and written for all
    branches
[ ] Calibration complete for classification tasks
[ ] All OOF outputs carry cv_strategy_id tags
```

---

### Phase 3B — Promotion and Fusion

**Skills:** `skill_11` → `skill_21` → `skill_13`

**Purpose:** Promote validated variants. Optionally expand
training data with pseudo-labels. Fuse candidates into
final submission.

**Human Gate 2 triggers after each `skill_11_gate` pass,
before candidate pool entry.**
**Human Gate 3 triggers before `skill_13_oracle_fusion`
runs.**

---

**`skill_11_gate` — Promotion conditions (ALL 5 must pass):**

```
1. Branch is absent from leaked_features list
2. fold_score_variance < effective_variance_threshold
   fold_score_variance computed with ddof=1 (unbiased
   sample variance, n-1 denominator) — consistent with
   the estimator used in skill_12
   Threshold normalisation:
       If config["task_type"] == "regression":
           effective_variance_threshold =
               config["variance_gate_threshold"] *
               (SKILL_STATE["eda"]["target_std"] ** 2)
           (raw threshold is scale-sensitive for
           regression metrics; target_std written by
           skill_04 during Phase 1)
       If config["task_type"] == "classification":
           effective_variance_threshold =
               config["variance_gate_threshold"]
           (classification metrics are bounded —
           no scale normalisation needed)
3. OOF improvement over anchor passes directional check:
       Effective gate margin:
           If config["task_type"] == "regression":
               effective_gate_margin =
                   config["gate_margin"] *
                   SKILL_STATE["eda"]["target_std"]
               (gate_margin: 0.001 is trivially small for
               RMSE on a target with σ_y = 500; σ_y
               normalisation makes the threshold scale-
               invariant across competitions)
           If config["task_type"] == "classification":
               effective_gate_margin = config["gate_margin"]
               (metrics are bounded — no normalisation)
       Baseline selection (safe lookup — pseudo_label_result
       may not exist on first pass through skill_11):
           retraining_active = SKILL_STATE.get(
               "pseudo_label_result", {}
           ).get("retraining_required", False)
           challenge_active = SKILL_STATE.get(
               "anchor_challenge", {}
           ).get("active", False)
           If retraining_active == true:
               baseline = anchor_oof_score_augmented
               # Augmented baseline takes precedence over
               # anchor_challenge because the training set
               # has changed and the original/challenged
               # baseline comparison is no longer valid.
           Else if challenge_active == true:
               baseline = anchor_oof_score_challenged
           Else:
               baseline = anchor_oof_score
       If config["metric_direction"] == "maximize":
           oof_score - baseline > effective_gate_margin
       If config["metric_direction"] == "minimize":
           baseline - oof_score > effective_gate_margin
4. skill_10 SHAP audit passed for this branch
5. human_gate_2_{branch}_approved present and true
   in SKILL_STATE.json
```

Gate failure writes a complete diagnosis to
`SKILL_STATE.json` and triggers an automated
`skill_20` on-demand run.

---

**`skill_21_pseudo_label` — Full contract:**

Guard conditions — ALL must be true before running:

```
1. config["task_type"] == "classification"
   (skill_21 is classification-only — skill_09 does not
   run for regression tasks, making Guard Condition 4
   permanently unresolvable for regression competitions.
   This guard makes that scope explicit rather than
   silently failing at Condition 4 every regression run.)
2. config["cv_strategy"]["type"] != "TimeSeriesSplit"
3. SKILL_STATE["leaked_features"] is empty
4. fold_score_variance < effective_variance_threshold
   (Guard Condition 1 ensures task_type is always
   "classification" here, so effective_variance_threshold
   equals raw variance_gate_threshold — no target_std
   normalisation applies. Stated as effective_variance_threshold
   for consistency with the normalised threshold system used
   in skill_11 and skill_12.)
5. Calibrated probabilities present from skill_09
6. Confident prediction threshold met
   (default: top 10% by confidence score)
```

On guard pass:

```
Assign target labels to selected test rows
Append pseudo-labelled rows to training matrix

Write to SKILL_STATE.json:
{
  "pseudo_label_result": {
    "ran": true,
    "n_pseudo_labels_added": <int>,
    "retraining_required": true,
    "guard_conditions_met": true,
    "guard_failure_reason": null,
    "execution_failure_reason": null,
    "guard_condition_flags": {
      "gc1_classification": true,
      "gc2_not_timeseries": true,
      "gc3_no_leaked_features": true,
      "gc4_variance_within_threshold": true,
      "gc5_calibrated_probs_present": true,
      "gc6_confidence_threshold_met": true
    }
  }
}

Orchestrator intercepts pipeline flow:
    Retrain anchor model (skill_08) on augmented dataset
    Write augmented anchor OOF score to SKILL_STATE.json
    under key: anchor_oof_score_augmented
    Original anchor_oof_score is preserved unchanged —
    it remains the pre-pseudo-label governance reference

    Pseudo-label CV fold assignment contract:
        The Phase 1 CV object is NOT rebuilt. Its fold
        index layout (rows 0 to N_train-1) is unchanged.
        Pseudo-labeled rows (indices N_train onward) are
        assigned to the training split of EVERY fold.
        Pseudo-labeled rows are NEVER assigned to any
        validation fold.
        OOF evaluation indices remain strictly identical
        to the pre-augmented Phase 1 split layout.
        Violating this contract causes either an
        IndexError crash or silent OOF score inflation
        from evaluating models on their own labels.

    Augmented variant OOF namespace contract:
        All OOF arrays generated during the pseudo-label
        retraining loop are written to isolated keys with
        the suffix _augmented. The naming convention is:
            branch_{branch_name}_oof_augmented
        The original pre-pseudo-label OOF arrays remain
        under their original keys:
            branch_{branch_name}_oof
        The retraining loop NEVER overwrites original OOF
        keys. Writing to an existing non-augmented key
        during this loop is a hard error — written to
        SKILL_STATE.json and the pipeline halts.
        This isolation ensures rollback can safely clear
        all _augmented keys without touching the original
        candidate pool records.

    Trigger targeted rerun of promoted model branches
    via skill_07 and skill_08 on augmented dataset

    New OOF scores written under augmented namespace:
        branch_{branch_name}_oof_augmented
    Tagged with cv_strategy_id

    skill_11 gate condition 3 uses
    anchor_oof_score_augmented as the baseline when
    retraining_required == true — never the original
    anchor_oof_score (different training sets make that
    comparison mathematically invalid)

    Pass retrained branches through skill_10 SHAP audit
    Pass retrained branches through skill_11 gate
    Require human_gate_2_{branch}_approved re-approval
    for each retrained branch

    If at least one retrained branch passes skill_11:
        Orchestrator proceeds to skill_13 using
        retrained candidate pool (augmented OOF arrays)

    If ZERO retrained branches pass skill_11
    (pseudo-label rollback path):
        Orchestrator aborts pseudo-labeling
        Clears from SKILL_STATE.json — augmented keys only:
            anchor_oof_score_augmented
            All branch_{name}_oof_augmented keys
        Original branch_{name}_oof keys are untouched —
        rollback targets only the _augmented namespace
        Resets pseudo_label_result.ran to false
        Writes pseudo_label_result.execution_failure_reason:
            "retrain_gate_failure_rollback"
        (guard_failure_reason remains null — all six guards
        passed; this is a post-guard execution failure)
        Restores original candidate pool — the branches
        that passed skill_11 BEFORE skill_21 ran, using
        their original branch_{name}_oof arrays
        Proceeds to skill_13 using original clean pool
        Original anchor_oof_score remains the governance
        reference throughout
```

On guard failure:

```
Write to SKILL_STATE.json:
{
  "pseudo_label_result": {
    "ran": false,
    "n_pseudo_labels_added": 0,
    "retraining_required": false,
    "guard_conditions_met": false,
    "guard_failure_reason":
      "not_classification | timeseries | leaked_features |
       high_variance | no_calibration | low_confidence",
    "execution_failure_reason": null,
    "guard_condition_flags": {
      "gc1_classification": <true|false>,
      "gc2_not_timeseries": <true|false>,
      "gc3_no_leaked_features": <true|false>,
      "gc4_variance_within_threshold": <true|false>,
      "gc5_calibrated_probs_present": <true|false>,
      "gc6_confidence_threshold_met": <true|false>
    }
  }
}

Proceed directly to skill_13 without retraining
```

---

**`skill_13_oracle_fusion` — Diversity and compliance
contract:**

```
Restrict inputs to branches that:
    Cleared skill_11 with all 5 conditions
    Have human_gate_2_{branch}_approved == true

For each pair of candidates:
    Correlation metric selection:
        If task_type == "classification":
            Use Pearson correlation — measures linear
            correlation in probability space; appropriate
            for calibrated outputs where ensemble blending
            operates on the linear scale.
            For use_probabilities == False (hard labels),
            Pearson on binary 0/1 outputs is equivalent
            to the phi coefficient — valid for diversity
            checking, no special handling required.
        If task_type == "regression":
            Use Spearman rank correlation — measures
            monotonic consistency between model outputs;
            appropriate when the relationship between two
            regression models may be monotonic but not
            linearly proportional.
    If correlation > 0.95:
        Drop the lower-scoring candidate

All candidates must have:
    Seeds set and logged
    Open-source tools only
    Fusion strategy explainable from config and state
    alone
```

---

**Three-lens check — Phase 3B:**

- **General:** Does the fusion strategy make sense for this
  problem family?
- **Specific:** Is the candidate pool diverse enough to
  provide genuine variance reduction?
- **Generalisation:** Can this fusion be explained and
  reproduced for Zindi code review within 48 hours?

---

**Phase 3B → Phase 4 gate:**

```
[ ] At least one branch promoted through skill_11
[ ] Human Gate 2 approved for all promoted branches
[ ] If skill_21 ran with retraining_required == true:
        guard_condition_flags verified — all six gc fields
          present with Boolean values in SKILL_STATE
        gc1_classification: task_type == "classification"
        gc2_not_timeseries: cv_strategy != TimeSeriesSplit
        gc3_no_leaked_features: leaked_features empty
        gc4_variance_within_threshold: fold variance check
        gc5_calibrated_probs_present: skill_09 output found
        gc6_confidence_threshold_met: top 10% threshold met
        Pseudo-label CV fold assignment contract verified:
          augmented rows in train splits only,
          OOF indices identical to Phase 1 layout
        Anchor retrained on augmented dataset —
          anchor_oof_score_augmented present in
          SKILL_STATE.json
        Retrained branches have new OOF scores tagged
          with cv_strategy_id
        skill_11 gate condition 3 compared retrained
          branch scores against anchor_oof_score_augmented
        skill_10 SHAP audit passed on retrained branches
        skill_11 gate passed on retrained branches
        Human Gate 2 re-approved for retrained branches
        If zero retrained branches passed skill_11:
          rollback confirmed — all _augmented keys cleared,
          original branch_{name}_oof arrays verified intact,
          execution_failure_reason: retrain_gate_failure_rollback,
          original candidate pool used for fusion
[ ] skill_13 uses most recent OOF arrays only —
    never stale pre-pseudo-label arrays
[ ] Human Gate 3 approved before fusion runs
[ ] Fusion diversity check complete —
    collinear candidates dropped
[ ] Final submission candidate identified
```

---

### Phase 4 — Governance

**Skills:** `skill_14` → `skill_16` → `skill_17` →
`skill_22`

**Purpose:** Format prediction arrays, run compliance
checks, submit, select final entries, and lock all
reproducible assets.

**Human Gate 4 triggers before `skill_14_inference` runs.**
**Human Gate 5 triggers before competition close.**

---

**`skill_14_inference` — Validation schema:**

```
Read task_type from challenge_config.json
Read use_probabilities from challenge_config.json

If task_type == "classification"
   AND use_probabilities == True:
    Assert all values within open interval (0, 1)
    Assert no rounding or threshold modification applied
    Warn if any value has fewer than 6 decimal places
    Confirm raw probability distribution preserved
      end-to-end

If task_type == "classification"
   AND use_probabilities == False:
    Assert all values satisfy: val == 0 or val == 1
      (value equality check — 0.0 and 1.0 pass,
       type is not asserted)
    Hard failure if any value does not equal 0 or 1
      (e.g. 0.5, 0.7, 1.3 — these are real errors,
       not warnings)
    No probability range checks applied

If task_type == "regression":
    Assert all values within target_domain_bounds
      recorded in challenge_config.json
    Assert no NaN, null, or infinite values
    Warn if output distribution variance is implausibly
      narrow (signal of mean collapse — broken model)
    Skip all probability-specific checks entirely

All task types:
    Assert row count matches test set exactly
    Assert ID column matches test set exactly
    Assert no duplicate ID markers
    Assert file format matches competition submission
      schema
```

---

**`skill_16_submit` — Budget management protocol:**

```
Before any Zindi API call:

    Read remaining_submissions from Zindi client

    If remaining_submissions <= 0:
        Write to SKILL_STATE.json:
            {
              "submission_blocked": true,
              "reason": "budget_exhausted"
            }
        Raise HardAbortException
        No API call under any condition

    If remaining_submissions == 1:
        Write warning to SKILL_STATE.json
        Halt thread
        Prompt human operator for explicit confirmation
        before proceeding
```

---

**`skill_17_governance` outputs:**

- Final submission selection documented with reasoning
- CV strategy used recorded
- All Human Gate approvals referenced by timestamp
- Gate 5 final pair selection recorded

---

**`skill_22_reproducibility_audit` — Sign-off checklist:**

```
[ ] challenge_config.json complete and schema-valid
[ ] cv_strategy block present with selection_reason
[ ] All OOF scores carry cv_strategy_id tags matching
    config
[ ] No skill defines a CV object internally
    (verified by static scan)
[ ] leaked_features empty for all promoted branches
[ ] File hashes match current raw data files
[ ] Environment lock file present and committed
[ ] No custom packages in any skill body
[ ] Seed written to config and logged in all OOF outputs
[ ] Submission file reproducible from config and state
    alone
[ ] All five human gate approvals recorded with
    timestamps
[ ] All sidecar recommendations resolved in
    SKILL_STATE.json
[ ] SKILL_STATE.json contains complete execution trace
[ ] Pipeline replayable from challenge_config.json and
    SKILL_STATE.json alone
[ ] If skill_21 ran with retraining_required == true:
        guard_condition_flags present with all six gc
        fields populated as Booleans
        guard_failure_reason covers only gc1–gc6 failures
        execution_failure_reason covers post-guard failures
        Pseudo-label fold contract verified: augmented
          rows in train splits only, OOF indices unchanged
        anchor_oof_score_augmented present in
        SKILL_STATE.json
        Retrained branch OOF scores present and tagged
        with cv_strategy_id
        Retrained branch gate comparisons used
        anchor_oof_score_augmented as baseline
        skill_10 and skill_11 passed on retrained
        branches confirmed
        If rollback occurred: all _augmented keys cleared,
          original branch_{name}_oof keys verified intact,
          execution_failure_reason written,
          confirmed in SKILL_STATE.json
[ ] If CV strategy override used at Gate 1:
        cv_strategy_override block present in
        SKILL_STATE.json with rationale and timestamp
        Override rationale present and non-empty
        Override recorded in governance report
          and history log
[ ] Competition history log entry written in correct
    schema to competition_history/history_log.jsonl
[ ] All required history log fields populated before
    sign-off
[ ] cv_strategy_override and rationale recorded in
    history log if Gate 1 override occurred
[ ] Cross-competition history log updated
```

---

**Three-lens check — Phase 4:**

- **General:** Does the governance report document the full
  decision chain clearly enough for a code reviewer?
- **Specific:** Is the submission file correct for this
  competition's exact schema and task type?
- **Generalisation:** Can a third party reproduce the exact
  submission from config and state alone?

---

**Phase 4 → Done gate:**

```
[ ] skill_22 reproducibility checklist fully passes
[ ] Human Gates 4 and 5 approved and recorded
[ ] Submission budget not exceeded
[ ] Governance report written and signed off
[ ] Cross-competition history log updated
```

---

## 5. Research Sidecar

**Skills:** `skill_00` (continuous) → `skill_18`,
`skill_19`, `skill_20` (triggered)

The sidecar is not a phase. It is a continuous intelligence
layer with distinct trigger points. It is non-blocking at
every consumption point (see A9).

---

### Trigger Schedule

| Skill | Trigger | Informs |
|---|---|---|
| `skill_00` | Competition start → close | All phases continuously |
| `skill_18` | Phase 1 completes | Phase 2B feature generation |
| `skill_19` | Phase 2A completes | Phase 2B feature patterns |
| `skill_20` | Phase 2B completes (anchor + SHAP ready) | Phase 3A audit, Phase 2B next variants |
| `skill_18` on-demand | `skill_20` raises unresolved hypothesis | `skill_20` validation loop |
| `skill_20` on-demand | `skill_11_gate` rejects a branch | Next `skill_07` variant generation |

---

### `skill_00` — Specific triggers

```
On competition intake (skill_02 completes):
    Begin polling discussion board
    Begin polling leaderboard

On every skill_16 submission:
    Record public LB score vs OOF delta
    drift_threshold = SKILL_STATE.get(
        "drift_threshold",
        config.get("drift_threshold", 0.05)
    )
    If delta > drift_threshold:
        Flag overfit risk to skill_11_gate

On any admin post announcing data patch or schema change:
    Issue absolute halt across active pipeline threads
    Write to SKILL_STATE.json:
        {
          "data_patch_detected": true,
          "patch_halt_timestamp": "<timestamp>",
          "patch_description": "<scraped post summary>"
        }
    Surface human decision gate — do NOT automatically
    trigger skill_02 re-intake. Automatic re-intake would
    break the config temporal lock and invalidate all OOF
    scores computed against the pre-patch X and y arrays.

    Human operator must choose one of:
        [R] RESTART — wipe all pipeline state, delete all
            OOF arrays, unlock config, run full Phase 1
            sequence from scratch on patched data
        [A] ABORT   — terminate competition run entirely,
            log patch as invalidating event in history log

    No downstream skill executes until one of these keys
    is written to SKILL_STATE.json by the operator.
    skill_00 does NOT reset skill_01 hash registries
    automatically — that action is part of the [R] path
    executed only after human confirmation.

Every 6 hours (configurable):
    Scrape new discussion posts
    Write findings to
      challenge_config.json["community_signals"]
    (skill_00 is the only permitted post-lock writer
     to challenge_config.json)
```

---

## 6. Reproducibility Contract

Every skill that trains a model must comply with all four
requirements.

**R1 — Seed is always set.**

```
Every model training call must set:
    random_state = config["reproducibility"]["seed"]
    numpy.random.seed(config["reproducibility"]["seed"])
    random.seed(config["reproducibility"]["seed"])

Seed written to challenge_config.json at Phase 1.
Never changed after Phase 1 completes.
Logged in every OOF output in SKILL_STATE.json.
```

**R2 — Rerun produces identical output.**

```
Rerunning any skill with identical config and state
must produce bit-identical OOF scores and submission
files. Non-reproducible skills must not be promoted
or submitted.
```

**R3 — No custom packages.**

```
All packages listed in requirements.txt or environment
lock file. No private or custom packages in any skill
body. skill_22 verifies this via import scanning at
sign-off.
```

**R4 — Submission file is reproducible from config and
state alone.**

```
Given challenge_config.json and SKILL_STATE.json, a
third party must be able to regenerate the exact
submission file. skill_22 verifies this before final
governance sign-off. This is the standard Zindi code
review requires.
```

---

## 7. Is This Reinforcement Learning?

The orchestrator is not a classical RL algorithm, but it
shares the core state-action feedback structure. The
distinction matters for how feedback loops are built.

### Structural Parallels

| RL Concept | Orchestrator Equivalent |
|---|---|
| Agent | Orchestrator control plane |
| Environment | Competition dataset + Zindi platform |
| State | `SKILL_STATE.json` + `challenge_config.json` |
| Action | Running a skill, promoting a branch, engineering a variant |
| Reward signal | OOF improvement, gate pass/fail, public LB delta |
| Policy | Phase map + gate conditions + three-lens rules |
| Episode | One competition lifecycle |

### Where It Differs

**Delayed and noisy rewards.** Public LB scores are
time-lagged, budget-constrained, and cover only 20–30%
of test data. OOF scores are faster proxy rewards but
statistically imperfect.

**Engineered policy.** Gate conditions, CV decision trees,
and phase maps are hand-designed, not learned via gradient
descent.

**No value function.** The orchestrator makes greedy local
decisions without modelling long-term multi-step
consequences.

### Feedback Loop Mechanics

Two explicit mechanisms approximate RL behaviour without
gradient updates:

**Cross-competition experience replay.** After every
competition close, CV strategy choices, feature types,
model architectures, and OOF-to-LB deltas are recorded
in a competition history log. `skill_18` and `skill_20`
read this log as prior knowledge for the next competition.

Cross-competition history log — minimum schema:

```
Location: competition_history/history_log.jsonl
Format:   One JSON object per line, one entry per
          competition close

Fields per entry:
{
  "competition_id": "",
  "task_type": "",
  "metric": "",
  "metric_direction": "maximize | minimize",
  "cv_strategy_type": "",
  "cv_strategy_override": false,
  "cv_strategy_override_rationale": null,
  "anchor_oof_score": 0.0,
  "best_promoted_oof_score": 0.0,
  "best_public_lb_score": 0.0,
  "oof_to_lb_delta": 0.0,
  "feature_types_used": [],
  "pseudo_label_ran": false,
  "final_rank": null,
  "gate_thresholds": {
    "shap_leak_threshold": 3.0,
    "variance_gate_threshold": 0.01,
    "gate_margin": 0.001
  },
  "competition_close_date": ""
}
```

**Bayesian threshold evolution.** After several competitions,
`shap_leak_threshold`, `variance_gate_threshold`, and
`gate_margin` are reviewed against historical OOF-to-LB
correlation data and updated in config. This is threshold
updating based on observed outcomes, not gradient
optimisation.

---

## 8. Definition of Done — Master Checklist

### Config Completeness

```
[ ] All Phase 1 fingerprint fields present and non-null
[ ] metric_direction present and set
[ ] use_probabilities present and set
[ ] cv_strategy block with type, n_splits, shuffle,
    random_state, group_col, stratify_col,
    and selection_reason
[ ] shap_leak_threshold set (default 3.0)
[ ] variance_gate_threshold set
[ ] gate_margin set (default 0.001)
[ ] submission_budget.total, .daily, .used all present
[ ] reproducibility.seed present and set
[ ] target_domain_bounds present (null allowed for
    classification)
[ ] community_signals array present
[ ] policy_filters array present
[ ] file_hashes match current raw data files
[ ] spatial_signal.group_col populated if spatial signal
    present and group_signal absent
```

---

### Per-Skill Completion Criteria

**`skill_00`**
```
[ ] Discussion board polling active
[ ] Leaderboard polling active
[ ] Data patch detection halts pipeline and surfaces
    human decision gate — [R] restart or [A] abort
[ ] skill_02 re-intake NOT triggered automatically —
    only after human [R] confirmation
[ ] data_patch_detected written to SKILL_STATE.json
    on detection before any other action
[ ] OOF-to-LB delta recorded after every skill_16
    submission
```

**`skill_01`**
```
[ ] MD5 hashes locked for all raw data files
[ ] Hashes written to challenge_config.json under
    file_hashes
```

**`skill_02`**
```
[ ] All fingerprint fields written to
    challenge_config.json
[ ] task_type, metric, target_col confirmed and non-null
[ ] metric_direction written to challenge_config.json
    maximize: AUC, F1, Accuracy
    minimize: RMSE, MAE, logloss
[ ] use_probabilities written to challenge_config.json
    True for probability submission competitions
    False for hard-label classification competitions
    (EY Frogs pattern — classification with 0/1 labels)
[ ] temporal_signal, group_signal, spatial_signal all
    evaluated
[ ] spatial_signal.group_col populated if spatial signal
    detected
[ ] target_domain_bounds written if
    task_type == regression
[ ] reproducibility.seed written
[ ] submission_budget fields written
```

**`skill_03`**
```
[ ] policy_writer() runs in Phase 1 —
    writes reports/feature_policy.json
[ ] policy_gate() runs as first action of Phase 2A —
    enforces blocked columns
[ ] No dataset-specific strings in either function
[ ] Two functions independently testable
[ ] Violation halts pipeline with written state entry
```

**`skill_04`**
```
[ ] Missingness correlation pass completed
[ ] mnar_columns and mcar_columns written to
    SKILL_STATE.json
[ ] Outlier columns flagged
[ ] Target skew computed and written
[ ] target_std computed and written to
    SKILL_STATE.json["eda"]["target_std"]
[ ] group_structure_confirmed and
    temporal_index_confirmed evaluated
[ ] Writes to SKILL_STATE.json only —
    never to challenge_config.json
```

**`skill_05`**
```
[ ] Full decision tree executed:
    temporal → group/spatial → stratified → standard
[ ] skill_04 EDA outputs confirmed in SKILL_STATE.json
    before skill_05 reads them
[ ] Spatial GroupKFold uses
    spatial_signal.group_col when group_signal absent
[ ] All column names read from config —
    no string literals
[ ] Selection reason written to challenge_config.json
[ ] CV object is the only CV object in the pipeline
[ ] random_state reads from
    config["reproducibility"]["seed"]
[ ] Config locks after this write completes
```

**`skill_06`**
```
[ ] MNAR indicator columns created before any fill
[ ] MCAR columns filled with median/mode
[ ] Constant columns dropped
[ ] Imputation order enforced — indicator first,
    fill second
```

**`skill_07`**
```
[ ] All variants use CV strategy from config
[ ] Target-dependent group aggregations (target col)
    follow two-mode contract: fold-restricted during CV,
    full-train during inference
[ ] Structural group aggregations (non-target cols)
    computed on full dataset — no two-mode restriction
[ ] Spatial lag of target follows two-mode contract:
    fold-restricted during CV, full-train during inference
[ ] Structural spatial features (Haversine, nearest-
    neighbour) computed on full dataset at any time
[ ] No target-dependent feature is missing from the
    inference feature matrix
[ ] Sidecar recommendations consumed if present,
    skipped if absent (non-blocking)
[ ] All OOF outputs tagged with cv_strategy_id
[ ] Config seed used for all training
```

**`skill_08`**
```
[ ] CV strategy read from config —
    not defined internally
[ ] Config seed used
[ ] Anchor OOF score written with cv_strategy_id tag
[ ] Anchor branch name and model config written to
    SKILL_STATE.json
[ ] If operator selected [C] at Gate 1:
    anchor_oof_score_challenged written to SKILL_STATE
    anchor_challenge block written with modification
      description, both OOF scores, rationale, timestamp
    No write to challenge_config.json under any condition
```

**`skill_09`**
```
[ ] Uses identical CV folds as skill_08
[ ] Classification tasks only
[ ] Calibrated OOF predictions written with
    cv_strategy_id tag
```

**`skill_10`**
```
[ ] SHAP computed per-fold on OOF predictions only
[ ] Aggregated across folds before threshold comparison
[ ] Full-train SHAP absent from skill body
[ ] If feature count < 2: relative SHAP ratio audit
    skipped, shap_audit_skipped_reason written to state,
    branch evaluated on fold variance gate alone —
    NOT automatically promoted
[ ] leaked_features written for every branch
[ ] Branches with non-empty leaked_features blocked
    from promotion
```

**`skill_11`**
```
[ ] All five promotion conditions checked
[ ] Gate condition 2 uses effective_variance_threshold:
    regression: config["variance_gate_threshold"] * (target_std ** 2)
    classification: config["variance_gate_threshold"] raw
[ ] Gate condition 3 uses effective_gate_margin:
    regression: config["gate_margin"] * target_std
    classification: config["gate_margin"] raw
[ ] target_std read from SKILL_STATE["eda"]["target_std"]
    — written by skill_04 in Phase 1
[ ] Gate condition 3 reads metric_direction from config
[ ] Gate condition 3 baseline uses safe state lookup:
    SKILL_STATE.get("pseudo_label_result", {})
    .get("retraining_required", False)
    — prevents KeyError on first pass before skill_21
    has ever run
[ ] Gate condition 3 challenge flag uses safe lookup:
    SKILL_STATE.get("anchor_challenge", {})
    .get("active", False)
    — prevents KeyError when no [C] challenge was used
[ ] Gate condition 3 baseline: anchor_oof_score_augmented
    when retraining_required == true — takes precedence
    over anchor_challenge because training set changed
[ ] If retraining_required == false and
    anchor_challenge.active == true: baseline is
    anchor_oof_score_challenged
[ ] Maximize metrics: improvement means score went up
[ ] Minimize metrics: improvement means score went down
[ ] No symmetric gate_margin applied to minimize metrics
[ ] Reads cv_strategy_override from SKILL_STATE
    if present
[ ] All gate comparisons use override OOF scores
    when override is active
[ ] Gate failure produces written diagnosis
[ ] Gate failure triggers skill_20 on-demand run
[ ] Human Gate 2 approval checked before candidate
    pool entry
```

**`skill_12`**
```
[ ] Fold scores and variance written
[ ] fold_score_variance computed with ddof=1
    (unbiased sample variance, n-1 denominator)
[ ] For regression: variance interpreted against
    effective_variance_threshold = variance_gate_threshold
    * (SKILL_STATE["eda"]["target_std"] ** 2)
[ ] For classification: raw variance_gate_threshold used
[ ] OOF-to-LB delta recorded when available
[ ] Recommended threshold written
```

**`skill_13`**
```
[ ] Human Gate 3 approved before running
[ ] Only fuses skill_11-passed candidates with Gate 2
    approval
[ ] OOF correlation check on all candidate pairs
[ ] Pearson correlation used for classification tasks
[ ] Spearman rank correlation used for regression tasks
[ ] No two candidates with correlation > 0.95 blended
[ ] All candidates have seeds set
[ ] Fusion strategy written to SKILL_STATE.json
[ ] Uses most recent OOF arrays only —
    never stale pre-pseudo-label arrays
```

**`skill_14`**
```
[ ] Human Gate 4 approved before running
[ ] Validation logic branches on task_type AND
    use_probabilities from config
[ ] Classification + use_probabilities True:
    probability range (0,1), decimal depth check,
    raw probability check
[ ] Classification + use_probabilities False:
    value equality check (val == 0 or val == 1),
    0.0/1.0 pass, any other value is hard failure
[ ] Regression: domain bounds check,
    no NaN/inf, distribution sanity check
[ ] No probability checks applied to regression outputs
    or hard-label classification outputs
[ ] ID column matches test set exactly
[ ] Row count matches test set exactly
[ ] No duplicate IDs
[ ] File format matches competition submission schema
[ ] Seed confirmed
```

**`skill_15`**
```
[ ] CV strategy selection event logged
[ ] Every phase transition logged
[ ] Every gate pass and failure logged with timestamp
[ ] Every human gate approval logged
[ ] Config lock event logged
```

**`skill_16`**
```
[ ] Submission budget checked — hard abort at zero
[ ] Single remaining submission triggers human
    confirmation
[ ] Submission validated by skill_14 before API call
[ ] Post-submission LB score recorded and passed to
    skill_00
```

**`skill_17`**
```
[ ] Final submission selection documented with reasoning
[ ] CV strategy recorded in governance report
[ ] All Human Gate approvals referenced by timestamp
[ ] Gate 5 selection recorded
```

**`skill_18`**
```
[ ] Runs after Phase 1 completes
[ ] Domain literature and metric optimisation evidence
    written to SKILL_STATE.json
[ ] On-demand run triggered when skill_20 raises
    unresolved hypothesis
[ ] No blocking behaviour if absent
```

**`skill_19`**
```
[ ] Runs after Phase 2A completes
[ ] Code patterns written specific to CV strategy and
    data structure found in Phase 1
[ ] No blocking behaviour if absent
```

**`skill_20`**
```
[ ] Runs after skill_08 and skill_10 complete
[ ] Hypotheses validated against anchor OOF and
    SHAP values
[ ] Every hypothesis resolved as accepted or rejected
    in SKILL_STATE.json
[ ] On-demand run triggered on every skill_11_gate
    rejection
[ ] No blocking behaviour if absent
```

**`skill_21`**
```
[ ] All six guard conditions checked before running
[ ] Guard Condition 1: does not run when
    task_type != "classification"
[ ] Guard Condition 2: does not run when
    cv_strategy == TimeSeriesSplit
[ ] Guard Condition 3: does not run when
    leaked_features non-empty
[ ] Guard Condition 4: does not run when
    fold_score_variance >= effective_variance_threshold
    (equivalent to raw variance_gate_threshold in classification)
[ ] Guard Condition 5: does not run without calibrated
    probabilities from skill_09
[ ] Guard Condition 6: does not run if confidence
    threshold not met
[ ] Uses calibrated probabilities from skill_09
[ ] Config seed used
[ ] Full pseudo_label_result schema written to
    SKILL_STATE.json on every run (ran,
    n_pseudo_labels_added, retraining_required,
    guard_conditions_met, guard_failure_reason,
    execution_failure_reason,
    guard_condition_flags with all six gc fields)
[ ] Retraining loop triggered when
    retraining_required == true
[ ] Pseudo-label CV fold assignment contract enforced:
    rows N_train onward assigned to training split
    of every fold — never to validation folds
[ ] OOF evaluation indices identical to pre-augmented
    Phase 1 split layout
[ ] Anchor model retrained on augmented dataset —
    anchor_oof_score_augmented written to SKILL_STATE.json
[ ] Original anchor_oof_score preserved unchanged
[ ] Augmented variant OOF namespace contract enforced:
    all retraining loop OOF arrays written to
    branch_{name}_oof_augmented keys exclusively —
    original branch_{name}_oof keys never overwritten
[ ] Hard error triggered if retraining loop attempts
    to write to any non-augmented OOF key
[ ] Retrained branches compared against
    anchor_oof_score_augmented — never original anchor
[ ] Retrained branches pass through skill_10 and
    skill_11 before fusion
[ ] Human Gate 2 re-approval obtained for all
    retrained branches
[ ] OOF outputs from retrained branches tagged with
    cv_strategy_id
[ ] Rollback path executed if zero retrained branches
    pass skill_11: only _augmented keys cleared,
    original branch_{name}_oof arrays intact,
    original candidate pool restored, proceeds to skill_13
[ ] Fusion uses most recent OOF arrays only
```

**`skill_22`**
```
[ ] All reproducibility checklist items pass (R1–R4)
[ ] All five human gate approvals recorded with
    timestamps
[ ] CV contract verified — single strategy,
    all outputs tagged
[ ] Environment lock file verified
[ ] No custom packages confirmed via import scan
[ ] If skill_21 ran with retraining_required == true:
        Guard Condition 1 (classification-only) confirmed
        Pseudo-label fold contract verified: augmented
          rows in train splits only, OOF indices unchanged
        guard_condition_flags present in pseudo_label_result
          with all six gc fields populated as Booleans
        anchor_oof_score_augmented present in SKILL_STATE
        Retrained branch OOF scores present and tagged
        Retrained branches gated against augmented anchor
        skill_10 and skill_11 pass confirmed on
          retrained branches
        If rollback occurred: all _augmented keys cleared,
          original branch_{name}_oof keys verified intact,
          execution_failure_reason written,
          original pool used, confirmed in state
[ ] If CV strategy override used at Gate 1:
        cv_strategy_override block present in
        SKILL_STATE.json with rationale and timestamp
        Override rationale present and non-empty
        Override recorded in governance report
          and history log
[ ] If anchor challenge used at Gate 1 ([C] selected):
        anchor_challenge block present in SKILL_STATE.json
        Both original and challenged OOF scores recorded
        anchor_challenge.active reflects operator selection
        Modification description and rationale non-empty
        No challenge writes present in challenge_config.json
[ ] Competition history log entry written in correct
    schema to competition_history/history_log.jsonl
[ ] All required history log fields populated before
    sign-off
[ ] cv_strategy_override and rationale recorded in
    history log if Gate 1 override occurred
[ ] Cross-competition history log updated
[ ] Pipeline replayable from config and state alone
```

---

### Per-Phase Gate Criteria

**Phase 1 → Phase 2A:**
```
[ ] challenge_config.json complete and schema-valid
[ ] task_type, metric, target_col non-null
[ ] metric_direction written and set
[ ] use_probabilities written and set
[ ] target_domain_bounds written if regression
[ ] File hashes locked
[ ] policy_filters written
[ ] reports/feature_policy.json present, non-empty,
    and valid JSON
[ ] feature_policy.json contains required keys:
    allowed_features, blocked_features, block_reasons
[ ] blocked_features contains at minimum all columns
    listed in challenge_config["policy_filters"]
[ ] skill_04 EDA outputs in SKILL_STATE.json
    — verified BEFORE skill_05 runs
[ ] If task_type == regression:
    target_std present in SKILL_STATE["eda"]["target_std"]
    — required for effective_gate_margin and
    effective_variance_threshold normalisation in skill_11
[ ] skill_05 cv_strategy written with selection_reason
    — only valid after skill_04 outputs confirmed
[ ] spatial_signal.group_col populated if needed
[ ] challenge_config.json temporal lock active
[ ] seed written to config
```

**Phase 2A → Phase 2B:**
```
[ ] policy_gate() passed — all blocked columns absent
[ ] skill_06 cleaning complete
[ ] MNAR indicators generated before fills
[ ] MCAR columns filled
[ ] Constant columns dropped
[ ] Cleaning outputs in SKILL_STATE.json
```

**Phase 2B → Phase 3A:**
```
[ ] Human Gate 1 approved
[ ] Anchor OOF score present and cv_strategy_id tagged
[ ] At least one variant OOF score present and tagged
[ ] No internal CV objects in any skill
```

**Phase 3A → Phase 3B:**
```
[ ] SHAP audit complete for all branches
[ ] leaked_features written for all branches
[ ] Fold score variance written for all branches
[ ] Calibration complete for classification tasks
[ ] All OOF outputs carry cv_strategy_id tags
```

**Phase 3B → Phase 4:**
```
[ ] At least one branch promoted through skill_11
[ ] Human Gate 2 approved for all promoted branches
[ ] If skill_21 ran with retraining_required == true:
        guard_condition_flags: all six gc fields present
          and Boolean — gc1 through gc6 confirmed
        Pseudo-label fold contract verified: augmented
          rows in train splits only, OOF indices unchanged
        anchor_oof_score_augmented present in
          SKILL_STATE.json
        New OOF scores tagged with cv_strategy_id
        Retrained branch scores gated against
          anchor_oof_score_augmented
        skill_10 SHAP audit passed on retrained branches
        skill_11 gate passed on retrained branches
        Human Gate 2 re-approved for retrained branches
        If rollback: all _augmented keys cleared,
          original branch_{name}_oof keys intact,
          execution_failure_reason written, original
          pool used for fusion
[ ] skill_13 uses most recent OOF arrays only
[ ] Human Gate 3 approved
[ ] Fusion diversity check complete
[ ] Final submission candidate identified
```

**Phase 4 → Done:**
```
[ ] skill_22 reproducibility checklist fully passes
[ ] Human Gates 4 and 5 approved and recorded
[ ] Submission budget not exceeded
[ ] Governance report written
[ ] Cross-competition history log updated
```

---

### OOF Contract Compliance

```
[ ] Single CV strategy object in challenge_config.json
[ ] No skill defines a CV object internally
[ ] All OOF scores tagged with cv_strategy_id
[ ] Orchestrator validates tags before score passing
[ ] skill_22 verifies full contract at sign-off
```

---

### Research Sidecar Trigger Compliance

```
[ ] skill_00 running from competition intake to close
[ ] skill_18 first run after Phase 1
[ ] skill_19 run after Phase 2A
[ ] skill_20 first run after skill_08 and skill_10
    complete
[ ] All skill_20 hypotheses resolved in SKILL_STATE.json
[ ] All on-demand sidecar runs logged
[ ] No sidecar failure halts main pipeline
```

---

### Zindi Compliance

```
[ ] Seed set and in challenge_config.json at intake
[ ] Seed logged in every OOF output
[ ] All models trained with config seed —
    no local overrides
[ ] Raw probabilities in classification submissions
    when use_probabilities == True —
    no rounding, no thresholding
[ ] Hard 0/1 labels in classification submissions
    when use_probabilities == False
[ ] target_domain_bounds enforced for regression
    submissions
[ ] No AutoML tools in any skill —
    preflight static scan confirms this
[ ] No custom packages in any skill body
[ ] 2 final submissions selected before close (Gate 5)
[ ] requirements.txt committed and verified
[ ] Submission reproducible from config and state alone
[ ] Code review package prepared
[ ] If top 10: code ready for 48-hour submission window
```

---

### Human Gate Compliance

```
[ ] Gate 1 approved before variant generation starts
[ ] If CV strategy override used at Gate 1:
    cv_strategy_override block present in
    SKILL_STATE.json with rationale and timestamp
[ ] Override written to SKILL_STATE only —
    challenge_config.json unchanged
[ ] Gate 2 approved for every promoted branch
[ ] Gate 2 re-approved for every retrained branch
    from skill_21
[ ] Gate 3 approved before skill_13 runs
[ ] Gate 4 approved before skill_14 runs
[ ] Gate 5 completed before competition close
[ ] No gate key written by any skill or automated
    process
[ ] Every gate approval recorded in reports/ with
    timestamp
```

---

### Architecture Integrity

```
[ ] No skill imports from another skill directly
[ ] No hardcoded competition-specific values in any
    skill
[ ] Every skill reads context from
    challenge_config.json
[ ] Every skill writes outputs to SKILL_STATE.json
[ ] Orchestrator is the only entity reading both files
[ ] Phase dependency chain enforced
[ ] challenge_config.json read-only after Phase 1 lock
[ ] Post-lock write by non-skill_00 skill is hard error
[ ] skill_00 community_signals writes are the only
    permitted post-lock writes
[ ] Sidecar failures do not halt the main pipeline
[ ] Preflight detects INIT vs ENFORCE mode
    automatically
[ ] INIT mode allows full Phase 1 sequence only
[ ] ENFORCE mode runs full check suite
[ ] Preflight confirms before any skill executes
```

---

### Scalability and Feedback Loop Integrity

```
[ ] New skill requires only adding module to
    zindian/skills/ — no orchestrator code changes
[ ] Phase map configurable via challenge_config.json
[ ] All gate thresholds configurable —
    no magic numbers in skill code
[ ] Every gate failure produces written diagnosis
[ ] skill_20 on-demand runs triggered automatically
[ ] Cross-competition history log updated after every
    close
[ ] Gate thresholds reviewable against historical
    OOF-to-LB data
```

---

*End of Source of Truth Document v2.0.1-Canonical*
*Patched from v2.0: 2 label fixes — Section 4 gate schema reference string updated to v2.0.1; 
Section 8 skill_21 DoD Guard Condition 4 synchronized to effective_variance_threshold.*
*Status: Fully signed off — zero open anomalies remaining.*