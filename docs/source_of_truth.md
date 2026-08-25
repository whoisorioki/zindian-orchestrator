# Zindian Orchestrator — Source of Truth Document

**Version:** v2.6
**Status:** CURRENT
**Scope:** Zindi tabular competitions (standard, spatial, temporal, grouped)
**Last updated:** August 2026 (v2.6: Pairwise Mutual Information S6, Consolidated S11 root writes, Preflight MT-OOF, R5 telemetry.aggregate, and logging deduplication)

---

## Table of Contents

1. Assumptions
2. Core Architectural Principles
3. Preflight Validation
4. Phase Architecture
5. Research Sidecar
6. Reproducibility Contract

---

## Documentation Map

| Document | Owns |
|---|---|
| **This file (source_of_truth.md)** | Architecture contracts, schemas, phase specs, gate logic, reproducibility |
| [orchestrator_overview.md](orchestrator_overview.md) | Non-technical overview, phase summaries for new readers |
| [AGENTS.md](../AGENTS.md) | Agent implementation guide, safe access patterns, known live risks |
| [quick_start.md](quick_start.md) | CLI bootstrap walkthrough, competition setup |
| [cli_integration_guide.md](cli_integration_guide.md) | All 21 CLI commands reference |
| [ledger_architecture.md](ledger_architecture.md) | DuckDB experiment tracking schema |
| [troubleshooting_guide.md](troubleshooting_guide.md) | Runtime errors, CI fixes |
| [document_map.md](document_map.md) | Consolidated structure maps, cross-document overlaps, and ownership matrix |

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

**A6 — `SKILL_STATE.json` is the single source of truth for execution state.**
No skill holds internal state between runs. All execution state is written to and read from `SKILL_STATE.json`.

*State Hygiene & Boundary Rule:* `SKILL_STATE.json` holds only what downstream skills or gates need to make a decision (booleans, counts, OOF scores, short column-name lists, phase/gate flags, file hashes, and small scalar summaries). Any diagnostic artifact for human review — per-feature, per-band, or per-row dictionaries whose size scales with feature/row count — belongs in `reports/` as JSON or Markdown, with at most a short pointer or single derived flag left in state.

Examples of the standard boundary:
- `skill_03`: `checks[]` + `policy{}` → `reports/audits/feature_policy.json` + `reports/audits/legality_report.md`. State gets `legality_status`, `feature_policy_written`, `last_legality_checked`.
- `skill_04`: heavy per-band/per-feature diagnostic dicts (`band_summary_stats`, `seasonal_amplitude`, `temporal_trends`, `target_correlation_per_feature`, `class_separability_index`) → `reports/diagnostics/eda_report.json`. State gets `temporal_index_confirmed`, `group_structure_confirmed`.
- `skill_10`: heavy SHAP ranking & corr pairs → `reports/audits/shap_analysis.json` + `reports/audits/shap_summary.md`. State gets `shap_top_features`, `shap_feature_count`, `pruning_delta_f1`, `pruning_pass`.
- `skill_15`: phase summaries → `reports/summaries/phase_*.md` + `reports/summaries/<phase>_summary.json`; session events → `reports/sessions/startup_*.jsonl`. State gets `last_reported`.

**Categorized report layout (`reports/`):** artifacts are written to categorized
subdirectories, and every consumer reads from the same categorized path (root-level
report duplicates are deprecated):
- `reports/audits/` — feature policy, legality report, SHAP leak audit (`skill_03`, `skill_10`), governance selections (`skill_17`), reproducibility audit (`skill_22`).
- `reports/diagnostics/` — EDA report/summary (`skill_04`), literature & domain hypotheses (`skill_18`, `skill_20`), and `diagnostics/predictions/` for OOF/test pseudo-label CSVs (`skill_21`).
- `reports/summaries/` — phase summary Markdown + JSON (`skill_15`).
- `reports/sessions/` — session-scoped startup event logs (`skill_15`).

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

**A9 — The research sidecar is non-blocking at every consumption point.**
Sidecar skills (`skill_18`, `skill_19`, `skill_20`) write recommendations to `SKILL_STATE.json` under `sidecar_recommendations`.
Skills that read sidecar output treat it as optional enrichment.
The correct pattern at every consumption point is:

```python
sidecar_recommendations = [
    rec for rec in SKILL_STATE.get("sidecar_recommendations", [])
    if isinstance(rec, dict) and rec.get("status") == "pending"
]
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

**A11 — Multi-target competitions are config-declared, never inferred.**
A competition is multi-target if and only if
`challenge_config.json["target_config"]["targets"]` contains more than one
entry. No skill infers multi-target status from data shape, column count,
or any heuristic. `skill_02` writes `target_config` during Phase 1 from
the competition's stated submission format — never guessed.
This preserves A5 (no hardcoded values) and A6 (`SKILL_STATE.json` is sole
execution-state authority) — `target_config` lives in `challenge_config.json`
as a *structural* declaration, consistent with Principle 2's config boundary.
Backward-compatibility rule: if `target_config` is absent or contains
exactly one entry, every skill reads `task_type`, `metric`, `target_col`,
etc. from the top-level config fields exactly as in v2.2 — unchanged code
path. Single-target competitions are byte-for-byte unaffected.

**A12 — Pseudo-label recombination policy is mandatory for mixed-task multi-target competitions.**
Whenever `target_config` has more than one target AND at least one target is classification, `target_config` must include `pseudo_label_recombination_policy`. Permitted values are exclusively `"freeze_unaugmented_targets_at_original"` and `"block_composite_until_all_targets_augmented_or_none"`. This field is absent for single-target competitions and for multi-target competitions where all targets are regression.

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
                    submission_budget, target_domain_bounds,
                    target_config (A11), file_manifest, plugin_config
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

#### OOF Record Schema

Every OOF-generating skill must write its output in this form in `SKILL_STATE.json` under `branch_{branch_name}_oof` (or `branch_{branch_name}_oof_augmented` during pseudo-label retraining):

```json
{
  "scores": [0.123, 0.456, ...],
  "cv_strategy_id": "stratified",
  "seed": 42,
  "branch_name": "variant-01",
  "model_config": { ... },
  "secondary_metrics": {
    "mae": 0.123,
    "mape": 0.045,
    "r2": 0.789
  }
}
```

When a training skill computes fold-aware Nadeau-Bengio corrections or
inverse-variance weighting, `model_config` may also include `fold_sizes` as
an ordered list of `[n_train, n_val]` pairs for each fold. This is the schema
home for the fold-size threading used by `skill_11` and `skill_12`.

#### Secondary Metrics Block
To avoid schema bloat in the root of `SKILL_STATE.json` while maintaining versatility across diverse Zindi regression challenges, all candidate and anchor OOF records must contain a nested `secondary_metrics` object containing:
- `mae`: Mean Absolute Error (regression tasks only)
- `mape`: Mean Absolute Percentage Error (regression tasks only).
  Computed exclusively over rows where the ground-truth target
  `y_true != 0`. Rows where `y_true == 0` are excluded entirely from
  both the numerator and denominator of the MAPE computation.
  When all rows in the validation fold have `y_true == 0`,
  `mape` is set to `null` with reason `"all_targets_zero"` rather than
  `0.0` or infinity. Zero or infinity would silently corrupt the
  `secondary_metrics` block and any downstream diagnostic that reads it.

- `r2`: Coefficient of determination (regression tasks only)

> **Target Met (S2-adjacent):** `zero_fraction`: Target sparsity diagnostic (fraction of ground-truth targets where `y_true == 0`).
> **Target Met (S2):** `mase` (Mean Absolute Scaled Error) diagnostic. Added to `secondary_metrics` ONLY when `config["temporal_signal"]["present"] == True`. MASE is omitted for non-temporal competitions.

For classification tasks, the `secondary_metrics` field may be omitted or set to `null`. For multi-target competitions, secondary_metrics is computed and stored per-target inside that target's own OOF record — there is no separate composite secondary_metrics block.

Multi-target variant of the same schema (only when target_config.targets
has more than one entry):

```json
{
  "scores": [0.123, 0.456, ...],
  "cv_strategy_id": "group_kfold_tournament",
  "seed": 42,
  "branch_name": "variant-01",
  "target_name": "total_goals",
  "model_config": { ... },
  "secondary_metrics": { "mae": 0.123, "mape": 0.045, "r2": 0.789 }
  // zero_fraction and mase: Target Met (S2 / S2-adjacent)
}
```

target_name is required when target_config has more than one target,
absent or null otherwise. Storage key becomes
branch_{branch_name}_{target_name}_oof instead of branch_{branch_name}_oof
only when multi-target is active. Single-target keys unchanged.

#### Regression Target Transformation Lifecycle

When `config["task_type"] == "regression"`, the training and
prediction loops must route target variables dynamically through
an evaluation matrix determined by `config["metric"]`:

1. **Target Pre-Transformation Matrix:**
   * If `config["metric"] == "rmsle"`: Transform targets to
     log-space: `y_trans = ln(y + 1)`.
   * If `config["metric"] == "root_mean_squared_error"` or
     `"mean_absolute_error"`: Maintain identity scale: `y_trans = y`.

2. **Prediction Domain Mapping Matrix:**
   * If `config["metric"] == "rmsle"`: Clip log-space predictions
     to eliminate negative values (`y_pred_log = max(0, y_pred_log)`),
     then execute inverse exponential mapping to restore original
     scale: `y_pred = exp(y_pred_log) - 1`.
   * If `config["metric"] == "root_mean_squared_error"` or
     `"mean_absolute_error"`: Apply domain clipping directly to
     the raw output using bounds specified in
     `config["target_domain_bounds"]`:
     `y_pred = max(min_bound, min(max_bound, y_pred))`.

3. **Score Computation:**
   * RMSLE is computed in original space:
     `sqrt( (1/N) * sum( (ln(y_i + 1) - ln(y_hat_i + 1))^2 ) )`
   * RMSE (root_mean_squared_error) and MAE (mean_absolute_error) are computed in original space with standard scikit-learn functions, operating on the domain-clipped predictions.

Breaking this contract in any skill invalidates all cross-branch score comparisons. A contract violation is a hard halt — not a warning.

For multi-target competitions, this lifecycle applies independently per target, using that target's own metric and `target_domain_bounds` from `target_config.targets[i]` — there is no shared transformation lifecycle. Each target is pre-transformed, clipped, and scored entirely on its own terms before composite aggregation (below).

#### Composite Score Computation (multi-target only)

```python
For each target_spec in target_config["targets"]:
    raw_score = oof_score for this target (computed via the existing per-task-type pipeline above)

    if target_spec["task_type"] == "regression":
        target_std = SKILL_STATE["eda"][f"{target_spec['name']}_std"]
        normalized_distance = abs(raw_score) / target_std
        if target_spec["metric"] == "rmsle":
            normalized_distance = raw_score  # directly (no division)

    if target_spec["task_type"] == "classification":
        normalized_distance = 1.0 - raw_score

    weighted_distances.append(normalized_distance * effective_weight)

composite_score = sum(weighted_distances) / total_weight
```

> **Target Met (Item 4):** Inverse-variance effective weighting. When `use_inverse_variance_weighting: True` is configured, target weight `w_k` is scaled inversely by target fold variance:
> `w_k_eff = w_k / (sigma_{k,NB}^2 + epsilon)`
> where `sigma_{k,NB}^2` is the Nadeau-Bengio corrected variance for target k's fold scores, `epsilon = 1e-8` prevents division by zero, and the NB factor is fold-size aware when `fold_sizes` are available:
> `sigma_{k,NB}^2 = Var_sample(ddof=1) * gamma_bar_k`
> `gamma_bar_k = min((1/K) * sum(n_{val,i} / n_{train,i}), 1.0)`
> If `fold_sizes` are absent, the fallback remains the equal-fold geometry correction `gamma_bar_k = 1/K + 1/(K-1)`.
> Epsilon-regime note: with the competitions currently represented in this workspace, no live competition state file is available to prove or disprove a near-zero-variance target from data. Operationally, the `epsilon=1e-8` safeguard only becomes relevant for degenerate near-constant fold-score sequences; the helper remains finite in that regime and is covered by `tests/test_scale_invariance.py::test_skill_12_near_zero_variance_stays_finite`.
> **IMPORTANT NOTE:** This is NOT Kendall & Gal uncertainty weighting (which requires joint differentiable loss training that this decision-tree pipeline does not use).

`composite_direction` is fixed as `"minimize_composite_distance"` — every term is already a "lower is better" distance, regardless of how many targets individually maximize or minimize.

#### Composite variance threshold (multi-target only)

```python
effective_target_std = sqrt(
    sum(w_i * sigma_i^2 for i in regression_targets)
    / sum(w_i for i in regression_targets)
)
effective_variance_threshold = (
    config["variance_gate_threshold"] * (effective_target_std ** 2)
)
```

Dividing by the sum of regression weights (rather than treating them as summing to 1.0) removes the suppression artifact when classification targets are present. Verified against single regression target at weight 0.60 (reduces to `target_std` unmodified) and two regression targets at 0.30/0.30 (produces proper weighted RMS independent of classification weight).


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

**Output:** `reports/audits/preflight/preflight_INIT_{timestamp}.json`

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
      (name, slug, task_type, metric, metric_direction,
      use_probabilities, drift_threshold, allowed_external_data,
      automl_permitted, team_allowed, submission_budget, data_shape,
      phase_skill_map, reproducibility, shap_leak_threshold,
      variance_gate_threshold, gate_margin, cv_strategy,
      community_signals, policy_filters, file_hashes)
    cv_strategy block present with all required subfields
    shap_leak_threshold, variance_gate_threshold,
      gate_margin, drift_threshold set
    automl_permitted == false enforced (Zindi compliance)
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
    OOF key pattern validation:
        Pattern: ^branch_(?P<branch>[a-zA-Z0-9\-]+)(?:_(?P<target>[a-zA-Z0-9_]+))?_oof$
        Single-target keys match with target group as None
        Multi-target keys match with both branch and target groups populated
        OOF tags must be validated by matching keys against the literal
        pattern branch_{branch_name}_{target_name}_oof using an optional
        named group for the target, maintaining legacy compatibility
    Completeness check (multi-target only):
        Preflight must confirm completeness — beyond pattern matching,
        it must confirm that every active branch has exactly N OOF records,
        where N = len(target_config["targets"]). A branch missing one
        target's OOF entirely must fail preflight.
        For every active branch_name: count of pattern matches with
        branch == branch_name must equal len(target_config["targets"])
        when target_config has more than one entry
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

**Output:** `reports/audits/preflight/preflight_ENFORCE_{timestamp}.json`

**Confirmation written to:**
`SKILL_STATE.json["preflight_confirmed"]`

---

> The full interactive session prompt layout is rendered at runtime by the orchestrator. See `scripts/preflight_enforce.py` for the exact template.

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
  "name": "",
  "slug": "",
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
    "group_col": null,
    "spatial_buffer_km": null
  },
  "missingness_level": "low | moderate | high",
  "allowed_external_data": false,
  "automl_permitted": false,
  "team_allowed": false,
  "drift_threshold": 0.05,
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
  },
  "infrastructure": {
    "hardware_type": "cpu | gpu | tpu",
    "region": "ke | us-east-1 | eu-central-1",
    "tdp_watts": 15.0,
    "pue": 1.0,
    "carbon_intensity_gco2_per_kwh": 494.0
  }
}
```

> `name` and `slug` are written by `skill_02` at intake. `infrastructure` is added by `skill_02` during the Phase 1 mutable window if absent. `allowed_external_data`, `automl_permitted`, `team_allowed`, and `drift_threshold` are required by preflight and must be present before ENFORCE mode runs. `competition_id` (used in some legacy references) maps to `slug` in the real config.

Additive fields for multi-target competitions (all optional/absent for single-target):

```json
{
  "target_config": {
    "targets": [
      {
        "name": "",
        "task_type": "regression | classification",
        "metric": "",
        "metric_direction": "maximize | minimize",
        "weight": 0.0,
        "target_domain_bounds": {"min": null, "max": null}
      }
    ],
    "composite_direction": "minimize_composite_distance",
    "pseudo_label_recombination_policy": "freeze_unaugmented_targets_at_original | block_composite_until_all_targets_augmented_or_none"
  },
  "file_manifest": { "train": "", "test": "" },
  "plugin_config": {}
}
```

**Mandatory field rule (A12):** Whenever `target_config` has more than one target AND at least one target is classification, `pseudo_label_recombination_policy` must be present. Permitted values are exclusively `"freeze_unaugmented_targets_at_original"` and `"block_composite_until_all_targets_augmented_or_none"`.

---

**`skill_03_legality` — Internal structural breakout:**

`skill_03` is implemented as two explicitly separated,
independently testable functions.

```
policy_writer():
    Reads : challenge_config.json,
            community_signals from skill_00
    Writes: reports/audits/feature_policy.json
    Fields: allowed_features, blocked_features,
            block_reasons
    Side effects: none — pure writer, no gating

policy_gate():
    Reads : reports/audits/feature_policy.json,
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

**`skill_04_eda` — Writes to `SKILL_STATE.json` (lean fields) and
`reports/diagnostics/eda_report.json` (heavy diagnostics):**

**[CORRECTION — v2.3]** The five per-band/per-feature diagnostic dicts
previously listed here as SKILL_STATE fields (`band_summary_stats`,
`seasonal_amplitude`, `temporal_trends`, `target_correlation_per_feature`,
`class_separability_index`) have been moved to `reports/diagnostics/eda_report.json`.
They do not belong in SKILL_STATE.json per rule A6-B — their size scales
with feature/band count. Only the lean boolean derived from them is kept
in state.

SKILL_STATE.json `eda` sub-block (lean fields only):

```json
{
  "eda": {
    "mnar_columns": [],
    "mcar_columns": [],
    "outlier_columns": [],
    "target_skew": 0.0,
    "target_std": 0.0,
    "group_structure_confirmed": false,
    "temporal_index_confirmed": false,
    "MAE_naive_baseline": 0.0
  }
}
```

`MAE_naive_baseline` is the mean absolute one-step naive forecast error
computed on the training target series. It is only meaningful for temporal regression
tasks — written as `0.0` for all non-temporal or non-regression competitions.
When a group column (`group_col`) is defined in the challenge configuration, the naive forecast
differences MUST be computed group-wise (partitioning the target series by the group column
and computing consecutive differences within each group separately, excluding boundary-crossings)
to prevent cross-group outlier deltas. Used as the denominator baseline for MASE (S2) once
that metric is implemented.

`reports/diagnostics/eda_report.json` holds the heavy diagnostics:

```json
{
  "data_quality": { "...": "full quality report" },
  "preprocessing_audit": { "...": "outlier_assessment, missingness, etc." },
  "band_summary_stats": {},
  "seasonal_amplitude": {},
  "temporal_trends": {},
  "target_correlation_per_feature": {},
  "class_separability_index": {}
}
```

The five band-aware diagnostics are computed dynamically from column
naming patterns (e.g., `VH_01`, `blue_12`) — no hardcoded band names
or competition-specific strings. When a dataset has no monthly composite
columns (no `BAND_MM` pattern), all five are empty dicts — no pipeline halt.

`target_std` is the standard deviation of the target column
computed on the full training set. It is used by `skill_11`
and `skill_12` to normalise `gate_margin` and
`variance_gate_threshold` for regression tasks where raw
thresholds are scale-sensitive. Written during Phase 1 before
the config lock closes.

**Implementation note — `temporal_index_confirmed` and
`group_structure_confirmed` derivation (v2.3):**

`temporal_index_confirmed` is set to `True` when either of the
following competition-agnostic signals is found:

1. The BAND_MM monthly composite pattern detection (which also
   populates `seasonal_amplitude` and `temporal_trends`) already
   confirmed temporal column structure — i.e., `seasonal_amplitude`
   or `temporal_trends` is non-empty after band detection completes.
2. Any feature column has a datetime64 dtype (`pd.api.types.is_datetime64_any_dtype`),
   OR is a string/object-dtype column (`pd.api.types.is_string_dtype`) that
   round-trips cleanly through `pd.to_datetime` on a 20-row sample AND exhibits
   sequential row monotonicity (`.is_monotonic_increasing` or `.is_monotonic_decreasing`).
   `is_string_dtype` is used rather than `is_object_dtype` because string columns
   can be Arrow string arrays (`dtype='str'`). Numeric (float/int) columns are
   explicitly excluded — `pd.to_datetime` interprets floats as nanosecond Unix timestamps.
   The monotonicity requirement prevents string ID columns containing date fragments
   (e.g., `sample_id` with `20200115_000` values) from triggering false positives.

No hardcoded column names are used. Detection relies solely on dtype inference,
monotonicity, and the existing BAND_MM pattern match.

`group_structure_confirmed` is set to `True` when any feature column
(excluding targets) satisfies all three of the following:
- `nunique > 1` (not constant)
- `nunique < len(df)` (not unique-per-row, i.e., not an ID column)
- `nunique / len(df) < 0.05` (at most 5% distinct values, meaning
  each value repeats across many rows — consistent with a grouping
  or categorical identifier column)

**Judgment call note:** The 0.05 cardinality ratio threshold was
introduced in v2.3 as a new heuristic. No existing convention for
group-cardinality detection was found elsewhere in the codebase
(`skill_02`, `skill_05`, `skill_07`) at the time this was written.
If a second convention is added elsewhere, both should be unified
to avoid two different definitions of "group structure" coexisting.

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

      > **Spatial CV Buffering (S7):** `skill_05_cv`'s `_apply_spatial_buffer` function is fully implemented: training set samples within `spatial_buffer_km` km of any validation fold sample are excluded from that fold's training split. `challenge_config_template.json`'s `spatial_signal` block includes `spatial_buffer_km: null`.
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
[ ] challenge_config.json matches v2.2 schema — all
    fields present and non-null where required
[ ] task_type, metric, target_col confirmed
[ ] metric_direction written and set
[ ] use_probabilities written and set
[ ] target_domain_bounds written if task_type == regression
[ ] file_hashes locked and written by skill_01
[ ] policy_filters written by skill_03 policy_writer()
[ ] reports/audits/feature_policy.json present, non-empty,
    and valid JSON
[ ] feature_policy.json contains required keys:
    allowed_data_sources, banned_transformations, lat_lon_permitted_as_feature
[ ] banned_transformations contains at minimum all columns
    listed in challenge_config["policy_filters"]
[ ] skill_04 EDA outputs present in SKILL_STATE.json
    — verified BEFORE skill_05 runs
[ ] If task_type == regression:
    target_std present in SKILL_STATE["eda"]["target_std"]
    — required for effective_gate_margin and
    effective_variance_threshold normalisation in skill_11
[ ] If temporal_signal.present == True and task_type == regression:
    MAE_naive_baseline present in SKILL_STATE["eda"]["MAE_naive_baseline"]
    — computed as mean absolute difference between consecutive temporal observations
      (calculated group-wise, excluding boundary crossings, if a group column is present):
      MAE_naive = mean(|y_t - y_{t-1}|) (required for MASE gating in skill_11)
[ ] skill_05 cv_strategy written with selection_reason
    — only valid after skill_04 outputs confirmed
[ ] spatial_signal.group_col populated if spatial_signal
    present and group_signal absent
[ ] challenge_config.json temporal lock active —
    confirmed read-only
[ ] seed written to config
[ ] skill_15 has logged all Phase 1 write events
[ ] If target_config present with >1 entry:
    SKILL_STATE["eda"][f"{name}_std"] present for every
    regression target_spec in target_config.targets
[ ] If target_config present with >1 entry:
    each target_spec's own task_type/metric/target_domain_bounds
    confirmed (in addition to, not instead of, top-level fields
    when single-target)
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
Reads: reports/audits/feature_policy.json
Asserts: all blocked columns absent from feature matrix
On violation:
    Write violation entry to SKILL_STATE.json
    Halt — do not proceed to skill_06
Proceeds to skill_06 only if gate passes
```

---

**`skill_06_preprocessing` — Imputation pipeline:**

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

Multi-target extension:

```python
def run():
    config = ChallengeConfig.load()
    targets = config.get("target_config", {}).get("targets")
    if not targets:
        ... existing v2.2 code, completely unchanged ...
        return
    for target_spec in targets:
        cv_strategy = config["cv_strategy"]  # same CV object for all
                                              # targets — A7 still holds
        oof_preds, oof_score = _train_cv(cv_strategy, target_spec,
            seed=config["reproducibility"]["seed"])
        write_oof_record(branch_name="anchor",
            target_name=target_spec["name"], scores=oof_score,
            cv_strategy_id=cv_strategy["selection_reason"],
            seed=config["reproducibility"]["seed"])
    composite = compute_composite_score(targets, SKILL_STATE)
    SKILL_STATE["anchor_oof_score"] = composite
    SKILL_STATE["anchor_oof_score_per_target"] = {
        t["name"]: SKILL_STATE[f"branch_anchor_{t['name']}_oof"]["scores"]
        for t in targets
    }
```

Critical preservation: anchor_oof_score remains a single scalar — the
composite. Every downstream consumer (skill_11 condition 3,
anchor_challenge, pseudo_label_result baseline selection, Gate 1 CV
override) works completely unchanged. Multi-target is invisible below
the composite-scoring layer.

Human Gate 1 surfacing requirement (multi-target only): the operator
review must see anchor_oof_score_per_target alongside the composite,
so one underperforming target cannot hide behind a favorable composite.
This is a presentation requirement on the Gate 1 prompt, not a gate
logic change.

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
      and top features from anchor
      **DOCUMENTATION ERROR:** Original text claimed "top SHAP features"
      but `skill_10_shap.py` has no anchor-only invocation mode. SHAP values
      are not available during `skill_07` feature generation. This rule cannot
      be implemented as documented.

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

> **Systematic MI and Pairwise Leakage Audit (S6):**
> - **Two-Tier Leak Audit:** Decoupled systematic pre-filtering MI audit runs independently on all regression features, every time, regardless of whether the SHAP dominance ratio check fires.
>   - **Classification:** NMI >= threshold remains primary/blocking.
>   - **Regression — Primary (blocking):** Pearson |r| >= `regression_threshold` (0.98) flags feature to `leaked_features`, blocks promotion, and triggers `skill_07` regenerate directive (checked on SHAP-dominant feature).
>   - **Regression — Verification (advisory, non-blocking):** If `enable_mi_regression_subsample: true` in config, subsampled `mutual_info_regression` runs systematically on all feature columns. Features flagged by MI are written to `SKILL_STATE.json["leakage_mi_advisory"]` (not `leaked_features`), do NOT block `skill_11` promotion, and do NOT trigger auto-regeneration.
>   - **Pairwise Multicollinear Leak Detection (advisory, non-blocking):** If `enable_mi_regression_subsample: true` is enabled in config, the system computes the joint mutual information score for all pairs of the top $N$ features (default $N=10$, config-overridable via `mi_pairwise_top_n`, ranked by mean absolute SHAP value). For each pair, the joint MI is computed on the 2-column feature matrix against the target, normalized by dividing the joint MI by the target entropy (for classification) or `var(y)` (for regression) to yield a scale-bounded score. If the normalized joint MI score exceeds the threshold `mi_pairwise_threshold` (default `0.90`), the pair is written to `SKILL_STATE.json["leakage_pairwise_mi_advisory"]`.
>     - **Reproducibility (R1):** Subsampling and noise injection use the global configuration seed to ensure deterministic, reproducible audits.
>     - **Limitations:** (1) **SHAP Scope Restriction:** Scanning is limited to the top-$N$ SHAP features; it does not capture split-leakage involving lower-ranked features or split-leakage across 3+ features. (2) **Multiple testing:** Performing $C(N, 2)$ tests (45 tests at $N=10$) increases the family-wise error rate; no multiple-testing correction (e.g. Bonferroni) is applied, hence the audit is strictly advisory.
>   - **Surfacing:** Non-empty `leakage_mi_advisory` and `leakage_pairwise_mi_advisory` entries are surfaced to the operator at **Human Gate 2** for review. Threshold parameters are read from `challenge_config.json` per A5.

Multi-target extension:

For each target_spec in target_config["targets"]:
    Run the existing skill_10 SHAP contract above, unchanged, scoped
    to this target's OOF predictions and per-fold trained model.
    Write leaked_features under key: leaked_features_{target_name}

skill_11 gate condition 1 becomes, for multi-target only: branch must
be absent from EVERY target's leaked-features list. Leakage on any
single target blocks promotion of the branch as a whole.

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

Multi-target variant:
```json
{
  "metric_analysis": {
    "per_target": {
      "<target_name>": { "fold_scores": [], "fold_score_variance": 0.0 }
    },
    "composite_fold_score_variance": 0.0,
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

composite_fold_score_variance uses the identical weighted-normalized-
distance approach as the composite score, computed per-fold before
aggregating with ddof=1.

For multi-target competitions, the variance threshold uses:

effective_target_std = sqrt(
    sum(w_i * sigma_i^2 for i in regression_targets)
    / sum(w_i for i in regression_targets)
)
effective_variance_threshold = (
    config["variance_gate_threshold"] * (effective_target_std ** 2)
)

where w_i is target_config["targets"][i]["weight"] and sigma_i is
SKILL_STATE["eda"][f"{target_config['targets'][i]['name']}_std"].
Dividing by the sum of regression weights ensures consistent scaling
regardless of classification target presence.

High fold score variance signals the model is not
generalising uniformly across the distribution.

> **Nadeau-Bengio Corrected Variance & 1-SE Promotion Margin (S1 & S9):**
> `skill_12_metric` computes Nadeau-Bengio corrected CV fold score variance (`fold_score_variance_nb`) and standard error (`se_oof`), and writes both to `metric_analysis`. `skill_11_gate` reads `se_oof` from state and applies a 1-SE promotion margin:
> `Var_NB = Var_sample(ddof=1) * (1/K + n_val/n_train)`
> `SE_OOF = sqrt(Var_NB)`
> `effective_gate_margin = max(gate_margin_scale_adjusted, 1 * SE_OOF)`
> S1 and S9 shipped together as required. Note that sample size scaling (including the `1/K` factor) is already mathematically incorporated into `Var_NB` itself via the K-fold geometry correction factor; dividing by K again in `SE_OOF` is a double-scaling error. `fold_score_variance` (the primary reported variance) is the NB-corrected value.
>
> **NB scope — single-target vs multi-target (verified against code, 2026-08-03):** The Nadeau-Bengio correction is applied to the **single-target** `fold_score_variance` and to each **per-target** `fold_score_variance` in the multi-target `per_target` block (`skill_12_metric.py` L121–123, L239–247). The multi-target **`composite_fold_score_variance` is NOT Nadeau-Bengio corrected** — it is computed as the raw unbiased sample variance (`ddof=1`) of the composite fold scores (`skill_12_metric.py` L173: `composite_variance = float(np.var(composite_fold_scores, ddof=1))`). This is a deliberate, code-verified asymmetry: the composite is an aggregate diagnostic, while the per-target and single-target variances feed the gate's 1-SE margin and inverse-variance weighting. `skill_11_gate` consumes the NB-corrected per-target variances via `_target_fold_variance()` (L191–220) and the single-target NB `fold_score_variance` + `se_oof` via `_effective_thresholds()` (L134–138).

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

**Pseudo-label recombination policy (A12):** Immediately after the retraining loop completes, and before the 'if at least one retrained branch passes skill_11' check, `skill_21` must enforce the `pseudo_label_recombination_policy` from `target_config`. Permitted values: `"freeze_unaugmented_targets_at_original"` (composite uses augmented OOF for classification targets, original OOF for regression targets) or `"block_composite_until_all_targets_augmented_or_none"` (composite calculation blocked unless all targets were augmented or none were).

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
           If config["metric"] in ["rmsle", "mase"]:
               effective_variance_threshold = config["variance_gate_threshold"]
               # RMSLE and MASE are scale-invariant/dimensionless; no scale normalisation needed.
           Else:
               effective_variance_threshold = (
                   config["variance_gate_threshold"] *
                   (SKILL_STATE["eda"]["target_std"] ** 2)
               )
               # raw threshold is scale-sensitive for original-scale
               # regression metrics; target_std written by skill_04 during Phase 1.
       If config["task_type"] == "classification":
           effective_variance_threshold = config["variance_gate_threshold"]
           # classification metrics are bounded — no scale normalisation needed.
3. OOF improvement over anchor passes directional check:
       Effective gate margin:
           If config["task_type"] == "regression":
               If config["metric"] in ["rmsle", "mase"]:
                   effective_gate_margin = config["gate_margin"]
                   # RMSLE/MASE are scale-invariant by construction — computed
                    # in dimensionless spaces. Raw threshold applies.
               Else:
                   effective_gate_margin =
                       config["gate_margin"] *
                       SKILL_STATE["eda"]["target_std"]
                   # For original-scale regression metrics (RMSE, MAE),
                   # target_std normalisation makes gate_margin
                   # scale-invariant across competitions with different
                   # target magnitudes. (gate_margin: 0.001 is trivially
                   # small for RMSE on a target with σ_y = 500.)
               If SKILL_STATE["eda"]["target_std"] == 0.0
                  AND config["metric"] not in ["rmsle", "mase"]:
                   effective_gate_margin = config["gate_margin"]
                   # Degenerate target_std — fall back to raw threshold.
                   # Warning written to SKILL_STATE["metadata_warnings"].
                   # Pipeline does not halt — warning is advisory only.
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

> **Target Met (Item 4):** Inverse-variance weighting mirror in the multi-target gate. On competitions where `target_config` has more than one entry, gate conditions 2 and 3 consume the same inverse-variance effective weights as the composite score:
> `w_k_eff = w_k / (sigma_k^2 + epsilon)`
> where `sigma_k^2` is the unbiased sample variance (`ddof=1`) of target k's fold scores and `epsilon = 1e-8`. The effective variance threshold and gate margin for multi-target gate comparisons are computed using these `w_k_eff` weights exactly as documented in the Composite Score Computation section (Section 2). **IMPORTANT NOTE:** This is NOT Kendall & Gal uncertainty weighting (which requires joint differentiable loss training that this decision-tree pipeline does not use). This mirror must ship together with the Section 2 inverse-variance implementation — the two formulas are the same formula and must never diverge.

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
   (default: fixed absolute thresholds conf_pos >= 0.85, conf_neg <= 0.15 from CONF_POS_DEFAULT and CONF_NEG_DEFAULT in skill_21_pseudo_label.py)
```

> **[IMPLEMENTED — v2.5] S8:** Hybrid Adaptive Pseudo-labeling.
> - **Mechanism:** Class-wise quantile selection with a 0.70 floor (`p1 >= 0.70` / `p0 >= 0.70` gates in `skill_21_pseudo_label.py` L779–780). Top `pseudo_quantile` fraction per class selected; defaults to top 20% when config key absent.
> - **Preconditions & Guards:** Guard condition GC6 still uses `CONF_POS_DEFAULT = 0.85` to gate whether the skill runs at all; quantile selection applies inside the iteration loop once guards pass. `min_pseudo_samples` guard at L782.
> - **Ranking & Multi-target:** Deterministic `method='first'` tie-breaking. Scoped independently per classification target in multi-target competitions.
> - **Recombination Timing:** Enforced **post-retraining**, immediately after the pseudo-label retraining loop finishes, before passing candidate branches to `skill_11`.

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

> **Residual Diversity / Kuncheva Pruning (S4):** `_prune_collinear` in `oracle_fusion_core.py` (L190-244) threads ground-truth `y_true` into the function body to calculate error residual vectors `e_m = y_pred_m - y_true`. Diversity pruning correlates error residual vectors (`e_A, e_B`) rather than raw predictions (`y_pred_A, y_pred_B`). Task-type correlation methods (Pearson for classification residuals, Spearman rank for regression residuals) are preserved. Call site at L687-691 passes `y_true=y_true`. Verified by `test_prune_collinear_residuals` in `tests/test_correlation_pruning.py` (L31-118).

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
        gc6_confidence_threshold_met: fixed absolute thresholds met
          (conf_pos >= 0.85, conf_neg <= 0.15)
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
Before any Zindi submission API call:

    Query live remaining_submissions from Zindi client (or read cached state if offline/error).

    If remaining_submissions <= 0:
        Raise HardAbortException ("Zindi reports zero remaining submissions today."
        or "State-side budget guard: zero submissions remaining.")
        No submission API call under any condition.

    If remaining_submissions == 1:
        Write budget_warning to SKILL_STATE.json:
            {
              "budget_warning": {
                "remaining_submissions": 1,
                "source": "live" | "cached",
                "timestamp": "<UTC ISO timestamp>"
              }
            }
        Prompt human operator for explicit confirmation before proceeding.
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

## Plugin Contract (multi-target competitions only)

```python
# plugins/base_extractor.py
from abc import ABC, abstractmethod
import pandas as pd
from pathlib import Path

class FeatureExtractor(ABC):
    @abstractmethod
    def extract_features(self, raw_data_dir: Path, config: dict
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        - All column names read from config — no string literals for
          competition-specific column names (A5).
        - Must NOT include target columns in output.
        - Must handle missing test data gracefully.
        - Must log feature names to reports/feature_manifest.json.
        - Must be deterministic (A7) — same input -> same output,
          no API calls, no randomness, no filesystem side effects
          beyond data/processed/.
        """
        pass
```

config["file_manifest"] and config["plugin_config"] are written by
skill_02 from the competition's actual file names and any plugin-
specific parameters, preserving A5 with zero exceptions. Any example
implementation (e.g. a World Cup plugin) must read group_col, train_file,
test_file, id_col, and aggregation column lists entirely from config —
zero hardcoded competition-specific string literals anywhere in the
plugin body.

---

## 5. Research Sidecar

**Skills:** `skill_00` (continuous) → `skill_18`,
`skill_19`, `skill_20` (triggered)

The sidecar is not a phase. It is a continuous intelligence
layer with distinct trigger points. It is non-blocking at
every consumption point (see A9).

---

### Sidecar State Interface & Consumption Contract

Sidecar recommendations are stored in `SKILL_STATE.json` under the `sidecar_recommendations` key as a list of recommendation objects.

**Recommendation Item Schema:**
```json
{
  "recommendation_id": "rec_001",
  "source_skill": "skill_19_code_miner",
  "target_phase": "phase_2_feature_engineering",
  "payload": {
    "feature_idea": "monthly_ratio_VH_VV",
    "rationale": "SAR polarization ratio for vegetation density",
    "confidence": 0.85
  },
  "timestamp": "2026-08-03T18:00:00Z",
  "status": "pending"
}
```

**Consumption & Fallback Rules:**
1. **Safe Access Pattern:** Every consuming skill must access `sidecar_recommendations` using `SKILL_STATE.get("sidecar_recommendations", [])`. Direct bracket indexing (`SKILL_STATE["sidecar_recommendations"]`) is strictly prohibited.
2. **Filtering:** Consuming skills filter for items where `status == "pending"`.
3. **Non-Blocking Fallback:** If `sidecar_recommendations` is absent, empty, or unreadable, the consuming skill proceeds seamlessly using standard fingerprint defaults. No exception is raised and execution is never halted.
4. **State Update:** Upon consuming a recommendation, the consuming skill updates that item's `status` to `"consumed"` (or `"dismissed"`) in `SKILL_STATE.json`.

---

### Trigger Schedule

| Skill | Trigger | Informs |
|---|---|---|
| `skill_00` | Competition start → close | All phases continuously |
| `skill_18` | Phase 1 completes | Phase 2B feature generation |
| `skill_19` | Phase 2A completes | Phase 2B feature patterns |
| `skill_20` | Phase 3A completes (anchor + SHAP ready) | Phase 3B audit, Phase 2B next variants |
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

### Cross-Competition History Log Schema

After every competition close, `skill_22` writes one entry to
`competition_history/history_log.jsonl`. `skill_18` and `skill_20`
read this log as prior knowledge for the next competition.

```
Location: competition_history/history_log.jsonl
Format:   One JSON object per line, one entry per competition close

Fields per entry:
{
  "slug": "",
  "competition_id": "",
  "completed_at": "",
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
  "variants_passed": 0,
  "promoted_branches": [],
  "selected_submissions": [],
  "reproducibility_audit_passed": false,
  "gate_thresholds": {
    "shap_leak_threshold": 3.0,
    "variance_gate_threshold": 0.01,
    "gate_margin": 0.001
  },
  "competition_close_date": "",
  "total_training_duration_seconds": 0.0,
  "total_carbon_kg_estimate": null,
  "tracker_method": "mlco2_formula | codecarbon | not_instrumented"
}
```

**Bayesian threshold evolution:** After several competitions,
`shap_leak_threshold`, `variance_gate_threshold`, and `gate_margin`
are reviewed against historical OOF-to-LB correlation data and
updated in config.

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

> **R6 — Computed-Artifact Fingerprinting Scope (Path 2).**
> - **Raw Intake Files:** Retain exact MD5 hashing (`skill_01`).
> - **Computed Artifacts Scope:** Evaluated exclusively at `skill_22` reproducibility sign-off (Path 2) across the following derived artifacts, keyed in `SKILL_STATE["derived_artifact_fingerprints"]`:
>   - `cleaned_feature_matrix` — written by `skill_06` (cleaned dataset).
>   - `engineered_feature_matrix_{branch_name}` — written by `skill_07` (engineered features, per branch).
>   - `oof_predictions_{branch_name}` — written by `skill_08` / `skill_21` (OOF predictions, per branch).
> - **Tolerance-Based 3-Tier Verification Bands:**
>   - **Tier 1 (<= 1e-6 relative delta):** Bit/float identical pass.
>   - **Tier 2 (1e-6 to 1e-5 relative delta):** Soft-warning issued; requires explicit operator sign-off at Human Gate 5.
>   - **Tier 3 (> 1e-5 relative delta):** Hard-halt — non-reproducible artifact rejected.
> - **[IMPLEMENTED — v2.5] S10:** `write_artifact_fingerprint()` (defined in `zindian/state.py` L347) is now called by `skill_06_preprocessing.py` (L196), `skill_07_features.py` (L1288, L1884), `skill_08_anchor.py` (L497, L827), and `skill_21_pseudo_label.py`. `skill_22_reproducibility_audit.py` `_audit_derived_artifact_fingerprints()` (L206) reads the populated dict and applies the 3-tier tolerance bands. R6 is closed.

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

**R5 — Carbon and compute impact is measured and reported.**

```
Every skill that performs model training, feature
engineering, SHAP computation, calibration, or
inference must instrument its compute duration and
estimated carbon output.

Storage extends the existing telemetry pattern:
    SKILL_STATE["telemetry.{skill_name}"] already contains:
        duration_sec       : wall-clock seconds (measured by
                             orchestrator run_skill() wrapper)
        peak_memory_mb     : peak RSS (measured via tracemalloc
                             by orchestrator)
    New fields added to the same key:
        carbon_kg_estimate : kg CO2 estimated for this skill run
        tracker_method     : "codecarbon" | "mlco2_formula"
                             | "not_instrumented"
        hardware_type      : "cpu" | "gpu" | "tpu"
        region             : cloud/local region code

Telemetry is measured by the orchestrator's run_skill()
wrapper for every skill automatically. Carbon estimation
is added by a post-skill hook: the orchestrator passes the
already-measured duration_sec and peak_memory_mb to a carbon
calculator, which computes carbon_kg_estimate using:

    Primary method: CodeCarbon (optional dependency)
    Fallback:       ML CO2 Impact formula

ML CO2 Impact formula fallback:
    TDP_watts = config["infrastructure"].get("tdp_watts", 15.0)
    PUE = config["infrastructure"].get("pue", 1.0)
    carbon_intensity = config["infrastructure"].get(
        "carbon_intensity_gco2_per_kwh", 494.0)
    energy_kwh = (TDP_watts * PUE * duration_seconds)
                 / 3_600_000
    carbon_kg_estimate = (energy_kwh * carbon_intensity) / 1000

config["infrastructure"] block (written by skill_02
during Phase 1 mutable window):
{
  "hardware_type": "cpu | gpu | tpu",
  "region": "ke | us-east-1 | eu-central-1 ...",
  "tdp_watts": 15.0,
  "pue": 1.0,
  "carbon_intensity_gco2_per_kwh": 494.0
}

[RESOLVED — v2.5] config["infrastructure"] is written by skill_02
at L838–849 within the allowed_write_phases window. The C1 bootstrap
phase string issue is resolved — "phase_1_integrity_locked" is in
skill_02's allowed_write_phases (L859–870). The fallback path
(not_instrumented) now applies only when CodeCarbon is absent and
the ML CO2 formula config keys are not present.

Aggregation is performed by the orchestrator's post-phase
hook (not by skill_15). After every phase completes, the
orchestrator aggregates per-skill telemetry into:
    SKILL_STATE["telemetry.aggregate"] = {
        "phase": phase,
        "total_duration_sec": 0.0,
        "total_carbon_kg_estimate": null,
        "skill_count": 0,
        "written_at": "timestamp"
    }

[RESOLVED — v2.6] `run_phase()` in `zindian/orchestrator.py` now aggregates per-skill telemetry and writes this block to state. `skill_22` validates its presence and contents at sign-off.

skill_22 verifies at sign-off:
    telemetry.aggregate present and non-null
    total_carbon_kg_estimate or explicit
        not_instrumented with reason
    config["infrastructure"] block present
    carbon_kg_estimate reported in governance report
        with tracker_method cited

Competition history log additions:
    "total_training_duration_seconds": 0.0,
    "total_carbon_kg_estimate": null,
    "tracker_method": "mlco2_formula | codecarbon
                       | not_instrumented"
```

Mandatory skills (must instrument carbon):
    _lightgbm_shared.py  (training loop — highest compute)
    skill_07_features.py (feature engineering)
    skill_08_anchor.py   (anchor model training)
    skill_09_calibration.py
    skill_10_shap.py     (SHAP — highest confirmed: 24.5s)
    skill_13_oracle_fusion.py
    skill_14_inference.py

Exempt skills (config/state reads, negligible compute):
    skill_01, 02, 03, 05, 06, 11, 12, 15, 16, 17, 22
    skill_00, 18, 19, 20, 21 (sidecar/research)

Hardware constraint: All carbon estimation assumes
local CPU (no GPU) by default. GPU carbon tracking
requires explicit hardware_type = "gpu" in config.


---

## 7. Known Gaps Registry

This section documents every confirmed discrepancy between the SoT
contract and the current codebase. Items are tracked by severity.
Last verified: August 2026 (v2.6 audit — S2 pinpointed to skill_08_anchor.py L521–535).

---

### RESOLVED — Removed from active tracking

The following items have been confirmed resolved by direct code inspection
and are recorded here for audit trail only.

| ID | Description | Resolution |
|---|---|---|
| C1 | Bootstrap `dag_phase` prevented config writes | `"phase_1_integrity_locked"` added to `allowed_write_phases` in skill_02 (L867) and skill_05 (L610); bootstrap sets this phase at L119 |
| GAP-4 | `skill_04_eda` did not write `temporal_index_confirmed` / `group_structure_confirmed` | Resolved v2.3 — both lean booleans now derived from BAND_MM pattern, datetime/monotonicity dtype inference, and <5% cardinality ratio |
| S1/S9 | Bessel's correction underestimation + absolute promotion margins | Implemented 2026-08-03 — `skill_12_metric` L167–182 computes NB-corrected `fold_score_variance_nb` + `se_oof`; `skill_11_gate` L134 applies `max(margin, 1.0 * se_oof)` |
| S2 | MAPE zero-target bias + MASE baseline & routing | Implemented 2026-08-25 — Secondary diagnostic MASE implemented; primary-metric routing key mapped in `skill_08_anchor.py` `metric_map`. Score-space correction tracked separately as F4 residual. |
| S3 | Non-uniform metric scaling / composite weighting | Implemented 2026-08-03 — Inverse-variance weighting in `skill_12_metric` L88–149 and `skill_11_gate` L188–236 |
| S4 | Correlation-based pruning used raw predictions not residuals | Implemented 2026-08-03 — `_prune_collinear()` in `oracle_fusion_core.py` accepts `y_true` and computes error residuals; call site at L687 passes `y_true` |
| S5 | Multi-target pseudo-label recombination policy not enforced | Implemented 2026-08-03 — Both policies (`freeze_unaugmented_targets_at_original`, `block_composite_until_all_targets_augmented_or_none`) enforced at `skill_21_pseudo_label.py` L567 and L1034–1132 |
| C4 | `skill_17` checked flat `human_gate_2_approved` instead of per-branch keys | `skill_17_governance.py` L96–104 now iterates `human_gate_2_*_approved` prefix pattern, explicitly excludes flat legacy key |
| M6 | `skill_11` / `skill_12` might silently fall back to wrong `target_std` key on multi-target | `skill_11_gate` L107–115 reads `target_std` with a fallback scan across all `_std`-suffixed keys in the EDA block |
| DRIFT-3 | Orchestrator had no validated split-skill dispatch mechanism | `zindian/orchestrator.py` L318–358 uses `hasattr` + `inspect.signature` to dispatch split-skill functions with filtered kwargs |
| C2 | Preflight did not validate `feature_policy.json` required keys | Not a preflight responsibility — `policy_gate()` in `skill_03_legality.py` validates required keys at Phase 2A runtime; no preflight change needed |
| S7 (partial) | Preflight wrote to `reports/` root instead of `reports/audits/preflight/` | `scripts/preflight_enforce.py` L877–880 now writes to `reports/audits/preflight/<timestamp>.json` |
| S7 (partial) | `skill_12_metric` did not consume buffered CV splits | Not a gap — `skill_12` reads `fold_scores` from existing OOF state records; it never re-runs CV splits |
| S7 | Spatial CV Buffer: `skill_09` not consuming buffered splits | Implemented 2026-08-24 — `skill_09_calibration.py` L207–210 calls `load_explicit_cv_splits(state)` when explicit splits are present |
| S8 | Fixed pseudo-label thresholding — adaptive quantiles not implemented | Implemented 2026-08-24 — class-wise quantile selection with 0.70 floor at `skill_21_pseudo_label.py` L762–788; `min_pseudo_samples` guard; `pseudo_quantile` config key drives selection |
| S10 | No skill wrote `derived_artifact_fingerprints` | Implemented 2026-08-24 — `write_artifact_fingerprint()` called by skill_06 (L196), skill_07 (L1288, L1884), skill_08 (L497, L827); `skill_22` verifier now has data to check |
| S6 | Multicollinear Leakage Systematic Pairwise MI scan | Implemented 2026-08-24 — pairwise MI scan for top-10 SHAP features in `skill_10_shap.py`. |
| S11 | skill_18/20 legacy root dual-writes | Implemented 2026-08-24 — consolidated all sidecar writes to `reports/diagnostics/` only; updated all consumers and tests. |
| Preflight | Multi-target OOF completeness check not per-branch | Implemented 2026-08-24 — added per-branch OOF completeness count assertion to A7 check in `preflight_enforce.py`. |
| R5 | `telemetry.aggregate` not written | Implemented 2026-08-24 — orchestrator post-phase loop aggregation added; verification checks implemented in `skill_22`. |
| Track 2B | Orphaned `feature_policy.json` at root | Implemented 2026-08-24 — verified via workspace and competition grep that no root writes or reads remain. |
| F3 | Hardcoded seeds in skill_10_shap.py | Resolved 2026-08-25 — Replaced hardcoded seed=42 with config-driven random_seed parameter. |
| F4 (partial) | MASE Routing in target lifecycle | Routing key landed 2026-08-25 — `"mase"` mapped in `skill_08_anchor.py` `metric_map`, preventing silent `oof_f1` fallback. Fold-score space remains RMSE (no naive-baseline scaling in `_lightgbm_shared.py`); residual tracked in OPEN section as F4. |

---

### OPEN — Active gaps

**F1 — Pairwise MI Scale Invariance**

```
Description:    The S6 pairwise MI audit function uses nearest neighbor distances to
                estimate mutual information. This raw MI estimate is not scale-invariant
                for continuous targets/features. Rescaling feature or target values
                causes the MI score to drift.
Status:         Pending normalization/standardization research (e.g., Linfoot correlation
                coefficient or copula-based normalization).
Severity:       Low — does not halt pipeline, but warning thresholds are scale-sensitive.
```

**F2 — KSG Validator Bivariate Reference Test**

```
Description:    The custom KSG bivariate MI estimator lacks direct unit tests validating
                its output accuracy against closed-form mathematical solutions (e.g.
                Bivariate Gaussian).
Status:         Pending numeric-equivalence reference test suite integration.
Severity:       Low — estimator functions correctly, but numerical precision is unverified.
```

**F4 (partial) — MASE Score-Space Residual**

```
Description:    Primary-metric routing exists ("mase" key in skill_08_anchor.py
                metric_map), but oof_logloss resolves to oof_rmse (skill_08 L223) and
                _lightgbm_shared.py computes plain RMSE fold scores — there is no
                branch dividing by MAE_naive_baseline when config metric is "mase".
                Meanwhile skill_11_gate SCALE_INVARIANT_METRICS treats mase thresholds
                as scale-invariant. A competition configuring metric: "mase" would
                therefore compare RMSE-space scores against MASE-calibrated
                thresholds — a unit-space mismatch, not a correct MASE evaluation.
Status:         Pending SoT §2 Regression Target Transformation Lifecycle definition
                of a "mase" case, plus a naive-baseline-scaled fold-scoring branch in
                _lightgbm_shared.py. Do not implement without that contract first.
Severity:       Medium — silently wrong gating scale if mase is ever configured;
                latent only (no active competition currently uses metric: "mase").
```

---

### DEFERRED — Future roadmap items

**[DEFERRED — v3.0] GAP-3 — SHAP interaction effects**

```
Description:    SHAP-based leakage detection captures interaction effects
                between features.
Code Reality:   skill_10_shap.py computes mean |SHAP| per feature only.
                TreeSHAP interaction values (shap_interaction_values) are
                not computed.
Status:         Deferred to v3.0 per project roadmap. Requires TreeSHAP v2.
Severity:       Low — interaction-based leakage is rare in standard tabular
                competitions.
```

---
*Version: v2.6 — OPEN (F1, F2, F4-residual)*
*Next: v3.0 — deferred items: GAP-3 (SHAP interaction effects)*
*S2/F4 note (2026-08-25): "mase" routing key mapped in skill_08_anchor.py metric_map;
fold-score space correction (naive-baseline scaling) remains open as F4 (partial).*
