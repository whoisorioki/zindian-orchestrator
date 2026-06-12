# Orchestrator Current State

This document captures the current state of the Zindian orchestrator as implemented in the repository. It focuses on the control plane, the skill registry, the phase layout, and the role of each skill in the pipeline.

## Current Architecture

The orchestrator is now a dynamic skill runner rather than a static import list.

### Core control flow

- `zindian/orchestrator.py` discovers every `skill_*.py` module under `zindian.skills` at import time.
- `SKILL_REGISTRY` is built dynamically with `pkgutil.iter_modules()` and `importlib.import_module()`.
- `run_skill()` executes one skill by name.
- `run_phase()` resolves the phase from `challenge_config.json` when `phase_skill_map` is present, otherwise it falls back to the built-in phase lists.
- `run_deep_research()` wires the research sidecar flow across Skills 18, 19, and 20.
- `resolve_competition_paths()` is the common path resolver used by skills that need the active competition directory.
- `ChallengeConfig` and `SkillStateStore` remain the two central state/config abstractions.

### Phase layout in the orchestrator

The built-in phase map for execution dispatch currently is:

- Phase 1: `skill_01`, `skill_02`, `skill_15`
- Phase 2: `skill_03`, `skill_08`
- Phase 3: `skill_04`, `skill_05`, `skill_09`, `skill_10`
- Phase 4: `skill_11`, `skill_16`
- Phase 5: `skill_13`, `skill_14`, `skill_17`

If `challenge_config.json` defines `phase_skill_map`, the orchestrator uses that instead. At import time it also validates that the configured skills exist in the discovered registry.

### Canonical Validation Sequence (Three-Lens Framework)

The Three-Lens engine (`zindian/three_lens.py`) evaluates pipeline checkpoints using the canonical 6-phase model defined in the Source of Truth:
- **Phase 1**: Competition Fingerprint + Config Lock (`skill_01`, `skill_02`, `skill_15`)
- **Phase 2A**: Data Cleaning (`skill_03`, `skill_06`)
- **Phase 2B**: Signal Search (`skill_07`, `skill_08`)
- **Phase 3A**: Generalisation Audit (`skill_09`, `skill_10`, `skill_12`)
- **Phase 3B**: Promotion and Fusion (`skill_11`, `skill_13`)
- **Phase 4**: Governance (`skill_14`, `skill_16`, `skill_17`, `skill_22`)

### Research pipeline wiring

The deep research path is separate from the normal phase loop:

1. Skill 18 collects literature and prior-art signals.
2. Skill 19 mines public code and strategy patterns.
3. Skill 20 turns those signals into validated hypotheses.

That path writes its artifacts into the active competition `reports/` directory.

## State Model

The orchestrator behavior is governed by two files inside the active competition folder:

- `SKILL_STATE.json` stores execution state, branch metadata, submission counters, and phase progress.
- `challenge_config.json` stores competition rules such as metric, probability handling, external-data policy, automation policy, and phase mapping.

The current design is intentionally config-driven so the same control plane can run different competitions without hardcoding competition-specific rules into the orchestrator.

## Skill Roles

The table below describes the current role of each skill module in the repository.

| Skill | Role | Current responsibility |
|---|---|---|
| `skill_00_discussion_monitor` | Discussion monitor | Watches competition discussion signals and captures useful community intelligence. |
| `skill_00_zindi_monitor` | Platform monitor | Tracks Zindi-side updates and competition metadata that may affect workflow. |
| `skill_01_integrity` | Integrity audit | Locks MD5 fingerprints for target and raw files so later transforms can be checked against a stable baseline. |
| `skill_02_intake` | Challenge intake | Reads competition metadata and writes rules into `challenge_config.json`. |
| `skill_03_legality` | Compliance filter | Checks that the current competition setup respects allowed data, policy, and competition constraints. |
| `skill_04_eda` | EDA / violation scan | Produces early data analysis and checks for domain or compliance issues. |
| `skill_05_cv` | CV architect | Chooses or validates the cross-validation strategy used by downstream model work. |
| `skill_06_cleaning` | Data cleaning | Performs lightweight preprocessing such as median filling and constant-column removal. |
| `skill_07_features` | Feature sandbox | Builds isolated feature variants, trains them, and records OOF/test artifacts. |
| `skill_08_anchor` | Anchor baseline | Produces the first confirmed baseline and establishes the anchor branch/metric. |
| `skill_09_calibration` | Probability calibration | Calibrates predicted probabilities using methods such as Platt scaling or isotonic regression. |
| `skill_10_shap` | Feature audit / leakage scan | Uses SHAP and feature inspection to look for leakage, instability, or suspicious predictors. |
| `skill_11_gate` | Promotion gate | Compares variants to the anchor and only promotes a branch when the gate passes. |
| `skill_12_metric` | Metric analysis | Evaluates metric trade-offs, thresholds, and OOF score behavior. |
| `skill_13_oracle_fusion` | Ensemble / fusion | Combines candidate models or submissions into a fused output. `skill_13_ensemble.py` is a compatibility shim that re-exports this implementation. |
| `skill_14_inference` | Submission formatting | Validates and post-processes submission files before external use. |
| `skill_15_reporter` | Reporter / ledger | Writes experiment summaries and maintains the DuckDB-backed record trail. |
| `skill_16_submit` | Submission governance | Checks submission formatting, budget, and comment policy before Zindi submission. |
| `skill_17_governance` | Final governance | Selects the final submissions and records reproducibility and selection decisions. |
| `skill_18_librarian` | Literature mining | Searches papers and domain evidence to propose hypotheses and variables. |
| `skill_19_code_miner` | Code mining | Extracts reusable patterns from public code and prior solutions. |
| `skill_20_scientist` | Hypothesis validation | Validates research hypotheses against the current feature/method context. |
| `skill_21_pseudo_label` | Pseudo-labeling | Expands training data with cautiously selected pseudo-labels when appropriate. |
| `skill_22_reproducibility_audit` | Reproducibility audit | Freezes and audits repository state so the working tree can be reproduced. |

## Skill Audit: Current Code, Logic, and Functionality

This section audits the present repository state skill by skill. It distinguishes between modules that are fully implemented, compatibility shims, and modules that are present but narrow in scope.

### Inventory summary

- Present numbered skills from `00` through `22`: all are present in some form.
- Duplicate/compatibility cases:
	- `skill_13_ensemble.py` is a shim that re-exports `skill_13_oracle_fusion.py`.
	- `skill_00` exists as two modules: `skill_00_discussion_monitor.py` and `skill_00_zindi_monitor.py`.
- Missing numbered skills in the current repository: none.

### Audit matrix

| Skill | Present? | Code state | Logic state | Functionality state | Notes |
|---|---|---|---|---|---|
| `skill_00_discussion_monitor` | Yes | Present as a dedicated monitor module. | Tracks community discussion signals. | Sidecar intelligence collection. | Used to capture discussion-derived hints rather than training logic. |
| `skill_00_zindi_monitor` | Yes | Present with a `run()` entrypoint. | Polls Zindi-side competition updates. | Platform monitoring / metadata refresh. | Complements discussion monitoring. |
| `skill_01_integrity` | Yes | Stable entrypoint and state writes. | Hash-locks the raw data boundary. | Verifies file integrity and reproducibility anchors. | One of the most critical guardrails; already tested through coverage. |
| `skill_02_intake` | Yes | Present and configurable. | Converts competition metadata into policy. | Populates `challenge_config.json` from the competition surface. | Acts as the competition rules intake layer. |
| `skill_03_legality` | Yes | Present and callable. | Checks allowed data / policy constraints. | Compliance filter for competition-specific legality. | Serves as a guard before model-specific work. |
| `skill_04_eda` | Yes | Present and lightweight. | Produces exploratory checks and issue detection. | EDA and early violation scan. | Narrower than a full analysis notebook; intentionally policy-aware. |
| `skill_05_cv` | Yes | Present with strategy selection entrypoint. | Chooses or validates cross-validation structure. | Cross-validation architecture helper. | Keeps split logic separate from model training. |
| `skill_06_cleaning` | Yes | Present and currently implemented. | Performs simple preprocessing only. | Fills numeric missing values and removes constant columns. | Low-risk cleaning layer; no heavy feature engineering. |
| `skill_07_features` | Yes | Present and central to feature search. | Orchestrates isolated feature variants. | Builds and evaluates variant feature sets with OOF/test artifacts. | Primary experimentation surface in the pipeline. |
| `skill_08_anchor` | Yes | Present and baseline-oriented. | Establishes the first accepted anchor branch/score. | Baseline model and anchor promotion setup. | The anchor defines the comparison point for later gating. |
| `skill_09_calibration` | Yes | Present and type-stable after the recent fix. | Fits calibration transforms on prediction probabilities. | Platt and isotonic calibration support. | Addressed earlier Pylance issues by making the calibrator branches explicit. |
| `skill_10_shap` | Yes | Present and audit-focused. | Inspects model explainability and suspicious features. | SHAP-based leakage / feature importance audit. | Helps detect fragile or suspicious feature behavior. |
| `skill_11_gate` | Yes | Present and gate-centric. | Compares variants to anchor and decides promotion. | Validation gate and branch reset control. | One of the main control points for safe iteration. |
| `skill_12_metric` | Yes | Present and analytical. | Evaluates threshold/OOF trade-offs. | Metric comparison and score analysis. | Useful when a competition metric is not directly optimized by default. |
| `skill_13_oracle_fusion` | Yes | Present and implemented. | Fuses multiple candidate outputs or models. | Ensemble / fusion behavior. | `skill_13_ensemble.py` remains a compatibility re-export only. |
| `skill_14_inference` | Yes | Present and narrow in scope. | Validates submission-shaped artifacts. | Submission formatting and post-processing. | Ensures outputs stay aligned with expected submission schema. |
| `skill_15_reporter` | Yes | Present and operational. | Writes summaries and ledger entries. | Experiment bookkeeping and reporting. | Observability layer for phase outputs and submissions. |
| `skill_16_submit` | Yes | Present and guarded. | Checks submission file and budget constraints. | Submission contract enforcement before Zindi calls. | Final local safety net before platform submission. |
| `skill_17_governance` | Yes | Present and policy-driven. | Selects and records final submissions. | Final governance and reproducibility selection. | Closes the loop on final submission choice. |
| `skill_18_librarian` | Yes | Present and refactored to active competition paths. | Mines literature for domain hypotheses and variable signals. | Literature search, evidence collection, and hypothesis proposal. | No longer hardcodes `ey-frogs`; uses active competition resolution. |
| `skill_19_code_miner` | Yes | Present and research-oriented. | Extracts reusable public-code patterns. | Code mining and strategy pattern harvesting. | Feeds the research-to-feature pipeline. |
| `skill_20_scientist` | Yes | Present and highly structured. | Validates hypotheses against available features and empirical checks. | Turns research signals into validated hypotheses. | This is the bridge from literature/code mining to feature generation. |
| `skill_21_pseudo_label` | Yes | Present and optional. | Applies cautious pseudo-label logic when appropriate. | Expands the training set with pseudo-labeled samples. | Usually used only when the competition setup supports it. |
| `skill_22_reproducibility_audit` | Yes | Present and completed. | Freezes and audits repository state. | Reproducibility snapshot and state validation. | Captures a stable repo state for later replay or audit. |

### Audit findings

- The skill set is complete across `00` through `22`; there are no missing numbered slots in the current repository.
- The main functional split is between control-plane skills (`01`, `02`, `08`, `11`, `15`, `16`, `17`) and research/modeling skills (`07`, `09`, `10`, `12`, `18`, `19`, `20`, `21`).
- `skill_13` is intentionally duplicated as a shim plus implementation because the test and compatibility layer still expect the `skill_13_ensemble` import path.
- The most important logic boundaries currently remain:
	- integrity and config intake before any transform,
	- isolated feature variants before gating,
	- validation gate before promotion,
	- submission guard before any external submit call,
	- reproducibility audit at the end of the chain.

### Test evidence for the audit

The following tests directly exercise the skill surface and the research scaffold:

- `tests/test_skill_coverage.py`
	- Verifies the expected skill exports exist and are callable.
	- Checks scientist inventory and validation behavior.
	- Confirms the Semantic Scholar client behaves correctly.
- `tests/test_deep_research_scaffolds.py`
	- Verifies librarian, scientist, code miner, governance, and orchestrator imports.
	- Checks that research/reporting artifacts exist and have the expected schema.

The repository currently validates with `pytest -q`.

## Notable Implementation Details

- The orchestrator no longer depends on a long chain of static skill imports.
- `run_deep_research()` now resolves paths through the active competition rather than assuming a specific competition name.
- `skill_18_librarian.py` no longer hardcodes `ey-frogs` paths and now resolves the active competition automatically.
- `skill_13_ensemble.py` exists as a shim for compatibility with tests and older references.
- The current validation state for the repository is clean: the full test suite passes.

## Practical Reading Order

If you are trying to understand the orchestrator quickly, read the files in this order:

1. [`zindian/orchestrator.py`](../zindian/orchestrator.py)
2. [`zindian/config.py`](../zindian/config.py)
3. [`zindian/state.py`](../zindian/state.py)
4. [`zindian/paths.py`](../zindian/paths.py)
5. [`zindian/skills/skill_01_integrity.py`](../zindian/skills/skill_01_integrity.py)
6. [`zindian/skills/skill_02_intake.py`](../zindian/skills/skill_02_intake.py)
7. [`zindian/skills/skill_07_features.py`](../zindian/skills/skill_07_features.py)
8. [`zindian/skills/skill_11_gate.py`](../zindian/skills/skill_11_gate.py)
9. [`zindian/skills/skill_15_reporter.py`](../zindian/skills/skill_15_reporter.py)
10. [`zindian/skills/skill_18_librarian.py`](../zindian/skills/skill_18_librarian.py)

## Validation

The repository currently validates with `pytest -q`.
