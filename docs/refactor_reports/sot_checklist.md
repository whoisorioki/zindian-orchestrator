# SoT Compliance Checklist & DoD

## Purpose

Centralized checklist mapping SoT rules to DoD checks. Designed for agents
working on Zindian skills—tick off items as you implement.

## How to use

1. Read the relevant section of `docs/source_of_truth.md` (the SoT).
2. Implement the skill or fix per the SoT contract.
3. Mark off DoD checklist items below.
4. Update this checklist with a new batch entry when done.

---

## Batch 22 — Downstream Plane Hardening (2026-06-03)
- Commit: aabab18fba79e0044eb7a7d4c315b5a199b86fcb
- Scope: Refactor downstream execution plane skills (14, 16, 21, 22) into stateless, configuration-driven pipeline modules.
- Completed:
  - [x] Skill 14: Integrated Human Gate 4 prerequisite checks, dynamic ID mapping, domain-specific continuous clipping/prob bounds validation, and atomic temporary file swaps.
  - [x] Skill 16: Applied prerequisite gate checks (Gate 4 and Gate 2), pre-flight matrix value validation, pre-request budget throttling, and state-driven OOF score lineage extraction.
  - [x] Skill 21: Enforced classification-only task guards, dynamic feature mask exclusions, universal cross-validation override tracking, strict training split data isolation, and the canonical nested `pseudo_label_result` state layout containing all six `gc1..gc6` boolean flags.
  - [x] Skill 22: Converted the static script into an active system auditor by removing competition constants, adding automated static AST scanning for forbidden AutoML tools, and validating environment lockfiles and OOF strategy tags.
- Evidence:
  - Static Check: Pyright/Pylance validation passes cleanly over skills 14, 16, 21, and 22.
  - Test suite extended with targeted unit tests under `tests/` covering boundary conditions; total local test run passes cleanly without regressions.
- Commit Message Convention: `docs(sot): Batch 22 update — downstream plane hardening`

## Batch 23 — Deep Research Sidecar Audit (2026-06-03)
- Commit: aabab18fba79e0044eb7a7d4c315b5a199b86fcb
- Scope: Audit research sidecar skills (18, 19, 20) for non-blocking contract and file/path generality compliance.
- Completed:
  - [x] **Non-Blocking Contract (A9) Verification**: Confirmed all three sidecar skills (`run_librarian`, `run_code_miner`, `run_scientist`) accept `**kwargs`, have try/except wrappers, and log failures rather than raising. No `--require-all` or blocking default flag in any sidecar call.
  - [x] **Strict File/Path Generality (A5) Verification**: Checked for hardcoded paths, absolute paths, and hardcoded competition names. Skill 18: `run_librarian` reads `config_path` and `cache_path` from arguments—clean. Skill 19: `run_code_miner` uses `reports_dir` from `resolve_competition_paths()`—clean. Skill 20: `run_scientist` reads all file paths from arguments—clean. No hardcoded strings found.
  - [x] **State-Aware Loop Triggers**: Verified sidecar triggers in orchestrator.py state-dependent calls: `skill_20` called when `skill_11` gate fails (non-blocking), `skill_18` called on unresolved hypotheses (non-blocking), `skill_19` periodically refreshes patterns (non-blocking).
- Evidence:
  - Manual audit of `zindian/skills/skill_18_librarian.py`, `zindian/skills/skill_19_code_miner.py`, `zindian/skills/skill_20_scientist.py`, and `zindian/orchestrator.py` sidecar wiring. All code paths verified for non-blocking behavior (A9) and generalised file/path usage (A5) per SoT rules.
- Evidence (file):
  - `docs/refactor_reports/sot_batch2.md` (containing full audit trail, relevant code excerpts, risk flags, and snapshot evidence)
- Commit Message Convention: `docs(sot): Batch 23 update — deep research sidecar audit`

## Batch 24 — Final Operational Core Hardening (2026-06-10)
- Commit: *(pending)*
- Scope: Hardening of three core operational skills (06, 15, 17) and repository infrastructure preparation for GitHub publish.
- Completed:
  - [x] **Skill 06 — Cleaning/Imputation**: Restructured cleaning loop so MNAR binary indicator compilation (`_is_missing`) runs completely across all MNAR tracks **before** any imputation. MCAR median/mode values derived from training fold matrix and applied uniformly to test. Dynamic variance scanning drops columns constant in both splits only. Conformed to `run(config, state) -> dict` SoT entry-point contract.
  - [x] **Skill 15 — Reporter**: Fixed semantic data mapping — metrics now extracted from `config.task_type` (not `config.domain`). Eliminated initialization-stage writes to long-term `history_log.jsonl` by routing startup events to session-scoped files under `reports/sessions/startup_{timestamp}.jsonl`.
  - [x] **Skill 17 — Governance**: Fixed gate key validation to check `human_gate_N_approved` (was checking non-standard `gate_N_timestamp`). Added structural lock on `state["selected_submissions"]` via `state["selected_submissions_final"]` boolean sentinel. Removed cross-skill import of `skill_22_reproducibility_audit`. Applied safe state access pattern (`.get()`) throughout. Conformed to `run(config, state) -> dict` entry-point contract.
  - [x] **Repository Infrastructure**: Added MIT `LICENSE` file to root. Synced `requirements.txt` via `pip-compile`.
  - [x] **Workspace Rules**: Created `docs/workspace_rules.md` (1013 lines, 20 sections) capturing all naming conventions, import rules, entry-point contracts, config/state access patterns, phase architecture, test conventions, CI/CD, and repository hygiene rules.
- Evidence:
  - Policy test suite (`test_cross_skill_policy`, `test_oof_schema`, `test_cv_policy`, `test_challenge_config_write_policy`, `test_skill_state_safe_access`, `test_skill_coverage`): **27 passed, 0 failed**
  - State safe access test (`test_no_state_bracket_reads_in_skills`): **PASSED** (after fixing `state["selected_submissions_locked_at"]` → `.get()`)
  - Compatibility alias `run_governance` restored for legacy importers
  - Seed discipline test (`test_seed_discipline`): pre-existing failure in dirty tree, passes on clean commit
  - `test_fetch_guard`: pre-existing failure unrelated to Batch 24
- Files Changed (Batch 24 scope):
  - `LICENSE` (new — MIT)
  - `docs/workspace_rules.md` (new — 1013 lines)
  - `zindian/skills/skill_06_cleaning.py` (refactored: MNAR pass, MCAR fold-derived, dynamic constants)
  - `zindian/skills/skill_15_reporter.py` (refactored: task_type mapping, session-scoped logging)
  - `zindian/skills/skill_17_governance.py` (refactored: gate keys, structural lock, no cross-skill import)
  - `docs/refactor_reports/sot_checklist.md` (this entry)
- Commit Message: `docs(sot): Batch 24 update — final operational core hardening.`

## Batch 25 — Phase 4 Governance Three-Lens Integration & Documentation Alignment (2026-06-12)
- Commit: *(pending)*
- Scope: Fully implement the missing Phase 4 (Governance) checks in the Three-Lens evaluation framework and align all checklists and documentation across the codebase.
- Completed:
  - [x] **Three-Lens Phase 4 Implementation**: Implemented `_eval_phase4_general`, `_eval_phase4_specific`, and `_eval_phase4_generalisation` in `zindian/three_lens.py` with strict checks for Gate 2 key flattening, governance report presence and schema checks, structural lock Sentinel, and shared history logs.
  - [x] **Robust Key Access & Path Safety**: Designed safe dictionary/SkillStateStore wrapper functions (`_get_state_val`, `_get_state_keys`) and path resolution utilizing configuration-derived roots to prevent failures across varying environments.
  - [x] **Expanded Test Suite**: Integrated 9 dedicated test cases into `tests/test_three_lens.py` to cover PASS/FAIL states for Gates 1-5 approvals, budget limits, structural lock presence, report schema corruption, and missing history logs.
  - [x] **Checklist and Documentation Alignment**: Batch updated all checklists inside `docs/source_of_truth.md`, `AGENTS.md`, and updated `docs/orchestrator_current_state.md` to correctly represent Phase 4 checks.
- Evidence:
  - Phase 4 three-lens tests, supported phase lists, and the full test suite pass 100% cleanly: **161 passed, 6 skipped**
  - Zero AutoML imports or cross-skill dependency violations.
- Files Changed:
  - `zindian/three_lens.py` (Phase 4 checkers and mapping updates)
  - `tests/test_three_lens.py` (Phase 4 test cases and phase lists)
  - `docs/refactor_reports/sot_checklist.md` (this entry)
- Commit Message: `docs(sot): Batch 25 update — Phase 4 three-lens integration & documentation alignment`