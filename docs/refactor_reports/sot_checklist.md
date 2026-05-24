# SoT Compliance Checklist & DoD

## Purpose
- Centralized checklist mapping the Source of Truth (SoT) rules to concrete DoD (Definition of Done) checks.
- Use this file to mark items completed after each refactor batch and to record evidence (commits, files changed, tests).

## How to use
- Before changing code, read the relevant SoT entry in `docs/source_of_truth.md` and this checklist item.
- Implement changes, run tests, and update this checklist with status, date, and commit SHA.
- Use the workspace todo list to track implementation tasks; reference the checklist item id in commits/PRs.

Checklist (grouped by SoT domain)

1) Config & Temporal Lock
- Check: No skill writes to `challenge_config.json` after Phase 1 completes.
  - DoD: grep for writes to `config_path` or `challenge_config.json`; any legitimate writer must be `skill_02_intake.py` or init scripts. Unit test: simulated Phase >1 run fails if write attempted.
  - Verify: manual review of commits; run `grep -R "config_path" -n zindian || true`.
  - Status: pending

2) Safe SKILL_STATE access
- Check: All optional keys read using `.get()` patterns to avoid KeyError (examples in AGENTS.md).
  - DoD: no occurrences of `state["cv_strategy_override"]` or similar bracket reads for optional keys. Automated grep: `grep -R "state\[" -n zindian | grep -v "state\[\"\w+\"\] ="`.
  - Verify: run static search, spot-check changed files, run unit tests that cover state paths.
  - Status: pending

3) CV Strategy & OOF Contracts
- Check: Single CV strategy is authored by `skill_05_cv` in Phase 1 and skills consume via `zindian.cv` helpers.
  - DoD: `challenge_config.json` contains `cv_strategy` block after Phase 1; skills use `zindian.cv.make_cv_splitter()` or accept an injected `cv` parameter. OOF outputs include `cv_strategy_id` and `seed`.
  - Verify: inspect `skill_05_cv.py` for `write` to config, grep for instantiations of `StratifiedKFold|GroupKFold` (should be only in utilities or `skill_05`).
    - Status: completed
    - Completed_by: agent
    - Date: 2026-05-24
    - Evidence: commit 0a84143 (CI policy test); skill_05 updated to persist cv_strategy in this branch.

4) SHAP & Feature Contracts
- Check: SHAP computed per-fold on validation only; no full-train SHAP.
  - DoD: SHAP loops run per-fold and compute mean absolute per-fold then mean across folds as in SoT examples.
  - Verify: unit test for `skill_10_shap.py` runs on a small synthetic dataset and checks output schema.
  - Status: in-progress
  - Notes: added per-fold SHAP unit test (`tests/test_shap_per_fold.py`) and suppressed known LightGBM/SHAP warning in test to maintain CI stability.
  - Evidence: tests added in commit d0d7f96; SHAP warning handling adjusted in latest commit (to be recorded).

5) Seed Discipline
- Check: All model training reads seed from `challenge_config.json` `reproducibility.seed` or a shared config; no local ad-hoc seeds.
  - DoD: grep for literal `random_state=` or `np.random.seed(` and assert they use config or constants; add tests that mock config seed.
  - Status: pending

6) SHARED FILES & No AutoML
- Check: No AutoML libraries imported in `zindian/skills/`.
  - DoD: `grep -R "auto-sklearn\|autogluon\|h2o\|tpot" -n` returns no hits. Any helper libs must be declared in `requirements.in`.
  - Status: passed (manual scan)

7) OOF Output Schema
- Check: OOF-producing skills write OOF outputs in the SoT schema and tag with `cv_strategy_id`.
  - DoD: `SKILL_STATE` contains keys like `branch_{branch_name}_oof` with `scores`, `cv_strategy_id`, `seed`, `branch_name`, `model_config`.
  - Verify: unit tests or schema validator; sample audit of `skill_12_metric.py` and `skill_05_cv.py`.
  - Status: in-progress

8) Tests & CI
- Check: Tests do not perform network calls during collection and are resilient to missing competition artifacts.
  - DoD: All tests pass in a clean environment; network interactions are behind `if __name__ == '__main__'` or mocked in tests.
  - Verify: `pytest -q` in CI and local; tests remain <= expected runtime.
  - Status: passed (31 tests)

9) Packaging & Requirements
- Check: `requirements.txt` is generated from `requirements.in` and committed.
  - DoD: `requirements.txt` exists and `scripts/compile_requirements.sh` documents generation commands.
  - Verify: run `bash scripts/compile_requirements.sh` in dev env (optional); CI should assert `requirements.txt` up-to-date.
  - Status: pending


Per-batch DoD process
- After each batch (1, 2, 3, ...), update this file under a `## Batch X` section with:
  - Date and commit SHA that implements changes.
  - Items completed (checked), items deferred, and new findings.
  - Links to PRs or commits.
  - Short remediation notes for any unresolved items.

Example Batch entry (fill after batch completes)

## Update rules (who, how)
- Who: the author of the change updates this file as part of the PR; CI reviewers verify DoD items before merging.
- How: Edit `docs/refactor_reports/sot_checklist.md` and commit a small update describing the batch completion. Update the todo list via the agent's managed todo tool.
- Commit message format: `docs(sot): Batch <N> update — <short desc>`

Template for a checklist item
```
- Check: <one-line description>
  - DoD: <succinct acceptance criteria>
  - Verify: <commands/files/tests to run>
  - Status: pending|in-progress|completed
  - Completed_by: <name/agent>
  - Date: <YYYY-MM-DD>
  - Evidence: <commit/PR/file links>
```

## Storage and audit
- Keep this checklist in `docs/refactor_reports/sot_checklist.md` and update after each batch. Reference it in PR descriptions and the `SKILL_STATE.json` where appropriate (e.g., `last_refactor_batch` metadata).

## Batch 1 — Completed (2026-05-24)
- Commit: 3647e8a
- Completed:
  - `skill_12_metric.py` — ddof=1 variance and SKILL_STATE writes. (DoD: unit test and manual inspection)
  - Tests refactor to avoid network calls. (DoD: pytest run)
- Deferred:
  - `skill_05_cv` must write `cv_strategy` into `challenge_config.json` during Phase 1. (Completed in this batch)
  - Additional commits: 0a84143 (CI/test policy updates), 93ae6c9 (skill_05 cv_strategy write + docs)
  - Evidence: tests passing (`pytest -q` -> 32 passed), commit SHAs: 3647e8a, 0a84143, 93ae6c9

## Batch 2 — In Progress (2026-05-24)
- Scope: Centralize CV factory, refactor LightGBM shared training, enforce CV instantiation policy, add tests (OOF schema, CV factory, SHAP per-fold), and add requirements fallback.
- Commits: 3647e8a, 0a84143, 93ae6c9, d0d7f96, 0a84143, d760bef, d0d7f96, 3033e1e
- Completed:
  - `zindian/cv.py` added — central CV factory (DoD: reviewed code + tests that exercise the factory).
  - `_lightgbm_shared.py` updated to accept `cv` param and use splitter (DoD: code change committed, training helpers use factory).
  - Multiple skills refactored to use `zindian.cv` (skill_10_shap, skill_07_features, skill_21_pseudo_label, skill_08_anchor).
  - CI policy test `tests/test_cv_policy.py` added and tightened to ignore virtualenv and site-packages (DoD: test passes in local run).
  - Unit tests: `tests/test_oof_schema.py`, `tests/test_cv_factory.py`, `tests/test_shap_per_fold.py` added and passing locally.
  - `requirements.txt` fallback added to repo.
- Remaining / In-progress:
  - SHAP output schema test to validate `skill_10_shap` artifact format (in progress — test added).
  - Finalize Batch 2 PR and run CI in upstream environment.

Evidence: local test run `pytest -q` -> 36 passed, commits present on branch `refactor/sot`.
