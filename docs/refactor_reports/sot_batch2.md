Batch 2 — SoT audit & findings (2026-05-24)

Purpose
- Audit repository for Source of Truth (SoT) violations discovered in Batch 1.
- Produce consolidated findings and recommended code changes for full SoT compliance.

Summary of automated scans
- Files scanned: all Python skill modules under `zindian/skills/` and top-level scripts.
- Commands run: repository-wide grep for writes to `challenge_config.json`, direct SKILL_STATE access patterns, and CV split objects.

Findings

1) Writes to `challenge_config.json`
- Allowed writer in Phase 1: `zindian/skills/skill_02_intake.py` (calls `write_config()`) — OK.
- Utility/script writers: `scripts/bootstrap_competition.py` writes `challenge_config.json` during INIT flow — acceptable as a helper.
- No other skill module writes to `challenge_config.json`. PASS (no unauthorized writes found).

Files: 
- zindian/skills/skill_02_intake.py  — writes config (Phase 1) ✓
- scripts/bootstrap_competition.py    — CLI helper (INIT) ✓


2) Unsafe SKILL_STATE bracket access (reading optional keys)
- The SoT mandates safe `.get()` patterns for optional keys such as `cv_strategy_override`, `anchor_challenge`, `pseudo_label_result`, `sidecar_recommendations`, and `drift_threshold`.
- Audit result: code already uses safe `.get()` patterns in key places:
  - `zindian/state.py` exposes safe accessors `resolve_active_cv_strategy_id()` and `is_anchor_challenge_active()`.
  - `skill_11_gate.py`, `skill_21_pseudo_label.py`, and other skills use `state.get(..., {}) .get(...)` patterns where appropriate.
- There are many intentional writes to state via `state["..."] = ...` after reading the state; this is allowed and expected.

Conclusion: No immediate unsafe reads found; remaining direct-bracket uses are writes and acceptable. Recommend a follow-up scan during changes to ensure new code follows `.get()` patterns.


3) Internal CV split objects outside `skill_05` (SoT violation)
- The SoT requires a single CV strategy written by `skill_05_cv` to `challenge_config.json` (and downstream skills should read that). No skill should independently define CV splits.
- Current violations (splits/constructors found outside `skill_05`):
  - `zindian/skills/_lightgbm_shared.py` — creates `StratifiedKFold` internally.
  - `zindian/skills/skill_10_shap.py` — creates `StratifiedKFold` internally.
  - `zindian/skills/skill_21_pseudo_label.py` — creates `StratifiedKFold` internally.
  - `zindian/skills/skill_07_features.py` — uses `StratifiedKFold`.
  - `scripts/audit_section2.py` (utility) — uses `StratifiedKFold` (acceptable as utility).

Recommendation:
- Introduce a single CV factory/helper (e.g. `zindian/cv.py` or add helpers in `zindian/skills/_cv_helpers.py`) that:
  - Reads `challenge_config.json` or `SKILL_STATE` to obtain the chosen CV strategy object (type, n_splits, group_col, seed).
  - Returns either an iterator of `(train_idx, val_idx)` splits or a ready-to-use splitter object.
- Refactor `train_lightgbm_cv`, `skill_10_shap`, `skill_21_pseudo_label`, `skill_07_features` to use the CV helper instead of instantiating local `StratifiedKFold`/`GroupKFold`.
- Ensure `skill_05_cv` writes the full cv_strategy object into `challenge_config.json` on Phase 1 (if not already). If `skill_05` currently writes only to SKILL_STATE, it must be updated to write the `cv_strategy` block to `challenge_config.json` (Phase 1 only).


4) Requirements and reproducibility policy
- `requirements.in` exists.
- No pinned `requirements.txt` found at repo root (scan result: none).
- SoT expects `requirements.txt` to be generated from `requirements.in` via `pip-compile` and committed.

Recommendation:
- Add `requirements.txt` generated from `requirements.in` (run `pip-compile requirements.in --output-file requirements.txt`) and commit.
- Add `scripts/compile_requirements.sh` (exists) as a helper; ensure README/docs instruct maintainers to run it when updating `requirements.in`.


Actionable next steps (Batch 3 plan)
1. Add a CV helper module and update `_lightgbm_shared.py` to accept precomputed splits or to call the helper.
2. Update `skill_05_cv` to write a `cv_strategy` block to `challenge_config.json` (Phase 1) instead of only SKILL_STATE.
3. Refactor `skill_10_shap.py`, `skill_21_pseudo_label.py`, and `skill_07_features.py` to use the shared CV helper.
4. Generate and commit `requirements.txt` from `requirements.in`.
5. Add unit tests that assert no skill instantiates `StratifiedKFold`/`GroupKFold` directly and that `challenge_config.json` contains `cv_strategy` after skill_05 run in INIT mode (mocks allowed).


Would you like me to implement Batch 3 now (I can start by adding the CV helper and updating `_lightgbm_shared.py`), or prefer I make a PR-style patch set for review first?
