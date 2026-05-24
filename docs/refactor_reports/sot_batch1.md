Batch 1 — SoT alignment (2026-05-24)

Summary:
- Created branch `refactor/sot` and committed current workspace changes.
- Scanned repository for SoT violations (cross-skill imports, AutoML imports, challenge_config writes).
- Implemented SoT-aligned `skill_12_metric.py` that computes unbiased fold variance (`ddof=1`) and writes `metric_analysis` to `SKILL_STATE.json`.
- Updated `skill_05_cv.py` to persist the chosen strategy's `fold_scores` to `SKILL_STATE.json` under `eda.fold_scores` so `skill_12` can consume it.

Files changed:
- zindian/skills/skill_12_metric.py
- zindian/skills/skill_05_cv.py

Next steps (proposed Batch 2):
1. Run tests and static checks (`pytest`, lints).
2. Audit all skill modules for direct writes to `challenge_config.json` outside Phase 1.
3. Enforce safe state access patterns across skill bodies (replace direct bracket access with `.get()`).
4. Add `requirements.txt` generation check and ensure `requirements.in` presence.

If you'd like, I can run tests and start Batch 2 now.
