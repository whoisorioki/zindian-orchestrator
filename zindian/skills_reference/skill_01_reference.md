Skill 01 — Integrity Audit (reference)
=====================================

Purpose
-------
- Compute and lock MD5 hashes for the target column and raw data files.
- Ensure data integrity before any transformations.

Primary implementation
----------------------
- `zindian/skills/skill_01_integrity.py`

Current implementation notes
----------------------------
- Reads `target_column` and `submission_target_column` from `challenge_config.json` when available, with fallbacks for older workspaces.
- Treats `Latitude`/`Longitude` as optional and warns when they are absent instead of hard-failing.
- Uses a canonical string-concatenation MD5 for the target column and file-level MD5 for raw inputs.
- Preserves `dag_phase` when `SKILL_STATE.json` is already beyond `phase_1_complete`.

Commands
--------
- Initial lock:

  python3 -m zindian.skills.skill_01_integrity

- Re-verify (compare current files to locked hashes):

  python3 -m zindian.skills.skill_01_integrity --re_verify

Audit findings & resolution status
----------------------------------
- **Regression Misalignment (Class Distribution print)**: [RESOLVED] Previously, target value verification and class distribution printing hardcoded binary values (0 and 1). This has been refactored: if `task_type == "regression"`, the skill bypasses binary class calculations and prints continuous target statistics (Min, Max, Mean, Std) to the trace.
- **Cleanup**: [RESOLVED] The `counts` dictionary is handled dynamically, and unused imports are cleaned up.

Outputs
-------
- Updates `competitions/<slug>/SKILL_STATE.json` with keys: `md5_target_hash`, `md5_train_file`, `md5_test_file`, `md5_sample_sub_file`, and sets `dag_phase` to `phase_1_complete`.
- Returns a summary dict with dataset counts and `class_distribution`.

