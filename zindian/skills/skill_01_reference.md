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

Audit findings (issues / misalignments)
------------------------------------
- Remaining cleanup: `counts` should consistently use the resolved `target_col` variable instead of a fixed literal.
- Remaining cleanup: the `hash_pandas_object` import should be removed if it is no longer used by the implementation.

Recommendations
---------------
- Keep the current config-aware and warning-based behavior.
- Finish the last cleanup items noted above if the skill is revised again.

Outputs
-------
- Updates `competitions/<slug>/SKILL_STATE.json` with keys: `md5_target_hash`, `md5_train_file`, `md5_test_file`, `md5_sample_sub_file`, and sets `dag_phase` to `phase_1_complete`.
- Returns a summary dict with dataset counts and `class_distribution`.

If you want, I can patch `skill_01_integrity.py` now to implement these recommendations (make hashing consistent and read target name from config). Reply `patch` to proceed.
