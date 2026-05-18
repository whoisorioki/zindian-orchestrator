Skill 01 — Integrity Audit (reference)
=====================================

Purpose
-------
- Compute and lock MD5 hashes for the target column and raw data files.
- Ensure data integrity before any transformations.

Primary implementation
----------------------
- `zindian/skills/skill_01_integrity.py`

Commands
--------
- Initial lock:

  python3 -m zindian.skills.skill_01_integrity

- Re-verify (compare current files to locked hashes):

  python3 -m zindian.skills.skill_01_integrity --re_verify

Audit findings (issues / misalignments)
------------------------------------
- Hardcoded target column name: `TARGET_COL = "Occurrence Status"`. Prefer reading from `challenge_config.json` when present.
- Submission target column hardcoded as `Target` — may differ across competitions.
- Strict assertions expect `Latitude` and `Longitude` in `Training_Data.csv`. This will fail for competitions without geolocation and is unnecessary for an integrity check; replace with a warning.
- The code asserts target values equal `[0,1]` exactly. Some datasets use strings (e.g., "Present"). The integrity skill should accept any type and only record values and counts; converting to a stable canonical form before hashing is safer.
- MD5 method mismatch: `compute_md5()` uses `pandas.util.hash_pandas_object` -> bytes -> MD5, while other scripts (in instructions) computed MD5 by concatenating stringified values. These two methods produce different hashes; standardize to a single approach to avoid false mismatches.
- `update_skill_state` sets `dag_phase` to `phase_1_complete` unconditionally; this is acceptable for first lock but should check current phase before downgrading on re-verify.

Recommendations
---------------
- Read `TARGET_COL` and `submission_target_col` from `ChallengeConfig` when available.
- Relax Latitude/Longitude assertions to warnings; only assert required files exist.
- Standardize target MD5 computation method across the repo (choose either row-concatenation or pandas hash) and update `SKILL_STATE.json` accordingly.
- On `re_verify`, do not downgrade `dag_phase` if it is already beyond `phase_1_complete`.

Outputs
-------
- Updates `competitions/<slug>/SKILL_STATE.json` with keys: `md5_target_hash`, `md5_train_file`, `md5_test_file`, `md5_sample_sub_file`, and sets `dag_phase` to `phase_1_complete`.
- Returns a summary dict with dataset counts and `class_distribution`.

If you want, I can patch `skill_01_integrity.py` now to implement these recommendations (make hashing consistent and read target name from config). Reply `patch` to proceed.
