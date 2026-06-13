Skill 14 — Inference / Post-processing (reference)
==================================================

Purpose
-------
- Applies task-aware post-processing transformations (probability clipping, boundary clipping, integer rounding) to candidate submission files.
- Reindexes the final predictions to match the canonical row order of `SampleSubmission.csv`.
- Verifies compliance with formatting guidelines before writing the output atomically.

Primary implementation
----------------------
- `zindian/skills/skill_14_inference.py`

Inputs
------
- Candidate submission CSV file.
- `ChallengeConfig` containing:
  - `task_type`: "classification" or "regression".
  - `use_probabilities`: bool.
  - `target_domain_bounds`: dict containing `min` and `max` bounds (for regression).
  - `sample_submission_filename` or `SampleSubmission.csv` template.
  - `id_column` and `target_col`.
- Prerequisite: Human Gate 4 approval timestamp in `SKILL_STATE.json` (`human_gate_4_approved == True`).

Outputs
-------
- Atomically writes post-processed submission file named `post_<original_name>.csv` in the same directory.
- Updates `SKILL_STATE.json` keys: `last_inference_path`, `last_inference_at`, and `last_updated` (when not in dry-run mode).

Commands
--------
- Run post-processing on a submission:
  ```bash
  python3 -m zindian.skills.skill_14_inference <submission.csv>
  ```
- Run in dry-run mode (checks without writing):
  ```bash
  python3 -m zindian.skills.skill_14_inference <submission.csv> --dry-run
  ```

Audit findings & resolution status
----------------------------------
- **Strict Human Gate 4 Check**: [RESOLVED] Refactored Human Gate 4 verification to be coercion-safe, explicitly accepting boolean `True`, dictionary configurations (`{"approved": True}`), and valid ISO timestamp strings.
- **Bound Enforcement on Regression**: Checked and validated regression domain bounds, enforcing fallbacks and fail-closed bounds checking when post-processing continuous domains.
