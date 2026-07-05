Skill 14 — Inference / Post-processing
======================================

Purpose
-------
- Apply task-aware post-processing rules to submission files.
- Validate submission format and value constraints.
- Enforce probability intervals, binary labels, or regression bounds.
- Ensure atomic file writes to prevent partial submissions.

Primary implementation
----------------------
- `zindian/skills/skill_14_inference.py`

Current implementation notes
----------------------------
- Requires Human Gate 4 approval before running (SoT §4 / §8).
- Reads task_type, use_probabilities, and target_domain_bounds from config.
- Resolves ID column dynamically (no hardcoded "ID" string).
- Reindexes submission to match SampleSubmission.csv ordering.
- Applies task-specific validation and clipping.
- Writes post-processed submission atomically (tempfile + os.replace).
- Never writes to challenge_config.json (Phase 1 temporal lock).

Commands
--------
- Post-process a submission:

  python3 -m zindian.skills.skill_14_inference submissions/sub_001_anchor.csv

- Dry run (validate without writing):

  python3 -m zindian.skills.skill_14_inference submissions/sub_001_anchor.csv --dry-run

What it writes
--------------
- `competitions/<slug>/submissions/post_<original_name>.csv`
- `competitions/<slug>/SKILL_STATE.json`
  - `last_inference_path`: path to post-processed submission
  - `last_inference_at`: ISO timestamp
  - `last_updated`: ISO timestamp

Inputs
------
- Submission CSV file (path provided as argument)
- `competitions/<slug>/data/raw/SampleSubmission.csv` (for format validation)
- `competitions/<slug>/challenge_config.json` (for task_type, use_probabilities, target_domain_bounds)
- `competitions/<slug>/SKILL_STATE.json` (for Human Gate 4 approval)

Outputs
-------
- Post-processed submission file with prefix `post_`
- Updated SKILL_STATE.json with inference metadata

Behavior & Safety
-----------------
- **Human Gate 4 prerequisite**: Halts immediately if `human_gate_4_approved` is not True.
- **Format enforcement**: Reindexes to match SampleSubmission.csv row order and columns.
- **Classification (probabilities)**: Asserts all values in open interval (0, 1).
- **Classification (hard labels)**: Asserts all values are 0 or 1.
- **Regression**: Clips values to target_domain_bounds [min, max].
- **Log1p support**: Applies log1p transformation if `submission_log1p=True` in config.
- **Atomic writes**: Uses tempfile + os.replace to prevent partial file writes.
- **No config mutation**: Never writes to challenge_config.json after Phase 1.

Validation rules
----------------
1. **ID column**: Must match between submission and SampleSubmission.csv
2. **Target column**: Must be present and numeric
3. **Row count**: Must match SampleSubmission.csv exactly
4. **Column order**: Reordered to match SampleSubmission.csv
5. **Value constraints**:
   - Classification (probs): 0 < value < 1 (strict)
   - Classification (labels): value ∈ {0, 1}
   - Regression: min ≤ value ≤ max (clipped)
6. **No NaN/Inf**: All values must be finite

Audit findings & resolution status
----------------------------------
- **Initial implementation**: No known issues at time of reference creation.
- **Human Gate 4 enforcement**: Correctly blocks execution without approval.
- **Dynamic ID resolution**: No hardcoded "ID" string literals.
- **Atomic writes**: Prevents partial file corruption.
- **Regression bounds**: Correctly clips to target_domain_bounds.

Notes
-----
- This skill is the final validation step before Skill 16 (submission).
- Always run on the fusion output (Skill 13) or best single variant.
- Dry run mode is useful for debugging format issues.
- Post-processed files are prefixed with `post_` to distinguish from raw submissions.
- The skill never modifies the original submission file.

Recommendations
---------------
- Always run Skill 14 before Skill 16 (submission).
- Use dry run mode to validate format before writing.
- Check post-processed file manually before submitting to leaderboard.
- For regression, verify target_domain_bounds are set correctly in config.
- For classification, confirm use_probabilities matches competition requirements.

Error handling
--------------
- **Missing Human Gate 4**: Raises RuntimeError with clear message.
- **Missing submission file**: Raises FileNotFoundError.
- **Missing SampleSubmission.csv**: Raises FileNotFoundError with config hint.
- **Format mismatch**: Raises ValueError with column details.
- **Value constraint violation**: Raises ValueError with range details.
- **Non-numeric values**: Raises ValueError with type conversion error.

Integration with other skills
------------------------------
- **Skill 13 (Oracle Fusion)**: Produces input submission file.
- **Skill 16 (Submit)**: Consumes post-processed submission file.
- **Human Gate 4**: Must be approved before this skill runs.
- **Skill 02 (Intake)**: Populates task_type and target_domain_bounds.
