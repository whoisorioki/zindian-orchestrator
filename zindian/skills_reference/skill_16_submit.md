Skill 16 — Zindi Submission
===========================


Purpose
-------
- Validate a submission file.
- Require a human gate before sending the submission to Zindi.
- Log the submission and fetch post-submit leaderboard information.

Primary implementation
----------------------
- `zindian/skills/skill_16_submit.py`

Commands
--------
- Submit a file through the governance flow:

  python3 -m zindian.skills.skill_16_submit path/to/submission.csv

- Show the submission board:

  python3 -m zindian.skills.skill_16_submit --submission-board

What it writes
--------------
- `competitions/<slug>/reports/submission_log.md`
- `competitions/<slug>/SKILL_STATE.json`

Current behavior notes
----------------------
- Validates structural alignment of the submission file against the sample submission using a canonical 8-check flow.
- Prompts for a human YES/NO before submitting.
- Uses `SampleSubmission.csv` as the reference layout.
- Fetches rank and leaderboard info after submission when available.

Audit findings & resolution status
----------------------------------
- **Validation Flow Mismatch**: [RESOLVED] Refactored the validation sequence to perform the canonical 8-check structural alignment sequence:
  1. Column layout check (columns match `SampleSubmission.csv` exactly).
  2. Row count check (matches `SampleSubmission.csv` exactly).
  3. ID column presence in submission.
  4. ID column presence in SampleSubmission.
  5. ID values set match check (values match SampleSubmission).
  6. ID values order match check (order matches SampleSubmission).
  7. Nulls check (no null values).
  8. Duplicate IDs check (no duplicate IDs in the ID column).
- **Submission Comments**: [RESOLVED] Uses metric-aware vocabulary (e.g. `oof_f1` for classification) in comments rather than hardcoded metrics.
- **Budget Guard**: Enforces submission limits before triggering the human validation gate.
