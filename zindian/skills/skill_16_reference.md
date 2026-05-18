Skill 16 — Submission Governance (reference)
===========================================

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
- Validates column order, row count, IDs, ID order, and nulls.
- Prompts for a human YES/NO before submitting.
- Uses `SampleSubmission.csv` as the reference layout.
- Fetches rank and leaderboard info after submission when available.

Findings
--------
- Validation is a 5-check flow here, not the 8-check flow described in Skill 08.
- The submission comment uses `oof_f1`, which is more aligned with the current metric than legacy `oof_rmse` wording.
- Remaining submissions are enforced as a simple budget guard before the human gate.

Notes
-----
- This is the last checkpoint before a live Zindi submission.
