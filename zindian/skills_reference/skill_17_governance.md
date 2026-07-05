Skill 17 — Submission Governance
================================

Purpose
-------
- Final human-gated selection of the two submission files for private leaderboard evaluation (Gate 5).
- Verifies that all prior prerequisite gates (1 to 4) have been approved.
- Applies a structural lock to `selected_submissions` to prevent further changes or retraining.

Primary implementation
----------------------
- `zindian/skills/skill_17_governance.py`

Inputs
------
- `ChallengeConfig` containing:
  - `slug`: competition identifier.
- `SKILL_STATE.json` containing:
  - `human_gate_1_approved`
  - `human_gate_2_approved`
  - `human_gate_3_approved`
  - `human_gate_4_approved`
  - `scored_submissions`: list of all candidate submission details (scores, dates, filenames).
  - `selected_submissions`: current selections.

Outputs
-------
- Updates `SKILL_STATE.json` keys:
  - `selected_submissions`: locked list of exactly 2 selected submission dicts.
  - `selected_submissions_locked_at`: ISO timestamp of lock.
  - `selected_submissions_final`: set to `True` (structural lock).
  - `human_gate_5_selection`: dict containing `"approved": True`, selection timestamp, and selections.
- Writes a JSON selection report at `competitions/<slug>/reports/final_selections.json`.

Commands
--------
- Run governance:
  ```bash
  python3 -m zindian.skills.skill_17_governance
  ```

Audit findings & resolution status
----------------------------------
- **Critical Bug (Boolean Type Gate Bypass Error)**: [RESOLVED] Refactored `_verify_prerequisite_gates` and `_verify_final_gate` to dynamically support boolean `True`, dictionary objects (`{"approved": True}`), and valid ISO timestamp strings. This fully aligns the control plane with the testing and orchestration gate requirements.
- **Lock Check**: The structural lock correctly halts any modifications once `selected_submissions_final` is `True`.
