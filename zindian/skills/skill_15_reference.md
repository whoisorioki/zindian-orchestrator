Skill 15 — Reporter (reference)
==============================

Purpose
-------
- Initialize the DuckDB ledger if needed.
- Generate a phase summary report from the current competition state.

Primary implementation
----------------------
- `zindian/skills/skill_15_reporter.py`

Commands
--------
- Generate the phase summary:

  python3 -m zindian.skills.skill_15_reporter

What it writes
--------------
- `competitions/<slug>/reports/phase_1_summary.json`
- `competitions/<slug>/reports/experiments.db` when the ledger is initialized through the reporter path

Current implementation notes
----------------------------
- Resolves `config_path`, `state_path`, and `reports_dir` from the active competition paths.
- Reads the current competition config and state before generating the summary.
- Safely handles missing ledger tables by reporting zero counts.
- Incorporates `integrity_audit.json` into the generated phase summary when present.

Current behavior notes
----------------------
- Reads `challenge_config.json`, `SKILL_STATE.json`, and the DuckDB ledger.
- Summarizes experiment and submission counts.
- Incorporates integrity audit data if `integrity_audit.json` exists.

Findings
--------
- This is a reporting utility rather than a modeling or gating skill.
- It is useful as a phase-1 handoff artifact because it ties config, state, and ledger status together.

Notes
-----
- Use this when you want a compact status snapshot for the competition workspace.
