Skill 00 — Zindi Monitor (reference)
===================================

Purpose
-------
- Canonical monitor that gathers competition intelligence, discussion flags, leaderboard and submission board data.
- Produces machine- and human-readable outputs consumed by downstream skills (notably Skill 03 legality gate).

Primary implementation
----------------------
- `zindian/skills/skill_00_zindi_monitor.py` (canonical)

Inputs
------
- Competition slug from `competitions/<slug>/challenge_config.json`
- Zindi auth from environment (`.env`) / `ZINDI_API_KEY` or CLI fallback

Outputs
-------
- `competitions/<slug>/reports/zindi_monitor.json` (machine-readable)
- `competitions/<slug>/reports/compliance_log.md` (human-readable)
- Updates `competitions/<slug>/SKILL_STATE.json` compliance, rank, remaining submissions

Commands
--------
- Run monitor (session start):

  python3 -m zindian.skills.skill_00_zindi_monitor

Notes
-----
- `skill_00_discussion_monitor.py` remains as a compatibility shim for older references, but `skill_00_zindi_monitor.py` is the canonical implementation.
- Use this reference file as the canonical pointer when auditing Skill 00.

Checklist (audit)
-----------------
- Confirms metric and `use_probabilities`.
- Detects external data hints and banned features in discussions.
- Writes `zindi_monitor.json` and `compliance_log.md`.
- Updates SKILL_STATE.json with `compliance` and `remaining_submissions`.

If you want the legacy, discussion-only monitor restored, tell me and I will re-add a compact wrapper.
