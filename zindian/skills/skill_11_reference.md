Skill 11 — Branch Gate (reference)
=================================

Purpose
-------
- Promote the best passing feature variant to a new anchor branch.
- Block promotion when no variant passed the gate.

Primary implementation
----------------------
- `zindian/skills/skill_11_gate.py`

Commands
--------
- Run the promotion gate:

  python3 -m zindian.skills.skill_11_gate

What it writes
--------------
- Creates or checks out `anchor-v<N>` via git.
- Updates `competitions/<slug>/SKILL_STATE.json` with the promoted anchor metadata.

Current behavior notes
----------------------
- Reads the active metric from `challenge_config.json`.
- Uses the metric-aware score keys when comparing the best variant against the current anchor.
- Resets feature-round counters after promotion.

Findings
--------
- The comparison logic is metric-aware, but the state update still writes AUC-specific anchor fields.
- This is a gate and branch-management skill, not a training skill.
- If `variants_passed` is zero, promotion is blocked immediately.

Notes
-----
- Use this only after Skill 07 has produced at least one passing variant.
