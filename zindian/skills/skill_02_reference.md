Skill 02 — Challenge Intake (reference & audit)
=============================================

Deep Research Split (Skill 18 / 19 / 20)
----------------------------------------
- Reference docs:
  - `zindian/skills/skill_18_reference.md` for the Librarian track.
  - `zindian/skills/skill_19_reference.md` for the Scientist track.
  - `zindian/skills/skill_20_reference.md` for the Code Miner track.
- `zindian/skills/skill_18_librarian.py`: domain literature retrieval track (the Why).
- `zindian/skills/skill_20_scientist.py`: synthesis track that turns approved evidence into bounded hypotheses.
- `zindian/skills/skill_19_code_miner.py`: machine-learning prior-art retrieval track (the How).
- Keep the tracks separate to avoid context drift and preserve auditable evidence paths.

Summary
-------
- `zindian/skills/skill_02_intake.py` reads the Zindi API and writes `challenge_config.json`.
- Current implementation contains EY-Frogs specific hardcoded defaults (metric, team size, use_probabilities, etc.).

Findings
--------
- Hardcoded values: `metric: "accuracy"`, `use_probabilities: True`, `team_size`, limits, and `name` — these should be parsed from the API response instead of hardcoded.
- `compliance_notes` are hardcoded and include a line `use_probabilities=True: do NOT threshold predictions` which contradicts `zindi_monitor.json` (which indicates `use_probabilities=False` for EY Frogs). Running this skill as-is would overwrite correct `challenge_config.json`.
- `update_skill_state` sets `dag_phase` to `phase_1_integrity` unconditionally. Careful: this advances the DAG and should only be done after a validated intake.

Risk
----
- Running this script will overwrite the authoritative `challenge_config.json` with stale/hardcoded defaults and may corrupt the pipeline's decisioning (metric, allowed data, gating rules).

Recommendations
---------------
- Do NOT run this skill until the hardcoded defaults are removed.
- Update `extract_config()` to parse fields from the API response and only apply safe fallbacks when fields are missing.
- Use `ChallengeConfig` and `validate_challenge_config()` to ensure written config conforms to schema.
- Remove any competition-specific literals (EY names, team size) from the code.
- Only update `dag_phase` when the extracted config has been validated and agreed by the operator.

Commands (do not run until fixed)
--------------------------------
Populate config (fixed version):

  python3 -m zindian.skills.skill_02_intake

Use this reference when patching the intake skill.
