Skill 02 — Challenge Intake (reference & audit)
=============================================

Deep Research Split (Skill 18 / 19 / 20)
----------------------------------------
- Reference docs:
  - `zindian/skills_reference/skill_18_librarian.md` for the Librarian track.
  - `zindian/skills_reference/skill_19_code_miner.md` for the Code Miner track.
  - `zindian/skills_reference/skill_20_scientist.md` for the Scientist track.
- `zindian/skills/skill_18_librarian.py`: domain literature retrieval track (the Why).
- `zindian/skills/skill_20_scientist.py`: synthesis track that turns approved evidence into bounded hypotheses.
- `zindian/skills/skill_19_code_miner.py`: machine-learning prior-art retrieval track (the How).
- Keep the tracks separate to avoid context drift and preserve auditable evidence paths.

Summary
-------
- `zindian/skills/skill_02_intake.py` reads the Zindi API and writes `challenge_config.json`.
- Current implementation contains EY-Frogs specific hardcoded defaults (metric, team size, use_probabilities, etc.).

Audit findings & resolution status
----------------------------------
- **Hardcoded Defaults & API Extraction**: [RESOLVED] Removed EY-Frogs specific hardcoded defaults. The skill now dynamically extracts `metric`, `use_probabilities`, `team_allowed`, and `max_team_size` from the Zindi API response. It also dynamically derives `task_type` and `use_probabilities` based on the metric type (e.g. classification for `logloss`/`auc`, regression for `rmse`/`mae`).
- **Null Safety / Config Schema**: [RESOLVED] Implemented safe defaults for `allowed_external_data` (default `False`), `automl_permitted` (default `False`), and `target_domain_bounds` (initialized to `{"min": None, "max": None}`) if missing from the API payload.
- **Intake Validation**: [RESOLVED] Integrates the schema validation utility `validate_challenge_config()` directly before writing the configuration to disk, ensuring schema conformance.
- **Unconditional DAG Phase Advancement**: [RESOLVED] Refactored the state update logic to prevent advancing the `dag_phase` to `phase_1_integrity` if `task_type` or `target_col` are still unconfigured (`None`/`null`), ensuring the pipeline cannot advance without operator validation of critical target keys.

Commands
--------
Populate challenge configuration:

  python3 -m zindian.skills.skill_02_intake
