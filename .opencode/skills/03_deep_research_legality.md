---

## description: "Skill 03 — Deep Research / Legality Check"

## Goal

Verify what’s allowed: external data, external models, code tier constraints.

## Rules

- If `challenge_config.allowed_external_data` is false, do not use external datasets.
- If `challenge_config.automl_permitted` is false, do not use AutoML tools.

## Output

- Update `competitions/<slug>/SKILL_STATE.json.dag_phase` to `phase_1_legality_checked`.
- Write a short note to `competitions/<slug>/reports/submission_log.md` describing constraints discovered.