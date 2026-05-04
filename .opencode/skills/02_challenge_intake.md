---
description: "Skill 02 — Challenge Intake (the Brain)"
---

## Goal
Populate a complete `competitions/<slug>/challenge_config.json` strictly from the competition rules page (no defaults).

## Steps (v2 CLI)
- Ensure competition workspace exists:
  - `python -m zindian_orchestrator init-competition --slug <slug>`
- Intake from an existing config draft (manual copy/paste from rules page):
  - `python -m zindian_orchestrator skill02-intake --slug <slug> --from-existing challenge_config.json`

## Outputs
- `competitions/<slug>/challenge_config.json`
- `competitions/<slug>/SKILL_STATE.json` updated:
  - `competition=<slug>`
  - `dag_phase=phase_1_intake`

