---
description: "Skill 06 — Feature Engineering (domain-conditional)"
---

## Rules
- Never introduce external data unless `challenge_config.allowed_external_data` is true.
- Must run leakage scan (Skill 10) on newly created features.

## Output
- Save generated features to `competitions/<slug>/data/processed/` (gitignored).
- Log feature count and description to `competitions/<slug>/reports/experiments.json`.

