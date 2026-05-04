---

## description: "Skill 04 — Violation EDA (domain-conditional)"

## Goal

Perform EDA only to detect violations/leakage patterns, guided by `challenge_config.domain`.

## Rules

- Never apply physical/solar constraints unless `challenge_config.domain` confirms it.
- Flag suspicious columns (IDs, timestamps, future info) for Skill 10.

## Output

- Persist findings summary to `competitions/<slug>/reports/`.
- Advance `dag_phase` to `phase_2_eda_complete`.