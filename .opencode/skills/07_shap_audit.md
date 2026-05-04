---
description: "Skill 07 — SHAP Audit"
---

## Goal
Compute SHAP feature importance for the current best model and persist the top 20.

## Output
- Write `competitions/<slug>/reports/shap_analysis.json` with:
  - top feature names and importances
  - branch name + run id

