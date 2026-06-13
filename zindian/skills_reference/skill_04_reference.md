Skill 04 — EDA / Data Quality Audit (reference)
==============================================

Purpose
-------
- Perform generic, competition-agnostic exploratory data analysis.
- Compute and persist the target column's standard deviation (`target_std`) for scale-invariant regression gating.

Primary implementation
----------------------
- `zindian/skills/skill_04_eda.py`

Current implementation notes
----------------------------
- Automatically detects the target column from config/state.
- Computes unbiased sample standard deviation (`ddof=1`) of the target.
- Identifies missingness patterns (MCAR vs. MNAR), constants, near-zero variance features, high-correlation pairs (>0.95), and PII risks.
- Generates `reports/eda_report.json` and a human-readable `reports/eda_summary.md`.

Commands
--------
- Run EDA:
  python3 -m zindian.skills.skill_04_eda

Audit findings & resolution status
----------------------------------
- **Bug (KeyError risk)**: [RESOLVED] Added a strict check to assert the target column is present in the DataFrame's columns before computing standard deviation, raising a clear `ValueError` on failure and eliminating any `KeyError` risk.
- **Structural Misalignment**: [RESOLVED] Modified the script to write target standard deviation (`target_std`), dead features, high correlation counts, and missingness columns nested inside an `"eda"` dictionary in the skill state store. This correctly satisfies downstream consumption expectations.

Outputs
-------
- Updates `competitions/<slug>/SKILL_STATE.json` with keys: `eda` (nested dict containing `target_std`, `dead_features`, `high_corr_pairs_count`, `eda_completed_at`) and sets `dag_phase` to `phase_1_eda_complete`.
