Skill 08 — Anchor Baseline
==========================

Purpose
-------
- Train the first anchor baseline.
- Validate the submission file, then require a human YES/NO prompt before submission.

Primary implementation
----------------------
- `zindian/skills/skill_08_anchor.py`

Commands
--------
- Train the anchor baseline without submitting:

  python3 -m zindian.skills.skill_08_anchor

- Train and trigger the human-gated submission flow:

  python3 -m zindian.skills.skill_08_anchor --submit

What it writes
--------------
- `competitions/<slug>/data/raw/oof_anchor.csv`
- `competitions/<slug>/submissions/sub_###_anchor.csv`
- `competitions/<slug>/SKILL_STATE.json`
- `competitions/<slug>/reports/` ledger entries via Skill 15 / DuckDB integration

Current behavior notes
----------------------
- Reads `features_train.csv` and `features_test.csv` from the processed data folder.
- Uses `SampleSubmission.csv` to infer the submission column name.
- Prefers the configured target column from config, with a fallback.
- Computes F1-optimized threshold for classification, or directly evaluates RMSE/RMSLE for regression.
- Supports anchor challenge mode where a baseline model can be evaluated and challenged via Gate 1 ([C] selected).

Audit findings & resolution status
----------------------------------
- **Regression and Secondary Metrics**: [RESOLVED] Supported in anchor training, storing secondary metrics under `secondary_metrics` in regression tasks.
- **Fold Scores Mismatch**: [RESOLVED] Individual fold score telemetry is successfully collected and written inside the OOF schema (`fold_scores` inside `model_config`) for consumption by downstream diagnostic layers.
- **Exclusion Hardcoding**: Enforced dynamic configuration lookups instead of hardcoding coordinate names.
