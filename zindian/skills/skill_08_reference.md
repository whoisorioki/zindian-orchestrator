Skill 08 — Anchor Baseline (reference)
=====================================

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
- Prefers the configured target column when present, with `Occurrence Status` as fallback.
- Normalizes feature and prediction arrays before scoring to keep sklearn / LightGBM returns stable.
- Computes an F1-optimized threshold on OOF predictions.
- The submission path is human-gated and should not submit automatically.

Findings
--------
- The training target is still hardcoded as `Occurrence Status` in the current implementation.
- Feature exclusion still hardcodes `Latitude` and `Longitude` rather than reading a layout rule from config.
- The code stores the current anchor metric in legacy `oof_rmse` / `anchor_oof_rmse` fields for compatibility.
- Some type-check warnings remain around prediction arrays unless outputs are normalized carefully.
- The anchor flow still follows the sample submission layout from the competition files, which is the safe default for the data-tab notebook pattern.

Notes
-----
- This is the anchor that downstream feature and gating skills compare against.
- If the team wants full normalization, this is the skill that should be cleaned up next.
