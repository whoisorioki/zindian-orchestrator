Skill 07 — Feature Engineering (reference)
=========================================

Purpose
-------
- Build TerraClimate-based feature sets for the active competition.
- Run one anchor plus isolated feature variants per round.

Primary implementation
----------------------
- `zindian/skills/skill_07_features.py`

Commands
--------
- Fetch TerraClimate data and extract features:

  python3 -m zindian.skills.skill_07_features

- Run a specific variant:

  python3 -m zindian.skills.skill_07_features --variant variant-06

What it writes
--------------
- `competitions/<slug>/data/processed/TerraClimate_14band.tiff`
- `competitions/<slug>/data/processed/features_train.csv`
- `competitions/<slug>/data/processed/features_test.csv`
- `competitions/<slug>/reports/feature_round_<N>.md`
- `competitions/<slug>/SKILL_STATE.json`

Current behavior notes
----------------------
- Uses TerraClimate variables and spatial features.
- Builds isolated variants rather than stacking untested feature changes.
- Target-dependent feature engineering functions must enforce the two-mode contract (fold-restricted during CV, full-train during inference). Structural features (Haversine, count groups) are computed globally.

Audit findings & resolution status
----------------------------------
- **Regression OOF outputs**: [RESOLVED] Secondary metrics block generation is supported for regression.
- **Fold Scores Mismatch**: [RESOLVED] Handled by returning validation fold metrics and appending `fold_scores` directly inside the `model_config` metadata block of the OOF schema. Backward compatibility for mock objects in older test suites is preserved via `getattr` fallbacks.
- **Two-mode contract enforcement**: Enforced two-mode behavior on target-dependent feature encoding during cross-validation split iterations vs final model training.

