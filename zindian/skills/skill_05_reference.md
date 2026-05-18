Skill 05 — CV Architect (reference)
===================================

Purpose
-------
- Compare cross-validation strategies for the active competition.
- Write the chosen CV strategy into `SKILL_STATE.json` for downstream use.

Primary implementation
----------------------
- `zindian/skills/skill_05_cv.py`

Commands
--------
- Compare strategies:

  python3 -m zindian.skills.skill_05_cv

- Force stratified CV:

  python3 -m zindian.skills.skill_05_cv --strategy=stratified

- Force spatial CV:

  python3 -m zindian.skills.skill_05_cv --strategy=spatial

What it writes
--------------
- `competitions/<slug>/SKILL_STATE.json`
  - `cv_strategy`
  - `cv_primary_metric`
  - `cv_stratified_oof_auc`
  - `cv_spatial_oof_auc`
  - `cv_gap`
  - `dag_phase` advances to `phase_3_features` only from early phases

Current behavior notes
----------------------
- Reads the active metric from `challenge_config.json` and uses it to score strategy quality.
- Uses processed `features_train.csv` from Skill 07.
- Falls back to `Occurrence Status` as the target name when no config override is present.
- Spatial CV requires latitude and longitude columns in the feature table.

Findings
--------
- The code is metric-aware now, but several variable names and legacy state fields still mention AUC.
- Spatial mode still depends on coordinate columns being present in `features_train.csv`.
- This is a comparison skill, not a submission skill; it should be run before feature work is locked in.

Notes
-----
- Use this reference when deciding whether the project should continue with stratified or spatial CV.
