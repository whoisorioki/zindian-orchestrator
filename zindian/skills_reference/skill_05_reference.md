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
  - `cv_strategy`: dictionary with keys `type`, `n_splits`, `shuffle`, `random_state`, `group_col`, and `selection_reason`.
  - `cv_strategy_type`
  - `cv_strategy_selection_reason`
  - `cv_group_col`
  - `dag_phase` advances to `phase_3_features` only from early phases.
- `challenge_config.json` (Phase 1 only, config temporal lock closes after this):
  - Updates the `cv_strategy` block.

Current behavior notes
----------------------
- Reads the active metric and `task_type` from `challenge_config.json`.
- Uses processed `features_train.csv` from Skill 07.
- Falls back to `Occurrence Status` as the target name when no config override is present.
- Spatial CV requires latitude and longitude columns for splitting logic when that strategy is selected; they should not be treated as model features.

Audit findings & resolution status
----------------------------------
- **Regression Target Type / Casting**: [RESOLVED] When `task_type == "regression"`, continuous target labels are loaded as `np.float32` rather than being forced to `int32`, preventing any decimal loss.
- **Regression Strategy Decision**: [RESOLVED] For regression tasks, when standard `KFold` is chosen, the selection reason is correctly recorded as `"standard regression strategy chosen for continuous target"`.
- **Target Logging & Distribution**: [RESOLVED] Adjusted logging in `build_spatial_splits` and `run()` to display target mean and min/max range instead of "class prevalence" or "positive rate" when the task is regression.
- **Coordinate Dependencies**: Spatial CV requires valid coordinates in the dataset and groups them using KMeans. If coordinates are banned by policy, spatial CV is unusable.

Recommendations
---------------
- Ensure `challenge_config.json` is in Phase 1 before running `skill_05` if you want it to write to config. Post-Phase-1, it is temporally locked.
