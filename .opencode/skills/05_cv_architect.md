---

## description: "Skill 05 — CV Architect"

## Goal

Choose CV strategy from `challenge_config.data_modality` and rules.

## Rules

- Timeseries CV only if timeseries is detected/declared by config, otherwise KFold/StratifiedKFold as appropriate.
- Metric type must follow `challenge_config.metric` and `metric_direction`.

## Output

- Write chosen CV plan into `competitions/<slug>/reports/experiments.json` (planned run entry).
- Advance `dag_phase` to `phase_2_cv_selected`.