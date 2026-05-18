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
- Uses TerraClimate variables only.
- Builds isolated variants rather than stacking untested feature changes.
- Uses F1 as the active gate metric even though some comments still mention AUC.
- Reads the anchor score from `SKILL_STATE.json` and blocks if it is missing.

Findings
--------
- Variant naming and report text still contain a few legacy AUC labels.
- The code still references historical variant groups that were designed around Lat/Lon, but current comments mark the compliant TerraClimate-only paths.
- The round report is the main artifact to inspect after a feature batch.

Notes
-----
- Treat this as the controlled feature-engineering entry point. Do not mix multiple unreviewed changes into one variant.
