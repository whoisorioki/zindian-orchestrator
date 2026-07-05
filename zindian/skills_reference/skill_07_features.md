Skill 07 — Feature Engineering
==============================

Purpose
-------
- Extract competition-specific features using configured plugin.
- Run one anchor plus isolated feature variants per round.
- Build hypothesis-derived features from challenge_config.json.

Primary implementation
----------------------
- `zindian/skills/skill_07_features.py`

Plugin architecture
-------------------
Skill 07 uses a plugin-based system for feature extraction:
- Plugin path is specified in `challenge_config.json` under `feature_extraction_plugin`
- Plugin must implement `extract(paths, tiff_path, config) -> (train_df, test_df)`
- Optional `fetch(paths, config, allow_network=True)` method for data retrieval
- No competition-specific logic in the skill itself

Example plugin configuration:
```json
{
  "feature_extraction_plugin": "plugins.terraclimate_extractor",
  "feature_engineering": {
    "polynomials": ["feature1", "feature2"],
    "interactions": [["feature1", "feature2"]],
    "ratios": [["feature1", "feature2"]],
    "conditions": [
      {"column": "feature1", "operator": "gt", "value": 10, "name": "feature1_high"}
    ],
    "target_dependent_bins": [
      {"column": "feature1", "q": 10, "name": "feature1_bin_target_mean"}
    ]
  }
}
```

Commands
--------
- Fetch data and extract features (plugin-dependent):

  python3 -m zindian.skills.skill_07_features --fetch

- Run a specific variant:

  python3 -m zindian.skills.skill_07_features --variant variant-06

- Force save submission even if gate fails:

  python3 -m zindian.skills.skill_07_features --variant variant-06 --force-save

What it writes
--------------
- `competitions/<slug>/data/processed/features_train.csv`
- `competitions/<slug>/data/processed/features_test.csv`
- `competitions/<slug>/data/processed/oof_<variant>.csv`
- `competitions/<slug>/data/processed/test_probs_<variant>.csv`
- `competitions/<slug>/submissions/<variant>_submission.csv` (if gate passes or --force-save)
- `competitions/<slug>/reports/feature_round_<N>.md`
- `competitions/<slug>/SKILL_STATE.json`

Current behavior notes
----------------------
- Feature extraction is fully delegated to the configured plugin.
- Builds isolated variants rather than stacking untested feature changes.
- Hypothesis-derived features (polynomials, interactions, ratios, conditions, target-dependent bins) are built from config.
- Target-dependent feature engineering functions must enforce the two-mode contract (fold-restricted during CV, full-train during inference). Structural features are computed globally.
- Multi-seed averaging (default 3 seeds) for robust variant evaluation.

Audit findings & resolution status
----------------------------------
- **Regression OOF outputs**: [RESOLVED] Secondary metrics block generation is supported for regression.
- **Fold Scores Mismatch**: [RESOLVED] Handled by returning validation fold metrics and appending `fold_scores` directly inside the `model_config` metadata block of the OOF schema. Backward compatibility for mock objects in older test suites is preserved via `getattr` fallbacks.
- **Two-mode contract enforcement**: Enforced two-mode behavior on target-dependent feature encoding during cross-validation split iterations vs final model training.
