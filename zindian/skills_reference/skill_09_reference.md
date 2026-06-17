Skill 09 — Probability Calibration (reference)
==============================================

Purpose
-------
- Apply probability calibration to OOF and test predictions.
- Supports Platt scaling (logistic regression) and isotonic regression methods.
- Fits calibrators fold-wise to prevent overfitting.

Primary implementation
----------------------
- `zindian/skills/skill_09_calibration.py`

Current implementation notes
----------------------------
- Reads OOF predictions from SKILL_STATE.json branch records.
- Applies fold-wise calibration to OOF predictions using CV splits.
- Fits global calibrator on full OOF set and applies to test predictions.
- Writes calibrated test probabilities to `data/processed/calib_*.csv`.
- Only applies to classification tasks (skips regression).
- Supports pseudo-label retraining workflow (augmented branches).

Commands
--------
- No calibration (copy original):

  python3 -m zindian.skills.skill_09_calibration --method none

- Platt scaling:

  python3 -m zindian.skills.skill_09_calibration --method platt

- Isotonic regression:

  python3 -m zindian.skills.skill_09_calibration --method isotonic

- Dry run (preview without writing):

  python3 -m zindian.skills.skill_09_calibration --method isotonic --dry-run

What it writes
--------------
- `competitions/<slug>/data/processed/calib_test_probs_<branch>.csv`
- `competitions/<slug>/SKILL_STATE.json`
  - `calibration_method`: "none", "platt", or "isotonic"
  - `calibration_written_at`: ISO timestamp
  - `calibration_candidate_branch`: branch name used for calibration
  - `calibration_candidate_oof_key`: state key for OOF record
  - `calibration_oof_cv_strategy_id`: CV strategy identifier
  - `branch_calibration_<branch>_oof`: OOF record for calibrated predictions

Inputs
------
- `competitions/<slug>/data/processed/features_train.csv` (for target labels)
- `competitions/<slug>/data/processed/test_probs_<branch>.csv` (uncalibrated test predictions)
- `competitions/<slug>/SKILL_STATE.json` (for OOF records and CV strategy)

Outputs
-------
- Calibrated test probability files in `data/processed/`
- Updated SKILL_STATE.json with calibration metadata
- OOF record for calibrated predictions (when method != "none")

Behavior & Safety
-----------------
- **Fold-wise calibration**: Prevents overfitting by fitting calibrators on training folds and applying to validation folds.
- **Global calibrator**: Fits on full OOF set for test prediction calibration.
- **Regression skip**: Returns early with status "SKIPPED" for regression tasks.
- **Branch resolution**: Automatically finds promoted branch from state (best_variant, anchor, or branch_*_oof records).
- **Augmented branch support**: Handles pseudo-label retraining workflow by looking for `_augmented` suffixes.

Audit findings & resolution status
----------------------------------
- **Initial implementation**: No known issues at time of reference creation.
- **Regression compatibility**: Correctly skips calibration for regression tasks.
- **Fold-wise safety**: Implements proper CV-aware calibration to prevent leakage.

Notes
-----
- Calibration is optional and should be evaluated against uncalibrated baseline.
- Isotonic regression is more flexible but can overfit on small datasets.
- Platt scaling assumes sigmoid-shaped calibration curve.
- Use `--dry-run` to preview calibration without writing files.
- Calibration does not change rank order of predictions, only probability values.

Recommendations
---------------
- Run calibration after anchor baseline (Skill 08) is confirmed.
- Compare calibrated vs uncalibrated submissions on leaderboard.
- Use isotonic for larger datasets (>1000 samples), Platt for smaller.
- Always validate calibration improves log loss or Brier score before submitting.
