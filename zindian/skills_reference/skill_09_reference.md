Skill 09 — Probability Calibration (reference)
==============================================

Purpose
-------
- Apply probability calibration (Platt scaling or Isotonic regression) to validation/test predictions of classification models.
- Ensure prediction alignment to actual class frequencies.

Primary implementation
----------------------
- `zindian/skills/skill_09_calibration.py`

Current implementation notes
----------------------------
- Automatically skips execution (returns `SKIPPED` status) for regression tasks, since calibration only applies to classification.
- Dynamic CV strategy retrieval, supporting config-defined strategy and active overrides.
- Performs fold-wise cross-calibration to avoid data leakage on validation probabilities, then fits a final calibrator on all OOF predictions to transform test predictions.
- Writes OOF records under a new namespace `calibration_{candidate_branch}` via `write_oof_record`.

Commands
--------
- Platt scaling:
  python3 -m zindian.skills.skill_09_calibration --method platt

- Isotonic calibration:
  python3 -m zindian.skills.skill_09_calibration --method isotonic

Audit findings (issues / misalignments)
------------------------------------
- None. The classification-only guard is robust and prevents runtime failure in regression pipelines.

Outputs
-------
- Calibrated test predictions written to `data/processed/calib_test_probs_<branch>.csv`.
- Updates `SKILL_STATE.json` with calibration metadata (`calibration_method`, `calibration_candidate_branch`) and writes calibrated OOF record under `branch_calibration_<candidate_branch>_oof`.
