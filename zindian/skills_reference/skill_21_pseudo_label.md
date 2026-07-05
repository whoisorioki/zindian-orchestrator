Skill 21 — Semi-Supervised Pseudo-Labeling
==========================================

Purpose
-------
- Semi-supervised learning via high-confidence pseudo-labels on the unlabelled test set.
- Uses an LGB + RF ensemble (50/50 blend) with sample weights.
- Enforces strict split isolation (pseudo-labelled rows appended *only* to training splits, never validation splits).

Primary implementation
----------------------
- `zindian/skills/skill_21_pseudo_label.py`

Inputs
------
- Processed train/test features (`features_train.csv`, `features_test.csv`).
- `ChallengeConfig` containing target/ID columns, reproducibility seed, and CV strategy.
- Optional overrides in `SKILL_STATE.json` for CV strategy.

Outputs
-------
- Appends high-confidence pseudo-labeled data points to training splits of each fold during CV iterations.
- Generates `oof_probs_pseudo_iter{N}.csv` and `test_probs_pseudo_iter{N}.csv` under the reports directory.
- Updates `SKILL_STATE.json` with the canonical `pseudo_label_result` block containing `ran`, `n_pseudo_labels_added`, `retraining_required`, `guard_conditions_met`, `guard_failure_reason`, and `guard_condition_flags` (Boolean flags for `gc1` to `gc6`).
- Writes the retrained OOF predictions to the `branch_{name}_oof_augmented` namespace (raising a RuntimeError if trying to overwrite a non-augmented OOF key).

Commands
--------
- Run pseudo-labeling:
  ```bash
  python3 -m zindian.skills.skill_21_pseudo_label
  ```
- Run in dry-run mode (verifies guards and reports without final updates):
  ```bash
  python3 -m zindian.skills.skill_21_pseudo_label --dry-run
  ```

Audit findings & resolution status
----------------------------------
- **Guard Condition Evaluation Mismatch**: [RESOLVED] Refactored `skill_21` to execute all six guard condition audits (`gc1`–`gc6`) early at the absolute entry point of the script, preventing any LightGBM or RandomForest model training when conditions fail.
- **Strict Task Type Check**: Properly throws a `ValueError` if `task_type != "classification"` (Guard Condition 1), preventing regression tasks from running pseudo-labeling.
- **State Variance Check Mismatch**: [RESOLVED] Corrected the variance checking path to read from `state["metric_analysis"]["fold_score_variance"]` to align with Skill 12's actual outputs.
