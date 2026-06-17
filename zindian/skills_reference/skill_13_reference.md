Skill 13 — Oracle Fusion / Ensemble (reference)
==============================================

Purpose
-------
- Perform out-of-fold blending and multi-lateral correlation pruning.
- Combine multiple model predictions using weighted averaging or stacking.
- Prune highly correlated predictions to reduce redundancy.

Primary implementation
----------------------
- `zindian/skills/skill_13_oracle_fusion.py` (canonical)
- `zindian/skills/skill_13_ensemble.py` (compatibility shim)
- Core logic: `zindian/oracle_fusion_core.py`

Architecture notes
------------------
Both `skill_13_oracle_fusion.py` and `skill_13_ensemble.py` are thin wrappers that delegate to `zindian.oracle_fusion_core`. They exist for:
1. **Backward compatibility**: Legacy code may reference `skill_13_ensemble`
2. **Naming clarity**: "Oracle Fusion" better describes the correlation-pruning behavior
3. **Test isolation**: Allows monkeypatching of imports for testing

The canonical name is **Oracle Fusion** (skill_13_oracle_fusion.py), but both modules provide identical functionality.

Current implementation notes
----------------------------
- Reads OOF predictions from multiple branches in SKILL_STATE.json.
- Computes pairwise correlations between OOF predictions.
- Prunes highly correlated predictions (keeps best performer).
- Blends remaining predictions using weighted averaging.
- Writes fused OOF and test predictions.
- Supports both classification and regression tasks.

Commands
--------
- Run oracle fusion:

  python3 -m zindian.skills.skill_13_oracle_fusion

- Dry run (preview without writing):

  python3 -m zindian.skills.skill_13_oracle_fusion --dry-run

- Legacy ensemble command (equivalent):

  python3 -m zindian.skills.skill_13_ensemble

What it writes
--------------
- `competitions/<slug>/data/processed/fusion_oof.csv`
- `competitions/<slug>/data/processed/fusion_test.csv`
- `competitions/<slug>/reports/fusion_report.md`
- `competitions/<slug>/SKILL_STATE.json`
  - `fusion_completed_at`: ISO timestamp
  - `fusion_branches`: list of branches included in fusion
  - `fusion_weights`: dict mapping branch names to blend weights
  - `fusion_pruned_branches`: list of branches pruned due to high correlation
  - `fusion_oof_score`: OOF score of fused predictions
  - `branch_fusion_oof`: OOF record for fused predictions

Inputs
------
- `competitions/<slug>/SKILL_STATE.json` (for OOF records from multiple branches)
- `competitions/<slug>/data/processed/test_probs_<branch>.csv` (test predictions from each branch)
- `competitions/<slug>/data/processed/features_train.csv` (for target labels)

Outputs
-------
- Fused OOF and test prediction files
- Fusion report with correlation matrix and weights
- Updated SKILL_STATE.json with fusion metadata

Behavior & Safety
-----------------
- **Correlation pruning**: Removes predictions with correlation > threshold (default 0.95).
- **Weight optimization**: Uses OOF scores to compute optimal blend weights.
- **Diversity preservation**: Keeps diverse predictions even if individually weaker.
- **Regression support**: Works for both classification and regression tasks.
- **Atomic writes**: Uses tempfile + os.replace for safe file operations.

Audit findings & resolution status
----------------------------------
- **Initial implementation**: No known issues at time of reference creation.
- **Dual module architecture**: Documented as intentional for backward compatibility.
- **Core delegation**: Both skill modules correctly delegate to oracle_fusion_core.

Notes
-----
- Oracle Fusion should run after multiple variants have been trained and validated.
- Correlation threshold can be configured in challenge_config.json.
- Fusion typically improves OOF score by 0.001-0.005 over best single model.
- Always validate fused submission on leaderboard before final selection.

Recommendations
---------------
- Run after at least 3-5 diverse variants have passed the gate.
- Compare fused submission against best single variant.
- Use correlation pruning to avoid overfitting to OOF set.
- Monitor fusion weights to ensure no single model dominates (>0.7 weight).
