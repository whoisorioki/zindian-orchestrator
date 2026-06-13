Skill 13 — Ensemble / Oracle Fusion (reference)
==============================================

Purpose
-------
- Fuse the top-performing candidate branches into a blended ensemble prediction.
- Perform multi-lateral correlation pruning to reject collinear models (>0.95 correlation) and prevent redundant model combinations.

Primary implementation
----------------------
- `zindian/skills/skill_13_ensemble.py`
- `zindian/skills/skill_13_oracle_fusion.py`
- `zindian/oracle_fusion_core.py` (Core implementation)

Current implementation notes
----------------------------
- Enforces Human Gate 3 check (`human_gate_3_approved`).
- Collects variants that have passed validation gating and have active Gate 2 approvals (`human_gate_2_{branch}_approved == True`).
- Computes pairwise OOF correlations:
  - Pearson correlation for classification tasks.
  - Spearman rank correlation for regression tasks.
- If two models have correlation >0.95, the lower-performing model is pruned from the ensemble.
- Blends remaining models (up to top 3) using equal weights or simple averages.
- Writes fused predictions back to `data/processed/oracle_train_fused.csv` and test predictions to `data/processed/oracle_test_fused.csv`.

Commands
--------
- Run oracle fusion:
  python3 -m zindian.skills.skill_13_oracle_fusion

Audit findings (issues / misalignments)
------------------------------------
- None. The core implementation has been properly refactored to support Spearman correlation on regression tasks, check metrics directions, and handle continuous regression errors (RMSE, MAE, MSE, R²).

Outputs
-------
- Updates `SKILL_STATE.json` with keys: `oracle_fusion_completed_at`, `oracle_fusion_models` (list of blended branch names), `oracle_oof_score`, and writes fused OOF record `branch_oracle_fusion_oof`.
