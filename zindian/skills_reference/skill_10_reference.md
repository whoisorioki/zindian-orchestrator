Skill 10 — Governed SHAP Audit (reference)
=========================================

Purpose
-------
- Compute fold-level TreeSHAP importances on validation fold predictions to audit and detect features causing potential leaks.
- Perform a correlation-pruning wrapper audit to filter out highly correlated features (>0.95 pearson correlation).

Primary implementation
----------------------
- `zindian/skills/skill_10_shap.py`

Current implementation notes
----------------------------
- Generates `reports/shap_analysis.json` and `reports/shap_summary.md` and updates `SKILL_STATE.json` with SHAP metrics.
- Uses validation-fold predictions for SHAP computation. Full-train SHAP is prohibited to avoid target leakage.
- Calculates correlation coefficient between all feature columns and identifies pairs with correlation >0.95.

Commands
--------
- Run SHAP audit:
  python3 -m zindian.skills.skill_10_shap

Audit findings (issues / misalignments)
------------------------------------
- **Classification Bias**: The current implementation of `_compute_shap_audit` and `run()` hardcodes classification metrics (`roc_auc_score` and `f1_score`). In regression tasks, these functions will crash with ValueError due to continuous target labels.
- **SHAP values dimensions**: LightGBM classifiers yield a list of two SHAP matrices (one per class) when evaluated by `TreeExplainer`, whereas regressors yield a single SHAP matrix. The `_as_positive_shap_values` helper handles this by checking `isinstance(raw_values, list)`, but this needs explicit verification and testing.
- **Correlation Metric**: It currently uses Pearson correlation (`frame[feature_cols].corr()`) for all tasks, whereas the SoT recommends using Pearson for classification and Spearman rank correlation for regression.

Recommendations
---------------
- Branch metrics based on `task_type`: use F1/AUC for classification, and RMSE/R² (and Spearman correlation) for regression.
- Verify SHAP array dimensional consistency dynamically.
- Update `test_probs` file prefix checking to remain consistent under regression names.

Outputs
-------
- JSON report at `reports/shap_analysis.json` and Markdown summary at `reports/shap_summary.md`.
- Updates `SKILL_STATE.json` with keys: `shap_completed_at`, `shap_top_feature`, `shap_top_features`, `high_corr_pairs_count`, `pruning_delta_f1`, `pruning_pass`, and writes OOF record `branch_shap_audit_oof`.
