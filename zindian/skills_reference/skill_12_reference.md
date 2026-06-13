Skill 12 — Metric Analysis (reference)
=====================================

Purpose
-------
- Compute validation fold score variance and prepare metrics for gating.
- Unbiased sample variance ($ddof=1$) is required to prevent underestimating variance at low split sizes (e.g. $n=5$).

Primary implementation
----------------------
- `zindian/skills/skill_12_metric.py`

Inputs
------
- Reads `SKILL_STATE.json["eda"]["fold_scores"]` (Note: this is a structural mismatch).
- Reads `target_std` from `SKILL_STATE.json["eda"]["target_std"]` for regression normalization.

Outputs
-------
- Updates `SKILL_STATE.json` with a `"metric_analysis"` block:
  - `fold_scores`: list of fold scores.
  - `fold_score_variance`: float ($ddof=1$).
  - `recommended_threshold`: float (optional, classification).
  - `oof_vs_lb_delta`: float (optional, default null/none).

Commands
--------
- Run metric analysis:
  ```bash
  python3 -m zindian.skills.skill_12_metric
  ```

Audit findings & resolution status
----------------------------------
- **Critical Gap (Input Key Mismatch)**: [RESOLVED] Refactored `skill_12_metric.py` to dynamically resolve the active branch OOF key, extract `fold_scores` from `model_config`, and apply a robust fallback cascade to other OOF keys or legacy fields.
- **Missing SOT Fields**: [RESOLVED] Added logic to populate and write `recommended_threshold` (for classification tasks) and `oof_vs_lb_delta` inside `"metric_analysis"`.
- **Regression Support (Wave 2)**: Enforced scale-invariant and continuous domain gating logic under Wave 2 parameters. Gating normalization by `target_std` has been properly delegated to the consuming gating checks (`skill_11`).
