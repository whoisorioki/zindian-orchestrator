# Zindian Orchestrator — Resolved Gaps & Audit Trail History

**Document Version:** v2.8  
**Last Updated:** September 2026  
**Paired Document:** `docs/source_of_truth.md`

This document records all architectural gaps, findings, and technical vulnerabilities that have been confirmed resolved by direct code inspection and unit/integration testing. This audit trail is decoupled from `docs/source_of_truth.md` to maintain a lean, actionable Source of Truth.

---

## 1. Resolved Audit Trail Summary

| ID | Description | Resolution |
|---|---|---|
| C1 | Bootstrap `dag_phase` prevented config writes | `"phase_1_integrity_locked"` added to `allowed_write_phases` in skill_02 (L867) and skill_05 (L610); bootstrap sets this phase at L119 |
| GAP-4 | `skill_04_eda` did not write `temporal_index_confirmed` / `group_structure_confirmed` | Resolved v2.3 — both lean booleans now derived from BAND_MM pattern, datetime/monotonicity dtype inference, and <5% cardinality ratio |
| S1/S9 | Bessel's correction underestimation + absolute promotion margins | Implemented 2026-08-03 — `skill_12_metric` L167–182 computes NB-corrected `fold_score_variance_nb` + `se_oof`; `skill_11_gate` L134 applies `max(margin, 1.0 * se_oof)` |
| S2 | MAPE zero-target bias + MASE baseline & routing | Implemented 2026-08-25 — Secondary diagnostic MASE implemented; primary-metric routing key mapped in `skill_08_anchor.py` `metric_map`. Score-space correction tracked separately as F4 residual. |
| S3 | Non-uniform metric scaling / composite weighting | Implemented 2026-08-03 — Inverse-variance weighting in `skill_12_metric` L88–149 and `skill_11_gate` L188–236 |
| S4 | Correlation-based pruning used raw predictions not residuals | Implemented 2026-08-03 — `_prune_collinear()` in `oracle_fusion_core.py` accepts `y_true` and computes error residuals; call site at L687 passes `y_true` |
| S5 | Multi-target pseudo-label recombination policy not enforced | Implemented 2026-08-03 — Both policies (`freeze_unaugmented_targets_at_original`, `block_composite_until_all_targets_augmented_or_none`) enforced at `skill_21_pseudo_label.py` L567 and L1034–1132 |
| C4 | `skill_17` checked flat `human_gate_2_approved` instead of per-branch keys | `skill_17_governance.py` L96–104 now iterates `human_gate_2_*_approved` prefix pattern, explicitly excludes flat legacy key |
| M6 | `skill_11` / `skill_12` might silently fall back to wrong `target_std` key on multi-target | `skill_11_gate` L107–115 reads `target_std` with a fallback scan across all `_std`-suffixed keys in the EDA block |
| DRIFT-3 | Orchestrator had no validated split-skill dispatch mechanism | `zindian/orchestrator.py` L318–358 uses `hasattr` + `inspect.signature` to dispatch split-skill functions with filtered kwargs |
| C2 | Preflight did not validate `feature_policy.json` required keys | Not a preflight responsibility — `policy_gate()` in `skill_03_legality.py` validates required keys at Phase 2A runtime; no preflight change needed |
| S7 (partial) | Preflight wrote to `reports/` root instead of `reports/audits/preflight/` | `scripts/preflight_enforce.py` L877–880 now writes to `reports/audits/preflight/<timestamp>.json` |
| S7 (partial) | `skill_12_metric` did not consume buffered CV splits | Not a gap — `skill_12` reads `fold_scores` from existing OOF state records; it never re-runs CV splits |
| S7 | Spatial CV Buffer: `skill_09` not consuming buffered splits | Implemented 2026-08-24 — `skill_09_calibration.py` L207–210 calls `load_explicit_cv_splits(state)` when explicit splits are present |
| S8 | Fixed pseudo-label thresholding — adaptive quantiles not implemented | Implemented 2026-08-24 — class-wise quantile selection with 0.70 floor at `skill_21_pseudo_label.py` L762–788; `min_pseudo_samples` guard; `pseudo_quantile` config key drives selection |
| S10 | No skill wrote `derived_artifact_fingerprints` | Implemented 2026-08-24 — `write_artifact_fingerprint()` called by skill_06 (L196), skill_07 (L1288, L1884), skill_08 (L497, L827); `skill_22` verifier now has data to check |
| S6 | Multicollinear Leakage Systematic Pairwise MI scan | Implemented 2026-08-24 — pairwise MI scan for top-10 SHAP features in `skill_10_shap.py`. |
| S11 | skill_18/20 legacy root dual-writes | Implemented 2026-08-24 — consolidated all sidecar writes to `reports/diagnostics/` only; updated all consumers and tests. |
| Preflight | Multi-target OOF completeness check not per-branch | Implemented 2026-08-24 — added per-branch OOF completeness count assertion to A7 check in `preflight_enforce.py`. |
| R5 | `telemetry.aggregate` not written | Implemented 2026-08-24 — orchestrator post-phase loop aggregation added; verification checks implemented in `skill_22`. |
| Track 2B | Orphaned `feature_policy.json` at root | Implemented 2026-08-24 — verified via workspace and competition grep that no root writes or reads remain. |
| F3 | Hardcoded seeds in skill_10_shap.py | Resolved 2026-08-25 — Replaced hardcoded seed=42 with config-driven random_seed parameter. |
| F4 (partial) | MASE Routing in target lifecycle | Routing key landed 2026-08-25 — `"mase"` mapped in `skill_08_anchor.py` `metric_map`. **Closed 2026-08-25 (v2.7):** full Option A MASE fold scoring shipped (per-fold MAE / global `MAE_naive_baseline`, `oof_mase` field, hard baseline-assert in `_lightgbm_shared.py`, upstream `ValueError` guard in `skill_08_anchor`). |
| VULN-1 / VULN-2 | Path resolution out-of-tree limitation & os.environ race condition | Resolved 2026-09-01 — `resolve_competition_paths()` supports explicit `competition_dir` argument and eliminates `os.environ` mutations. |
| VULN-3 | `policy_writer()` automl_permitted config overwrite | Resolved 2026-09-01 — Enforced strict boolean `AND` between `config.get("automl_permitted")` and `not comp.get("automl_banned")`. |
| Finding 3 | Multi-target SHAP flat state backfill gap | Resolved 2026-09-01 — `_run_multi_target_shap()` backfills primary target SHAP results to `shap_top_features`, `shap_top_feature`, and `shap_feature_count`. |
| Finding 5 | Pseudo-label counter state field naming ambiguity | Resolved 2026-09-01 — Renamed `model_config["n_pseudo_labels_added"]` to `augmented_training_set_size` and `n_pseudo_samples_injected` in `skill_21_pseudo_label.py` to prevent collision with `pseudo_label_result.n_pseudo_labels_added`. |

---

## 2. Detailed Technical Resolutions

### F1 — Pairwise MI Scale Invariance
```
Description:    The S6 pairwise MI audit function's regression branch computed
                score_val = joint_mi / var(y_raw), but joint_mi was estimated
                in y_scaled space (std-normalized, variance ≈ 1). Dividing by
                raw var(y) reintroduced the target scale — rescaling Y by c
                inflated var(y) by c² while joint_mi was unchanged, making
                score_val scale-dependent.
Status:         RESOLVED 2026-08-25 (v2.8). Fix: divide by var(y_scaled) instead
                of var(y_raw). Confirmed by test_regression_mi_score_is_scale_invariant
                in tests/test_ksg_mi_bivariate_gaussian.py.
```

### F2 — KSG Validator Bivariate Reference Test
```
Description:    The custom KSG bivariate MI estimator lacked direct unit tests
                validating its output accuracy against closed-form mathematical
                solutions (bivariate Gaussian).
Status:         RESOLVED 2026-08-25 (v2.8). tests/test_ksg_mi_bivariate_gaussian.py
                added with 4 tests: independent Gaussians → MI ≈ 0; convergence
                to -0.5*ln(1-ρ²) within 20-30% at n=5000 for ρ=0.8 and ρ=0.5;
                and scale-invariance regression guard (F1).
```

### F4 — MASE Score-Space Residual
```
Description:    Primary-metric routing existed ("mase" key in skill_08_anchor.py
                metric_map) but fold scores resolved to plain RMSE — no
                naive-baseline branch — so a mase config compared RMSE-space
                scores against MASE-calibrated thresholds.
Status:         RESOLVED 2026-08-25 (v2.7). Full Option A MASE folding shipped:
                (a) SoT §2 defines the "mase" lifecycle; (b) _lightgbm_shared.py
                scores each fold as MAE(y_val, yhat_val) / MAE_naive_baseline
                with a hard assertion (no silent unscaled fallback) and exposes
                a dedicated oof_mase field (oof_rmse = oof_mase for backward
                compatibility); (c) skill_08_anchor threads the baseline and
                raises ValueError before training if a mase config has no
                positive MAE_naive_baseline. Skill_11 thresholds are now correctly
                applied to MASE-space scores.
```
