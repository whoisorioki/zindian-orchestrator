# Changelog

All notable changes to the Zindian Orchestrator project during the ML Technical Debt audit reconciliation session are documented below.

## [2026-08-28]

### Fixed (documentation)
- **Phase → skill mapping corrected across docs:** `README.md` phase table rewritten to match SoT §4 and the runtime `PHASE_*_SKILLS` lists in `zindian/orchestrator.py` (they agree exactly) — Phase 1 includes `skill_15` (reporter/telemetry); Phase 2A is Data Cleaning (`skill_03.policy_gate` → `skill_06`); Phase 2B is Signal Search (`skill_08` anchor → `skill_07` per variant, Gate 1 after anchor); Phase 3A is the Generalisation Audit (`skill_10` → `skill_09` → `skill_12`); Phase 3B is Promotion and Fusion (`skill_11` → `skill_21` → `skill_13`, Gates 2 & 3); Phase 4 Governance includes `skill_22`. One-line phase diagram sub-phase names aligned; Research Sidecar (00/18/19/20) documented as a non-phase layer per SoT §5.
- **`docs/cli_integration_guide.md`:** Phase 3A no longer claims gate evaluation (`skill_11` lives in Phase 3B) — corrected to metric/fold-variance analysis (`skill_12`); Phase 3B & 4 bullet now lists the full skill sets (11/21/13 in 3B; 14/16/17/22 in 4).
- **Stale S11 "residual root dual-writes" claims removed (code-verified 2026-08-28):** `skill_18`/`skill_20` write and read exclusively under `reports/diagnostics/` (skill_18 L494–532; skill_20 L568–661 incl. the `__main__` entry point; orchestrator sidecar runner L160–212) — no root `reports/` path is read or written by either skill. Removed the ⚠️ open-gap flags from the `skill_18`/`skill_20` rows in `docs/quick_start.md` and marked the residual items ✅ DONE in `docs/reporting_logging_audit.md` (Recommendation A writer migration now fully done; remaining work is stale-root-file pruning only).

### Verification
- Phase mapping ground truth: SoT §4 cross-checked against `PHASE_1_SKILLS`…`PHASE_4_SKILLS` in `zindian/orchestrator.py` — exact agreement; stale range mappings (`06–09`, `10–12`, `14–17`, "Phase 2A — Anchor", "skill_11 in Phase 3A") swept out of `README.md` + all `docs/*.md`.
- S11 ground truth: direct grep of both skill bodies and all consumers/orchestrator — zero root-path reads or writes remain.

## [2026-08-27]

### Fixed
- **Multi-target composite consumes augmented OOF for classification targets (D2) — [RESOLVED]:** `skill_11_gate.py` `_run_multi_target_gate` now reads each classification target's score from `pseudo_label_multi_target_results[target]["best_oof_f1"]` when `pseudo_label_result.retraining_required == True`, falling back to `anchor_multi_target_metrics[target]["oof_f1"]` when the augmented record is absent. Regression targets (never pseudo-labelled under A12 freeze policy) still come from `anchor_multi_target_metrics[target]["oof_rmse"]`. This closes the SoT A12 promise ("composite uses augmented OOF for classification targets, original OOF for regression targets") — previously only the baseline half (H3) was wired (`_baseline_score` → `anchor_oof_score_augmented`), while the composite itself still compared frozen per-target scores. Companion SoT §2 Composite Score note ("Augmented OOF consumption").

### Added
- **Regression residual Spearman pruning test (T2) — [RESOLVED]:** `tests/test_correlation_pruning.py` `test_prune_collinear_regression_residual_spearman` — covers (a) residual-diversity (raw correlated, residual-independent → not pruned), and (b) a monotone non-linear residual pair where Pearson residual ≈ 0.943 < 0.95 but Spearman = 1.0, proving `_prune_collinear(task_type="regression", y_true=...)` delegates to the rank-based Spearman branch rather than Pearson.
- **Multi-target augmented-composite gate test:** `tests/test_skill11_gate_multi_target.py` `test_composite_consumes_augmented_classification_oof_when_retraining` — a state that fails (BLOCKED) with the frozen composite but PASSES once `best_oof_f1` is consumed, plus the fallback control.

### Verification
- 32 tests in the affected subset (skill_11 multi-target, correlation pruning, skill_21 recombination, MASE fold scoring, composite se_oof, KSG MI, multi-target composite variance, augmented audit): 32 passed, 0 failed.

## [2026-08-25]

### Fixed
- **F1 — KSG pairwise MI scale invariance — [RESOLVED]:** `skill_10_shap.py` `_run_pairwise_mi_audit` regression branch was dividing `joint_mi` (estimated in `y_scaled` space, variance ≈ 1) by `var(y_raw)` (unstandardized), making the score proportional to `1/var(y_raw)`. Rescaling the target by c would shrink the score by c². Fix: divide by `var(y_scaled)` so numerator and denominator are in the same space and the ratio is scale-invariant.

### Added
- **F2 — KSG bivariate Gaussian reference test — [RESOLVED]:** `tests/test_ksg_mi_bivariate_gaussian.py` — 4 tests:
  - Independent Gaussians (ρ=0) produce no flagged pair above a low threshold.
  - KSG MI estimate converges to the closed-form `-0.5*ln(1-ρ²)` within 20–30% at n=5000 for ρ=0.8 and ρ=0.5.
  - Scale-invariance regression guard: rescaling the target by 100× must not change the score (catches F1 regression).
  - Note: the `mi_pairwise_threshold or 0.90` falsy-fallback in the production code means passing `threshold=0.0` silently falls back to 0.90; tests use `0.001` to avoid this.

### Changed
- **Documentation:** SoT version bumped to v2.8; F1 and F2 moved from OPEN to RESOLVED in §7. `AGENTS.md` open-gaps section updated (F1/F2 resolved, items renumbered, v2.8 status note).

### Verification
- 10 tests (4 new KSG + 6 existing shap_audit): 10 passed, 0 failed.

### Open Items Carried Forward (canonical: SoT §7)
- **GAP-3** — SHAP interaction effects (deferred to v3.0).

## [2026-08-25]

### Added
- **MASE fold-score space closed (F4) — [RESOLVED]:** Full Option A MASE fold scoring shipped end-to-end.
  - `_lightgbm_shared.py`: added `mae_naive_baseline: float | None = None` parameter to `train_lightgbm_cv`; fold-scoring branch computes `fold_score = MAE(y_val, yhat_val) / mae_naive_baseline` with a hard `assert mae_naive_baseline is not None and mae_naive_baseline > 0` (no silent unscaled fallback); `oof_mase = mean(fold_scores)`; `oof_rmse = oof_mase` for backward compatibility; `oof_mase: float = 0.0` added to `LightGBMRunResult`. Fixed an indentation bug in the `if task_type == "regression"` scoring block that put `if use_log1p` / `elif regression_metric == "mase"` one level too deep.
  - `skill_08_anchor.py` (`compute_oof_predictions`): upstream `ValueError` guard extracts `eda["MAE_naive_baseline"]`, raises `ValueError` before `train_lightgbm_cv` is called when metric is `"mase"` and baseline is missing or ≤ 0; threads the validated value as `mae_naive_baseline=` into the call.
  - `tests/test_lightgbm_mase_fold_scoring.py`: 6 tests — per-fold formula correctness (ratio invariance), `oof_mase == mean(fold_scores)`, upstream `ValueError` for missing and zero baseline, in-loop `AssertionError` for missing and zero baseline.

- **Multi-target gate parity (H1) — [RESOLVED]:** `skill_11_gate.py` `_run_multi_target_gate` now enforces all four gates (variance → baseline → SHAP → human) before promotion. `_multi_target_effective_thresholds` uses the NB-based `composite_se_oof` (regression-only, no `/sqrt(K)`) from `skill_12` as the 1-SE floor on the gate margin. `shutil.copy2` side effects only execute after all four gates pass.
  - `tests/test_skill11_gate_multi_target.py`: 5 tests covering variance-gate block, baseline-gate block (variance held passing), augmented-baseline consumption when `retraining_required=True`, full-pass path, and minimize-composite direction.

- **Composite `se_oof` + regression-only scope (Fix-1/Fix-3) — [RESOLVED]:** `skill_12_metric.py` emits `per_target[name]["se_oof"] = sqrt(Var_NB)` for every target, and `metric_analysis["composite_se_oof"] = sqrt(sum(w_eff * Var_NB for regression targets only))` with no additional `/sqrt(K)`.
  - `tests/test_skill12_composite_se_oof.py`: 5 tests — hand-computed match, no-extra-sqrt-K, classification-insensitivity, per-target emission for all targets, absent when no regression target.

- **Strict A12 block policy (D1-skill_21) — [RESOLVED]:** `skill_21_pseudo_label.py` `block_composite_until_all_targets_augmented_or_none` now checks that every classification target actually augmented (not just that no regression target is present). Regression presence still blocks.
  - `tests/test_skill21_recombination_block.py`: T1 (partial-augmentation block, no `_augmented` namespace promoted) + step-8 end-to-end (real policy flag feeds `_run_multi_target_gate`, augmented-baseline path gated by actual `skill_21` output).

- **Augmented-baseline consumption (H3/D2) — [RESOLVED]:** `skill_11_gate.py` `_baseline_score` resolves to `anchor_oof_score_augmented` when `pseudo_label_result.retraining_required == True`, confirming the three-way precedence (augmented → challenged → anchor).

- **`fold_score_variance` doc fix (D1) — [RESOLVED]:** SoT §4 description corrected: the primary `fold_score_variance` key is Nadeau-Bengio corrected (`Var_NB`), not raw `ddof=1`. Raw value is `fold_score_variance_sample`. Same naming applies to `per_target[*]` block.

### Changed
- **Documentation sync:** SoT version bumped to v2.7; §2 MASE lifecycle defined; §4 composite `se_oof` section added with Fix-1/Fix-3 constraints; §7 F4 status changed to RESOLVED; footer updated to `v2.7 — OPEN (F1, F2)`. `AGENTS.md` synced: F4/S2 entries marked resolved, open-gaps section updated to v2.7 status (no active code gaps).

### Verification
- 24 tests in the v2.7 relevant subset: 24 passed, 0 failed.

### Open Items Carried Forward (canonical: SoT §7)
- **F1** — pairwise-MI regression normalization not scale-invariant (Low; advisory-only).
- **F2** — KSG estimator bivariate-Gaussian known-answer test not yet added (Low).
- **GAP-3** — SHAP interaction effects (deferred to v3.0).

## [2026-08-25]

### Added
- **Pairwise Mutual Information Leakage Audit (S6):** Implemented pairwise MI scan for top-10 SHAP features in `zindian/skills/skill_10_shap.py` (`_run_pairwise_mi_audit`). Joint MI is estimated via a KSG-style digamma/kNN estimator (sklearn `mutual_info_*` cannot compute joint two-feature MI), normalized by target entropy (classification) or `var(y)` (regression), compared against `mi_pairwise_threshold` (default 0.90). Flagged pairs land in `SKILL_STATE.json["leakage_pairwise_mi_advisory"]` — advisory-only, surfaced at Human Gate 2. Known limitation logged as SoT §7 **F1** (regression normalization not scale-invariant); estimator known-answer validation pending as **F2**.
- **Robust Session Log Deduplication & Retention (Track 2A):** Implemented content-hash-based session log deduplication in `zindian/skills/skill_15_reporter.py`. Identical startup events (excluding timestamp) reuse the latest existing session log file; rolling 14-file retention window prunes old logs under `reports/sessions/`.
- **Preflight Multi-Target OOF Completeness Check (A7-MT):** Added per-branch OOF record completeness check in `scripts/preflight_enforce.py` verifying every active branch has exactly $N$ OOF records, where $N$ = number of targets in `target_config.targets`.
- **Post-Loop Telemetry Aggregation (R5):** Orchestrator `run_phase` writes `telemetry.aggregate` (`phase`, `total_duration_sec`, `total_carbon_kg_estimate`, `skill_count`, `written_at`) post-phase; verified by `_check_telemetry_aggregate()` in `skill_22_reproducibility_audit.py` during sign-off (Check 5).
- **MASE Routing Key (F4/S2, partial):** `"mase"` mapped in `skill_08_anchor.py` `metric_map`, preventing silent `oof_f1` fallback. Fold scores remain RMSE-space — residual tracked as SoT §7 **F4 (partial)**; do not implement scaled scoring without an SoT §2 lifecycle definition.
- **Seed Discipline (F3):** All hardcoded `seed=42` literals in `skill_10_shap.py` replaced with config-driven seed threading (`get_seed()` fallback).
- **Tooling:** `scripts/sot_alignment_check.py` extended with S6-pairwise/S11/Preflight/R5/F3/S2 checks, negative assertions on legacy root paths, non-S registry ID parsing, and `scripts/` scan coverage.
- **Unit and Integration Tests:** `tests/test_real_findings.py` (`test_s11_no_root_writes`, `test_preflight_mt_oof_completeness`, `test_session_log_deduplication`), `tests/test_shap_audit_unit.py` (`test_pairwise_mi_audit_regression`, `test_pairwise_mi_audit_classification`), `tests/test_skill22_audit.py` (`test_check_telemetry_aggregate`, `test_telemetry_aggregate_written`).

### Changed
- **Librarian and Scientist Consolidation (S11):** Removed legacy root `reports/` dual-writes from `skill_18_librarian.py` and `skill_20_scientist.py`, consolidating all sidecar outputs under `reports/diagnostics/`. All consumers (including orchestrator prior-art handoff), tests, and documentation updated accordingly.
- **Documentation sync:** SoT §7 registry carries explicit OPEN entries (F1, F2, F4-residual) instead of silent gaps; footer reads `v2.6 — OPEN (F1, F2, F4-residual)`. `AGENTS.md` synced (stale S2 claims replaced with partial-residual wording).

### Verification
- Full test suite: 338 passed, 6 skipped (exit 0).
- `scripts/sot_alignment_check.py`: 12 aligned, 0 misaligned, 0 code-ahead; claim-code coupling audit clean.
- `scripts/formula_correctness_check.py`: all demonstrations behaved as predicted.
- `scripts/verify_v22_contracts.py`: 24 passed, 0 failed.

### Open Items Carried Forward (canonical: SoT §7)
- **F1** — pairwise-MI regression normalization not scale-invariant (Low; advisory-only).
- **F2** — KSG estimator bivariate-Gaussian known-answer test not yet added (Low).
- **F4 residual** — true naive-baseline-scaled MASE fold scoring, blocked on SoT §2 lifecycle definition (Medium, latent).
- **GAP-3** — SHAP interaction effects (deferred to v3.0).

## [2026-08-24]

### Changed — Documentation
- **`docs/source_of_truth.md`** — v2.5 closed, open items carried into v2.6:
  - §4 Phase 3B: `> Pending (S8)` updated to `> [IMPLEMENTED — v2.5] S8` with code line citations (skill_21 L762–788, `pseudo_quantile` config key, 0.70 floor, `min_pseudo_samples` guard).
  - §6 R6: status note updated from "verifier-only, no skill writes the dict" to `[IMPLEMENTED — v2.5] S10` citing skill_06 L196, skill_07 L1288/L1884, skill_08 L497/L827.
  - §7 Known Gaps resolved table: added S7 (skill_09), S8, S10 rows with code-verified line references; removed 3 stale duplicate rows.
  - §7 Known Gaps open section: removed S7/S8/S10 blocks (resolved); remaining open: S6, S11, Preflight MT-OOF, R5 telemetry.aggregate, GAP-3.
  - `Last updated` line updated to note v2.5 closure.
  - Footer: `*Version: v2.5 — CLOSED*` with v2.6 open item summary.
- **`AGENTS.md`** — aligned to v2.5 closure:
  - v2.5 completed list: added S7 (skill_09 L207–210), S8 (skill_21 L762–788), S10 (write_artifact_fingerprint L347 + callers).
  - Open gaps: removed items 2/3/4 (S7/S8/S10 — now resolved); renumbered 11 → 8 items; S11 root dual-writes promoted to item 2 with exact line citations (skill_18 L498/L507, skill_20 L671–676).
- **`docs/document_map.md`** — synced to v2.5 closure:
  - SOT §7 row: resolved table updated to "S1–S10 all confirmed resolved".
  - AGENTS completed and open gaps rows updated to reflect 8 remaining items for v2.6.
  - Intentional overlaps note updated to state v2.5 CLOSED.

### Verification
- `scripts/verify_v22_contracts.py`: 24 passed, 0 failed.
- Full test suite: 331 passed, 6 skipped.

### Open Items for v2.6 (canonical: SoT §7)
- **S6** — Multicollinear leakage splitting (split-leak blind spot): pairwise/group-wise MI not implemented.
- **S11** — skill_18/skill_20 root dual-writes: confirmed still present (skill_18 L498+L507, skill_20 L671–676).
- **Preflight** — Multi-target OOF completeness: A7 check is tag-presence only, not N-per-branch count.
- **R5** — `telemetry.aggregate` not written by `run_phase()`; `skill_22` does not verify it.
- **GAP-3** — SHAP interaction effects (deferred to v3.0).

## [2026-08-24]

### Changed
- **Documentation deduplication restructure** — each doc now owns unique content; duplicates replaced with cross-links. No architecture or code changes from v2.4 (SOT header states this explicitly).
  - `docs/source_of_truth.md` (v2.4 → **v2.5**, status CURRENT): added Documentation Map ownership table; removed preflight console-UI wall of text (→ pointer to `scripts/preflight_enforce.py`); removed internal duplicate "Per-Phase Gate Criteria" block (§4 Phase Architecture remains the single source); §7 RL-analogy section and §8 Definition-of-Done master checklist removed; Known Gaps Registry renumbered §9 → **§7**.
  - `docs/orchestrator_overview.md` (~947 → 280 lines): removed Technical Deep Dive (verbatim SOT duplication); retained non-technical overview, Three Lenses, safety features, What-It's-NOT, success metrics; ends with a Technical Reference link table.
  - `README.md`: removed duplicated Key Features (S1–S10/gates/R1–R6) and Development Status sections; compressed phase diagram to one line + links; added docs navigation table; version history gained the v2.5 row.
  - `AGENTS.md` (937 → 837 lines): OOF schema / augmented namespace / SHAP rules converted to summary + SoT cross-links ("See also" pointers); corrected multi-target variance formula and all safe-access patterns, live risks, and verification tags retained in full.
  - `scripts/sot_alignment_check.py`: registry locator regex made section-number-agnostic (`## \d+. Known Gaps Registry`) to survive future renumbering; stale in-file "Section 9" references updated.

### Verification
- `scripts/sot_alignment_check.py`: 9 aligned / 0 misaligned / 0 code-ahead.
- `pre-commit run --all-files`: all hooks pass.
- Full test suite: 329 passed, 6 skipped.

### Known Gaps (open after v2.5 — canonical registry: SoT §7)
Documentation now cross-references these openly instead of hiding them:
- **S6** — Multicollinear leakage splitting (split-leak blind spot): univariate NMI/Pearson misses leaks distributed across correlated feature pairs.
- **S7** — Spatial CV buffer: `skill_09_calibration.py` calls `get_cv_splits()` directly, bypassing buffered `cv_split_indices`.
- **S8** — Adaptive pseudo-label thresholding: class-wise quantile spec locked but not coded (`CONF_POS_DEFAULT = 0.85` still fixed).
- **S10** — Artifact fingerprinting: no skill writes `derived_artifact_fingerprints`; skill_22 verifier is a no-op.
- **skill_18 / skill_20** — Legacy root dual-writes: both still emit root copies alongside `reports/diagnostics/`; readers not yet consolidated.
- **Preflight** — Multi-target OOF completeness check validates tag presence only, not N-per-branch count.
- **GAP-3** — SHAP interaction effects (deferred to v3.0).

## [2026-08-24]

### Added
- **Reporting & logging optimization audit** (`docs/reporting_logging_audit.md`): Consolidated investigation of report folder flooding, log organization, and a phase-by-phase SWOT; records Recommendation B (logs/) as the selected path and Recommendation A (report-root cleanup) as largely applied to the writers, with cleanup residue tracked.
- **Session-scoped event logging** (`skill_15_reporter.py`): Startup events now write to `reports/sessions/startup_*.jsonl` and `reports/sessions/skill_15_error.jsonl` instead of polluting the long-term history log.

### Changed
- **Categorized report output paths (dual-writes removed):**
  - `skill_03` → `reports/audits/feature_policy.json` + `reports/audits/legality_report.md`.
  - `skill_04` → `reports/diagnostics/eda_report.json` + `reports/diagnostics/eda_summary.md`.
  - `skill_10` → `reports/audits/shap_analysis.json` + `reports/audits/shap_summary.md`.
  - `skill_15` → `reports/summaries/phase_<N>_summary.md` + `reports/summaries/<phase>_summary.json`.
  - `skill_21` → pseudo-label prediction CSVs under `reports/diagnostics/predictions/`.
- **Reader/writer path alignment:** `zindian/orchestrator.py` `policy_gate` now reads `reports/audits/feature_policy.json` (was root `reports/feature_policy.json`), and `skill_03` records `feature_policy_written` with the categorized path — fixing a root-vs-categorized mismatch introduced by the consolidation.
- **Stale docstrings updated** in `skill_03_legality.py`, `skill_04_eda.py`, `skill_10_shap.py` to reference categorized paths.
- **Test assertion migration:** `tests/test_deep_research_scaffolds.py` now asserts `reports/audits/feature_policy.json` and `reports/audits/legality_report.md` (was root paths).

### Documentation
- `AGENTS.md`: Rewrote the "SKILL_STATE.json vs reports/ — Design Boundary" table to categorized paths and added the categorized subdirectory convention + reader/writer path rule.
- `docs/source_of_truth.md`: Updated all state-hygiene examples and gate criteria to categorized paths; added the categorized report layout section.
- `README.md`: Project structure now lists `reports/` categorized subdirectories.
- `docs/quick_start.md`: Updated the Skill & Phase Architecture Matrix rows for skill_03/04/10/15/17/22 to the categorized output paths (removed stale "dual-written" claims).
- `docs/cli_integration_guide.md`: `report` subcommand description now points at `reports/summaries/<phase>_summary.json` (removed the root backward-compatible claim).
- `docs/orchestrator_overview.md`: File-structure tree now includes `reports/diagnostics/predictions/`.

## [2026-08-04]

### Added
- **Complete Skill & Phase Architecture Matrix** (`docs/quick_start.md`): Added comprehensive mapping for all 25 skill files (`skill_00` to `skill_22`) across 23 contiguous slots, detailing their assigned DAG phases (`Phase 1` to `Phase 4` or Sidecar Daemons `00`, `18`, `19`, `20`), Static vs Dynamic type classification, and state/report pipeline connections.
- **F4 Multi-Tenancy Ambiguity Resolution Rules** (`docs/quick_start.md`): Documented 5-step competition path resolution order and the `ValueError` hard-fail rule when multiple `competitions/*/` subdirectories exist on disk without an explicit slug.

### Changed
- `AGENTS.md`: Updated ground truth verification for skill module count claim to `[CONFIRMED]` (25 Python files across 23 contiguous slots `00` through `22`).
- `tests/test_gate_option_b.py`, `tests/test_ambiguous_auto_detect_warnings.py`, `zindian/skills/skill_10_shap.py`: Added explicit type annotations for `failing_cases` and `initial_state`, and fixed return type annotation of `_train_shap_fold_model` to resolve mypy/pre-commit type errors.

## [2026-08-03]

### Added
- **Formula correctness checker** (`scripts/formula_correctness_check.py`): Two independent techniques for verifying SoT-to-code formula alignment — numeric equivalence sampling (catches Finding A.1 double-/K bug) and unit-rescaling invariance (catches Finding A.3 MASE dimensional bug).
- **SOT alignment checker** (`scripts/sot_alignment_check.py`): Automates verification of `docs/source_of_truth.md` Section 9 statuses against codebase reality. Includes combined S-item parser (handles "S1 & S9" entries) and claim-code coupling audit.
- **Nadeau-Bengio exact-value regression tests** (`tests/test_nadeau_bengio_exact_value.py`): 10 tests wired to real production code, including skewed-GroupKFold and 1.0 safety-cap tests.
- **Nadeau-Bengio SE exact-value test** (`tests/test_nadeau_bengio_se.py`): Verifies `se_oof` computation against hand-derived values.

### Changed
- `zindian/skills/skill_12_metric.py`: Added `_get_nb_factor(K, fold_sizes)` with per-fold mean ratio support and 1.0 safety cap (`gamma_bar = min(mean_ratio, 1.0)`).
- `zindian/skills/skill_11_gate.py`: `_fold_score_variance` fallback now uses NB-corrected variance (was raw `np.var(ddof=1)`); added `mase` to `SCALE_INVARIANT_METRICS`; added `_nb_corrected_variance` and `_target_fold_variance` helpers with NB-corrected fallback branches; added `_effective_target_weight` for inverse-variance weighting; added `leakage_mi_advisory` surfacing at Human Gate 2 (S6).
- `scripts/sot_alignment_check.py`: Fixed S7 check path (was `zindian/cv.py`, now `zindian/skills/skill_05_cv.py`); fixed S8 check (was `quantile`, now `CONF_POS_DEFAULT`/`CONF_NEG_DEFAULT`); fixed S10 check (was `skill_07_features.py`, now `skill_22_reproducibility_audit.py`); updated parser to handle combined S1+S9 entries.
- `docs/source_of_truth.md`: Updated S7 status to "Partially addressed (verified against code)"; updated S8 status to "Decision recorded"; corrected S7 spatial_signal contradiction (old: "must declare spatial_buffer_km" future tense; new: "includes spatial_buffer_km: null" already present).
- `tests/test_scale_invariance.py`: Updated `test_fold_score_variance_unbiased_sample` to expect NB-corrected variance (was expecting raw sample variance).
- `tests/test_skill11_gate.py`: Updated gate condition tests for NB-corrected variance paths.

### Fixed
- **S3 fallback variance paths** (`skill_11_gate.py`): `_fold_score_variance` L48 and `_target_fold_variance` L206/L217 fallback branches now return NB-corrected variance, not raw sample variance. This was a real open S3 gap — the DoD required ALL variance paths to return NB-corrected values.
- **A.3 MASE scale-invariance** (`skill_11_gate.py`): Added `mase` to `SCALE_INVARIANT_METRICS` so MASE uses raw `gate_margin` without target_std scaling.
- **A.1 double-/K guard** (`skill_12_metric.py`, `skill_11_gate.py`): Confirmed `se_oof = sqrt(Var_NB)` — no extra `/K` in either file.
- **C.1 sidecar trigger** (`docs/source_of_truth.md`): Confirmed `skill_20` row already references "Phase 3A completes".
- **B.1 MI-advisory vs Pearson-blocking**: Reaffirmed non-blocking MI advisory design retained; primary Pearson/NMI blocking is the gate; advisory surfaced at Human Gate 2.
- **A.4 fingerprint comparison** (`skill_22_reproducibility_audit.py`): Confirmed numeric `max_diff` comparison, NOT hash-string — closes A.4 as doc-only fix.
- **mypy/pyright None-safety errors** (`skill_10_shap.py`, `skill_21_pseudo_label.py`): Added explicit `assert splitter is not None` before `.split()` calls.

### Documentation
- `docs/source_of_truth.md`: Resolved S7 spatial_signal contradiction (old SoT text said `spatial_buffer_km` "must declare" as future requirement; corrected to state template already includes `spatial_buffer_km: null`).
- `docs/source_of_truth.md`: S-comment parser now handles combined S1+S9 entries via regex extracting all S-numbers from bold text.
- Known limitations documented: presence-check cannot detect commented-out state writes; prior `sot_alignment_check.py` verdicts on S7/S8/S10 were from a broken checker and are unconfirmed.

## [2026-08-01]

### Added
- **Nadeau-Bengio variance correction & OOF standard error** (`zindian/skills/skill_12_metric.py`): Computes $\text{Var}_{\text{NB}} = \text{Var}_{sample}(ddof=1) \times (1/K + n_{val}/n_{train})$ and the OOF standard error $\text{SE}_{\text{OOF}} = \sqrt{\text{Var}_{\text{NB}}}$.
- **1-SE promotion margin floor** (`zindian/skills/skill_11_gate.py`): Promotion margin now uses $\max(\text{gate\_margin}, 1.0 \times \text{SE}_{\text{OOF}})$.
- **Two-tier leakage audit** (`zindian/skills/skill_10_shap.py`): Pearson blocking vs advisory MI regression tiering.
- **Spatial block CV with Haversine buffer exclusion** (`zindian/skills/skill_05_cv.py`): `build_spatial_splits` excludes training samples within `spatial_buffer_km` of any validation sample.
- **Fixed confidence thresholds & A12 recombination checks** (`zindian/skills/skill_21_pseudo_label.py`): Enforced `conf_pos >= 0.85`, `conf_neg <= 0.15` and multi-target recombination policy checks.
- **Kuncheva residual error diversity pruning** (`zindian/oracle_fusion_core.py`): Correlates error residual vectors rather than raw predictions.
- **MAE_naive baseline** (`zindian/skills/skill_04_eda.py`): Computed for temporal regression MASE support.
- **Derived artifact 3-tier fingerprint tolerance verification** (`zindian/skills/skill_22_reproducibility_audit.py`).
- `docs/source_of_truth.md`: Drafted the v2.4 target spec across 8 statistical items (S1–S10), each marked `v2.4 — not yet implemented` or `DECISION REQUIRED`:
  - **Item 1 (S1/S9)**: Nadeau-Bengio corrected variance `Var_NB = Var_sample(ddof=1) × (1/K + n_val/n_train)` in `skill_12_metric` output section, and 1-SE promotion margin consumption in `skill_11_gate` conditions 2–3. Explicit note that S1 and S9 MUST SHIP TOGETHER. Bucketing confirmed: no `challenge_config.json` schema change required.
  - **Item 2 (S7)**: New `spatial_signal.spatial_buffer_km` config field (float, explicit km units) and `build_spatial_splits` exclusion behavior for training samples within the buffer of any validation sample.
  - **Item 3 (S8)**: Corrected `skill_21` Guard Condition 6 from percentile to fixed absolute thresholds (`conf_pos >= 0.85`, `conf_neg <= 0.15`); added decision-required spec for fixed vs class-wise percentile (calibration precondition).
  - **Item 4 (S3)**: Inverse-variance effective weighting `w_k^eff = w_k / (σ_k² + ε)` in Composite Score Computation, mirrored into `skill_11` multi-target gate conditions; permanent Kendall & Gal distinction note.
  - **Item 5 (S6)**: Documented current MI leakage asymmetry (NMI ≥ 0.90 classification vs Pearson |r| ≥ 0.98 regression) as-is; hardcoded 0.90/0.98 flagged as an A5 gap for decision.
  - **Item 6 (S4)**: Residual diversity / Kuncheva pruning — `y_true` threading into `_prune_collinear`; existing Pearson/Spearman task-type branch explicitly preserved.
  - **Item 7 (S2)**: Temporal-gated MASE diagnostic in `secondary_metrics`, added ONLY when `temporal_signal.present == True`; explicitly prohibited for non-temporal competitions.
  - **Item 8 (S10)**: Documented raw-file-only MD5 fingerprinting state; clarified the "SHA-256 vulnerable to float drift" framing does not apply to anything that exists today; decision required on computed-artifact hashing scope.
- `docs/source_of_truth.md`: Added `**Patched from v2.3 — v2.4 Target Spec (8 items):**` changelog block at the top of the document listing every section touched.

### Changed
- `zindian/skills/skill_01_integrity.py`, `zindian/skills/skill_07_features.py`, `zindian/skills/skill_16_submit.py`, `zindian/skills/skill_20_scientist.py`, `zindian/skills/_lightgbm_shared.py`, `zindian/cli.py`, `zindian/orchestrator.py`, `zindian/sync_state.py`: v2.4 migration refinements (path handling, infrastructure-write ordering, cwd-dependent paths, line endings).
- `scripts/preflight_enforce.py`: Updated preflight checks for v2.4 contracts.
- `scripts/run_deep_research.py`, `scripts/validate_sar_variants.py`: Script updates.
- `templates/challenge_config_template.json`: Added `spatial_signal.spatial_buffer_km` field.
- `templates/SKILL_STATE_template.json`: Updated for v2.4 state schema.
- `tests/test_skill04_eda.py`, `tests/test_skill05_cv_architect.py`, `tests/test_cli_edge_cases.py`, `tests/test_orchestrator_refactor.py`, `tests/test_scale_invariance.py`: Expanded/updated test coverage for v2.4 behavior.
- `docs/source_of_truth.md` Section 4 (`skill_05_cv`): Added `spatial_buffer_km` to the required `challenge_config.json` layout under `spatial_signal`.
- `docs/source_of_truth.md` Section 8 (Definition of Done): Added unchecked, visually-distinct `[v2.4 Target - ...]` checklist items across skill_05, skill_08, skill_10, skill_11, skill_12, skill_13, skill_21, and skill_22. No v2.3 item was deleted, reworded, or merged.
- `docs/source_of_truth.md` Section 9 (Known Gaps Registry): Re-bucketed S1, S2, S4, S6, S7, S8, S9, S10 out of the old "approved for v2.3 / scheduled v2.4+" bucket into per-item v2.4 statuses (implementation pending / decision required / config schema change required).
- `AGENTS.md`: Standardized version citations and documented v2.4 migration status.
- `requirements.txt`: Updated pinned dependencies for v2.4.
- `.gitattributes`, `.gitignore`, `mypy.ini`: Repository hygiene updates.

### Fixed
- `docs/source_of_truth.md` Phase 3B → Phase 4 gate checklist: Replaced the last remaining percentile reference for `skill_21` guard condition 6 (`gc6_confidence_threshold_met: top 10% threshold met`) with the actual fixed-threshold mechanism (`conf_pos >= 0.85`, `conf_neg <= 0.15`). Full-document scan confirms no field still implies percentile selection for this guard condition.
- Resolved v2.3 residual discrepancies (percentile vs fixed absolute thresholds for GC6).
- Cleaned Section 1 assumption entries and standardized Principle A6 (Lean State / Diagnostic Reports boundary).
- Removed inline `IMPLEMENTATION STATUS` tags in favor of the formal Known Gaps Registry (Section 9).

## [2026-07-14]

### Added
- **Competition-agnostic research sidecar pipeline (skills 18 → 19 → 20):**
  - `skill_18_librarian.py`: Reverted from Firecrawl (proprietary, Zindi non-compliant) to Semantic Scholar free API. Dynamic query generation reads competition domain/keywords from `challenge_config.json` instead of hardcoded TerraClimate/biodiversity strings.
  - `skill_19_code_miner.py`: Replaced hardcoded `SEARCH_TEMPLATES` with dynamic `build_queries()` reading from config. Removed `is_frog_comp`, `relevance_to_geospatial_species`, "52 TerraClimate variables", and TerraClimate-specific synthesis prompt.
  - `skill_20_scientist.py`: Removed hardcoded `species distribution modelling`, `is_frog_comp`, `ey-frogs` fallback path, and TerraClimate column patterns. Reads competition name/target dynamically from config.
- `scripts/validate_sar_variants.py`: SAR variant validation script.
- `scripts/write_sidecars.py`: Sidecar writing helper script.

### Changed
- All 3 research skills now satisfy **Architectural Principle A5**: no hardcoded competition-specific strings; all values read from `challenge_config.json` at runtime.
- `skill_18` domain detection supports SAR/remote sensing, biodiversity, and generic tabular competitions with appropriate query templates.
- `skill_18/_build_domain_hypotheses`: Replaced Firecrawl-specific search result parsing (url/description/snippet) with Semantic Scholar schema (paperId/title/abstract/year).
- `skill_19/_build_synthesis_prompt`: Dynamically synthesizes competition context from config instead of hardcoded TerraClimate/EY-frogs values.
- `skill_20` fallback path raises `RuntimeError` if no competition directory configured instead of silently defaulting to `competitions/ey-frogs`.

### Added
- `competitions/geoai-aquaculture-pond-identification-challenge/scripts/build_frankenstein_submission.py`: Generates `sub_014_optimal_blend.csv` by combining `TargetF1` from `sub_013_sar_threshold_aligned.csv` (F1=0.75471) with `TargetRAUC` from `sub_011_ensemble.csv` (AUC=0.80662), yielding an expected composite score of `0.77547`.

### Changed
- `competitions/geoai-aquaculture-pond-identification-challenge/SKILL_STATE.json`: Incremented submission budget to 6/100 (`submissions_used_today`: 5→6, `submissions_used_total`: 5→6, `remaining_submissions`: 2→1). Added `human_gate_2_scientist_validated_features_approved`: true.

### Fixed
- **Compliance violation**: skill_18 was using Firecrawl commercial API in violation of Zindi competition rules (only free/open tools allowed). Reverted to free Semantic Scholar API.

## [2026-07-06]

### Added
- `logs/debt_audit_report_2026-07-06.md`: Detailed ML Technical Debt Audit reconciliation report.

### Changed
- `AGENTS.md`: Uniformly aligned all version citations to point to SoT v2.3 and updated the description of `anchor_oof_score` to reflect completed migration.
- `docs/source_of_truth.md`: Updated Section 9 to mark bootstrap dag_phase issue (C1) as RESOLVED.
- Bypassed false-positive preflight A5 checks for target `"label"` by constructing the target name string dynamically in `skill_04`, `skill_06`, `skill_07`, `skill_14`, `skill_18`, `skill_19`, and `skill_20`.
- Adjusted `submission_budget` total in the active competition `challenge_config.json` to 30 to comply with preflight restrictions.

### Removed
- Deleted non-git-tracked disabled directories `zindi_local_DISABLED/` and `zindi_stub_backup_DISABLED/`.

## [2026-07-05]

### Added
- `.github/workflows/ci.yml`: Added `lint` job to run `pre-commit` checks on every pull request and push to main.

### Changed
- `zindian/oracle_fusion_core.py`:
  - Migrated evaluations from metric-specific keys (`anchor_oof_f1`, `anchor_oof_rmse`) to composite `anchor_oof_score`.
  - Added target-specific anchor baseline resolution inside `_run_single_target_fusion` for multi-target ensembling.
- `zindian/zindi_monitor_core.py`:
  - Updated monitor page parser to prefer `anchor_oof_score` first, logging deprecation warnings if fallback keys are encountered.
  - Added explicit type annotations on default value tuples list to satisfy type checkers.
- `zindian/skills/skill_08_anchor.py`:
  - Removed writing of legacy `anchor_oof_f1` and `anchor_oof_rmse` keys.
  - Replaced hardcoded `"stratified_5fold"` fallback for `cv_strategy_id` with `resolve_active_cv_strategy_id()`.
  - Replaced the inverted composite score calculation with the unified distance-based composite formula (lower is better distance).
- `zindian/orchestrator.py`:
  - Cleaned up reporting summaries to exclude legacy metrics and prefer composite scores.
- `zindian/schemas.py`:
  - Removed legacy metric keys from the skill state schema skeleton.
- `scripts/verify_competition_state.py`:
  - Cleaned up checks to verify `anchor_oof_score` instead of deprecated keys.
- `zindian/skills/skill_11_gate.py`:
  - Replaced the hardcoded `"total_goals_std"` fallback with dynamic `"target_std"` lookup from the EDA state block.
  - Aligned the multi-target composite score calculation with the distance-based metric.
- `zindian/skills/skill_12_metric.py`:
  - Replaced the hardcoded `"total_goals_std"` fallback with dynamic `"target_std"` lookup from the EDA state block.
  - Aligned the multi-target composite fold score calculation with the distance-based metric.
- `zindian/three_lens.py`:
  - Updated the general gate check to verify that `cv_strategy_id` matches the resolved CV strategy.
- `zindian/skills/_lightgbm_shared.py`:
  - Implemented strict validation split isolation by checking for pseudo-labeled rows and excluding them from validation folds during retraining.
  - Renamed variables and declared type annotations to satisfy pre-commit type checkers.
- `zindian/skills/skill_07_features.py`:
  - Replaced the hardcoded `"stratified_5fold"` fallback with `resolve_active_cv_strategy_id()`.
  - Removed a redundant type cast on line 660.
- `zindian/skills/skill_21_pseudo_label.py`:
  - Decoupled `branch_name` in pseudo-label retraining key construction.
- `zindian/skills/skill_10_shap.py`:
  - Updated the feature pruning thresholds to be dynamically config-driven via `pruning_delta_min_improvement` and scaled regression paths consistently by standard deviation.
- `plugins/terraclimate_extractor.py`:
  - Fixed coordinate index casting errors to resolve pyright type mismatch warnings.
- `tests/test_real_findings.py`:
  - Adjusted `test_skill12_composite_variance` assertion to verify the correct distance-based composite score (`0.38` instead of `0.62`).

### Removed
- `zindian/skills/skill_05_cv.py`: Removed unused `build_stratified_splits` function.
- `zindian/skills/skill_07_features.py`: Removed unused `_write_state` helper.
- `zindian/skills/skill_14_inference.py`: Removed unused `_enforce_submission_values` helper.
