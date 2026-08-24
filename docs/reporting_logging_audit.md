# Orchestrator Reporting & Logging Audit — Consolidated Report

**Status:** Consolidated review of the "Investigative Report & SWOT Analysis" against the actual codebase.
**Last verified:** August 2026
**Scope:** Reporting footprint, log organization, and SWOT across Zindian pipeline phases.
**Method:** Claims from the original report were re-checked against `zindian/skills/*.py`,
`scripts/preflight_enforce.py`, `zindian/orchestrator.py`, and the live
`competitions/ey-biodiversity-challenge/` state. Code behaviour is ground truth.

---

## Executive Summary

The original report is **structurally accurate on the core problem**: the orchestrator's
dual-write pattern has flooded the root of `competitions/<slug>/reports/` with redundant,
categorized, and timestamped artifacts, and the `logs/` directory holds one flat file per
skill. The two recommendations (A — eliminate root flooding; B — reorganise logs) are sound
and safe given the test suite.

However, **four factual claims required correction** before action:

1. **Platt / Isotonic calibration is NOT an open opportunity** — it is already implemented in
   `zindian/skills/skill_09_calibration.py` (both `platt` and `isotonic` methods).
2. **Skill 11 degeneracy is NOT unhandled** — `skill_11_gate.py` has explicit
   `target_std == 0.0` handling that emits a `metadata_warnings` entry and falls back to raw
   thresholds. The real gap is weaker than reported (warn-and-proceed, not unhandled).
3. **Test count is 335, not 329** (collected via `.venv/bin/python -m pytest --co -q`).
4. **Preflight flooding is 33 files, not 24+**; **root report files number 59, not "60+"** —
   the magnitudes were correct, the exact figures were stale.

These corrections are folded into the consolidated SWOT and Action Plan below.

**Decision recorded:** **Recommendation B (logs/ optimisation) was selected.** The log
layout work proceeds under B (see Part 4). **Recommendation A (report-root cleanup) was
largely applied to the writers before the docs were aligned:** skill_03/04/10/15/17/21/22
already write *exclusively* to categorized subdirectories (`audits/`, `diagnostics/`,
`summaries/`, `diagnostics/predictions/`, `sessions/`). The remaining A work is limited to:
relocating `preflight_*.json` into `reports/audits/preflight/`, dealing with the residual
`skill_18`/`skill_20` legacy root copies, and pruning stale root files (see Part 4 + Part 5).

---

## Part 1 — Verified Investigation: Report Folder Flooding

### Root cause: dual-write proliferation

To retain backward compatibility for legacy scripts and tests, production skills write each
report artifact to **both** the root of `reports/` and a categorized subdirectory. Every
occurrence is confirmed in code (not inferred):

| Skill | Root write | Categorized write | Source |
|---|---|---|---|
| skill_03 (legality) | `feature_policy.json`, `legality_report.md` | `audits/feature_policy.json`, `audits/legality_report.md` | `skill_03_legality.py` L284–292, L305–330 |
| skill_04 (EDA) | `eda_report.json`, `eda_summary.md` | `diagnostics/eda_report.json`, `diagnostics/eda_summary.md` | `skill_04_eda.py` L560–565 |
| skill_10 (SHAP) | `shap_analysis.json`, `shap_summary.md` | `audits/shap_analysis.json`, `audits/shap_summary.md` | `skill_10_shap.py` L586–593 |
| skill_15 (reporter) | `phase_<N>_summary.md`, `<phase>_summary.json` | `summaries/` (both formats) | `skill_15_reporter.py` L666–676, L711–719 |

### Measured flood (live `ey-biodiversity-challenge/reports/`)

- **59 artifact files** sit directly under `reports/` root (JSON/MD/CSV).
- **33 timestamped preflight records** (`preflight_ENFORCE_*.json`, `preflight_INIT_*.json`)
  written by `scripts/preflight_enforce.py` L861–875.
- **Phase summaries duplicated** at root *and* under `summaries/`.
- **Heavy diagnostics duplicated:** `eda_report.json/.md` (root + `diagnostics/`) and
  `shap_analysis.json/.md` (root + `audits/`).
- **Model probability arrays** (`oof_probs_pseudo_iter{0..3}*.csv`,
  `test_probs_pseudo_iter{0..3}*.csv`) dumped at `reports/` root by
  `skill_21_pseudo_label.py` L820–821 — with **no** categorized copy.

---

## Part 2 — Investigation: `logs/` Organization

Confirmed in `zindian/orchestrator.py` L246–253: the orchestrator creates
`<slug>/logs/` and opens **one file per skill** — `skill_01.log`, `skill_03_policy_writer.log`,
`skill_04.log`, etc. — capturing stdout via a `Tee` shim. Observed files:

```
skill_01.log  skill_02.log  skill_03_policy_writer.log  skill_04.log  skill_05.log  skill_15.log
```

**Assessment:** a flat, per-skill layout. It cannot be re-assembled into a chronological
session view, and the top-level `logs/` namespace grows with every new skill split.

---

## Part 3 — Consolidated SWOT by Phase

### Phase 1 — Setup, Intake & CV (Skills 01–05, 15)

| | |
|---|---|
| **S** | High-fidelity MD5 data fingerprinting; discussion monitor flags coordinate bans. |
| **W** | **Skill 05 placeholder bug (live defect):** `challenge_config.json` contains `StratifiedKFold` with `selection_reason = "placeholder - skill_05 will select final strategy"`. `skill_05_cv.py` L366–375 treats any non-`auto`/`compare`/empty type as a final user decision, bypassing data-driven CV selection entirely. |
| **W** | No auto-detect alert for unconfigured spatial/temporal coordinates that are physically present. |
| **O** | Skill 05 should ignore config CV when `selection_reason` contains `"placeholder"`. |
| **T** | Suboptimal CV → target leakage / inflated local OOF estimates. |

### Phase 2 — Preprocessing, Baseline & Variants (Skills 06–08)

| | |
|---|---|
| **S** | Automated MCAR/MNAR tracking; smooth `--variant` branching. |
| **W** | No consolidated summary of variant feature definitions. |
| **O** | Auto-generate feature-engineering cards enumerating combined columns. |
| **T** | Custom features accidentally referencing target statistics if `mode`/fold checks are omitted (Two-Mode Feature Contract). |

### Phase 3 — Auditing, Calibration & Promotion (Skills 09–13, 21)

| | |
|---|---|
| **S** | Validation-fold-only SHAP audit (`skill_10_shap.py` L337–338). |
| **S** | 1-SE Nadeau-Bengio inverse-variance gate on promotion. |
| **W** | `target_std == 0.0` handling is warn-only. `skill_11_gate.py` L107–126 detects it and falls back to raw thresholds with a `metadata_warnings` entry — pipeline continues rather than failing loudly. |
| **O** | Platt / Isotonic **already integrated** in `skill_09_calibration.py`. Reframe opportunity as: make Platt/Isotonic the config *default* and validate OOF probability calibration reliability. |
| **T** | Pseudo-label overfitting. Mitigated by `MAX_ITERATIONS = 4` cap in `skill_21` L58, L747; residual threat is within-budget decision-boundary drift. |

### Phase 4 — Final Inference & Governance (Skills 14, 16, 17, 22)

| | |
|---|---|
| **S** | Three-tier governance gate-block (`skill_17_governance.py`); absolute prediction clipping (`skill_14_inference._enforce_*`). |
| **W** | Governance selection reports no per-target metric breakdown (confirmed: not in `skill_17`). |
| **O** | Automated git tagging of commits tied to approved submissions (confirmed not implemented). |
| **T** | Zindi rate limits consumed by redundant test runs. |

---

## Part 4 — Recommendations

### Recommendation B — Optimise `logs/` *(SELECTED — implemented)*

1. **Session-based subdirectories:** write per-skill logs under
   `<slug>/logs/run_<timestamp>/skill_XX.log` so one run is traceable in sequence.
2. **Unified session log:** single `logs/run_<timestamp>/session.log` with
   `=== SKILL XX ===` separators written in addition to per-skill files.

**Impact on `reports/` — none.** `reports/` keeps its current state byte-for-byte.

### Recommendation A — Eliminate root report flooding *(PARTIALLY APPLIED — writer migration done)*

The writer-side consolidation has already landed in code before doc alignment:

- skill_15 → `reports/summaries/phase_<N>_summary.md` + `reports/summaries/<phase>_summary.json`
- skill_03 → `reports/audits/feature_policy.json` + `reports/audits/legality_report.md`
- skill_04 → `reports/diagnostics/eda_report.json` + `reports/diagnostics/eda_summary.md`
- skill_10 → `reports/audits/shap_analysis.json` + `reports/audits/shap_summary.md`
- skill_21 → probability arrays under `reports/diagnostics/predictions/`
- skill_17 (governance) and skill_22 (audit) → `reports/audits/`

Remaining work (writer + cleanup residue):

1. **Move preflight records:** relocate `preflight_{INIT,ENFORCE}_{timestamp}.json`
   writes into `reports/audits/preflight/`.
2. **Residual root dual-writes:** `skill_18_librarian` and `skill_20_scientist` still emit
   legacy root copies (`reports/literature_cache.json`, `reports/domain_hypotheses.json`,
   `reports/{validated,failed}_hypotheses.json`); their readers still read the root path.
   Consolidate writer + reader to `reports/diagnostics/` in the same change.
3. **Prune stale root files:** idempotent `scripts/optimize_report_footprint.py` to archive
   the leftover root duplicates produced before the writer consolidation.

> [!NOTE]
> Writer → reader paths are now aligned for skill_03 (orchestrator `policy_gate` and
> `feature_policy_written` state both use `reports/audits/feature_policy.json`).

---

## Part 5 — Action Plan

**Test suite: 335 tests** (corrected from "329").

### Track 1 — Recommendation B *(immediate, zero report impact)*

1. Extend `zindian/orchestrator.py` (L246–253): write per-skill logs under
   `competitions/<slug>/logs/run_<timestamp>/` instead of the flat `logs/` root.
2. Also write a unified `logs/run_<timestamp>/session.log` alongside the per-skill files.
3. Update any test asserting a flat `logs/skill_XX.log` path to the session-scoped path.

### Track 2 — Recommendation A *(writer migration DONE — cleanup residue remains)*

Writer consolidation is complete for skill_03/04/10/15/17/21/22 (Part 4). Remaining:

1. Relocate preflight writes to `reports/audits/preflight/` + update the SoT preflight
   output lines (`reports/preflight_INIT_*.json`, `reports/preflight_ENFORCE_*.json`).
2. Consolidate skill_18/skill_20 residual root dual-writes to `reports/diagnostics/`
   (writer + reader together).
3. Add `scripts/optimize_report_footprint.py` and prune stale root artifacts.
4. Document deprecation in `docs/deprecated_report_paths.md`.

---

## Part 6 — Corrected Claim Log

| # | Original claim | Verdict | Correction |
|---|---|---|---|
| 1 | "60+ files under reports root" | Stale | 59 files (verified listing) |
| 2 | "24+ preflight files" | Stale | 33 preflight files |
| 3 | "329 tests" | Inaccurate | 335 tests collected |
| 4 | Platt/Isotonic = open opportunity | Incorrect — already shipped | `skill_09` implements both; reframe as "make default / validate" |
| 5 | skill_11 "does not adjust when std==0" | Overstated | Has warn-and-fallback; gap is warn-only, not unhandled |
| 6 | Skill 05 placeholder bypasses auto-CV | Correct & live | Live config contains the placeholder value |
| 7 | Governance lacks target-level metrics | Correct | No per-target breakdown in `skill_17` |
| 8 | No git tagging | Correct | Not implemented |
| 9 | Skill 21 high loop overfitting | Partially mitigated | Hard cap `MAX_ITERATIONS = 4` |

---

*Consolidated from the original investigative report + direct code verification.
Verified against SoT v2.4. Maintained by [whoisorioki](https://github.com/whoisorioki).*
