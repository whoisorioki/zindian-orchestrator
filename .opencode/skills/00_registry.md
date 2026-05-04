---

## description: Zindian Orchestrator v2.0 skill registry (17 skills)

## How to use these skills

- Each file in `.opencode/skills/` is a **repeatable procedure** for the agent.
- Every skill must be **competition-aware** via `competitions/<slug>/challenge_config.json`.
- Every skill must **read and write** `competitions/<slug>/SKILL_STATE.json` on state changes.

## Skill list (v2.0)

- **01**: Integrity Governance
- **02**: Challenge Intake (populate/validate `challenge_config.json`)
- **03**: Deep Research / Legality Check (external data/models policy)
- **04**: Violation EDA (domain-conditional checks only)
- **05**: CV Architect (split strategy from config + modality)
- **06**: Feature Engineering (domain-conditional, leakage-audited)
- **07**: SHAP Audit (top features → `reports/shap_analysis.json`)
- **08**: Anchor Baseline (OOF metric + first governed submission)
- **09**: Granular Calibration (group residual mean matching)
- **10**: Leakage Check (flags HIGH_RISK features)
- **11**: Branch Generator (hypothesis-driven branch naming)
- **12**: Metric Trade-off Analysis (regression-only unless config says otherwise)
- **13**: Fusion Candidate Search (find robust blends)
- **14**: Inference Guard (probabilities vs regression clipping)
- **15**: Reporter (ledger + markdown logs)
- **16**: Self-Critique Gatekeeper (blocks fusion unless stable)
- **17**: Submission Governance (select exactly 2 final submissions + rationale)