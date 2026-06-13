# Zindian Orchestrator — README

An **autonomous ML competition agent framework** for Zindi Africa competitions.

> 🎯 **Framework, not specific competition** — Works for any Zindi competition by reading competition rules dynamically.

Every action should declare which problem it serves: Problem 1 (generic Zindian agent) or Problem 2 (EY Biodiversity execution).

## 🏗️ Architecture Overview

### Core Principles

1. **Competition Agnosticism** — Reads `challenge_config.json` before decisions
2. **Atomic State** — Tempfile + os.replace prevents corruption
3. **Data Integrity** — MD5 hash lock on target column
4. **Submission Governance** — Budget guard, structured comments
5. **Audit Trail** — DuckDB ledger for all experiments

### Phases

```
Phase 0: Foundation (Wiring + Auth)              ✅ 100%
Phase 1: Integrity + Intake (MD5 Lock + Config) ✅ 100%
Phase 2: Anchor Baseline (Legality + Anchor)     ✅ 100%
Phase 3: Features + Calibration + SHAP           ✅ 100%
Phase 4: Gating & Submission                     ✅ 100%
Phase 5: Fusion + Final Submit + Governance      ✅ 100%
Phase 6: Tabula Init CLI                        ✅ 100%
Phase 7: Multi-competition Validation           🔄 In Progress
```

---

## 📁 Project Structure

```
zindian-orchestrator/
├── competitions/                     ← per-competition workspace (challenge_config + SKILL_STATE + data/)
│   └── <slug>/
│       ├── challenge_config.json
│       ├── SKILL_STATE.json
│       ├── data/
│       ├── notebooks/
│       └── reports/
├── AGENTS.md                        Master specification (600 lines)
├── CLEANUP_GUIDE.md                 Guide to remove per-competition files
│
├── specs/                           Durable specifications
│   ├── requirements.md              7 FRs + 3 NFRs
│   ├── design.md                    Architecture + data flow
│   └── tasks.md                     Phase 0-5 checklist
│
├── zindian/                         Python package (core logic)
│   ├── state.py                     SKILL_STATE.json reader/writer
│   ├── config.py                    challenge_config.json reader
│   ├── ledger.py                    DuckDB wrapper
│   ├── cv.py                        CV splits generator
│   ├── paths.py                     Path resolver
│   ├── schemas.py                   Schema validator
│   ├── zindi_client.py              Zindi API wrapper (agent-mode)
│   ├── orchestrator.py              Skill orchestration
│   └── skills/                      All 22 implemented skills
│       ├── skill_00_discussion_monitor.py
│       ├── skill_00_zindi_monitor.py
│       ├── skill_01_integrity.py
│       ├── skill_02_intake.py
│       ├── skill_03_legality.py
│       ├── skill_04_eda.py
│       ├── skill_05_cv.py
│       ├── skill_06_cleaning.py
│       ├── skill_07_features.py
│       ├── skill_08_anchor.py
│       ├── skill_09_calibration.py
│       ├── skill_10_shap.py
│       ├── skill_11_gate.py
│       ├── skill_12_metric.py
│       ├── skill_13_ensemble.py
│       ├── skill_13_oracle_fusion.py
│       ├── skill_14_inference.py
│       ├── skill_15_reporter.py
│       ├── skill_16_submit.py
│       ├── skill_17_governance.py
│       ├── skill_18_librarian.py
│       ├── skill_19_code_miner.py
│       ├── skill_20_scientist.py
│       ├── skill_21_pseudo_label.py
│       └── skill_22_reproducibility_audit.py
│
├── tabula/                          Tabula competition bootstrapper CLI
│   ├── init.py
│   ├── __main__.py
│   └── __init__.py
│
├── scripts/
│   ├── bootstrap_competition.py
│   ├── compile_requirements.sh
│   ├── init_ledger.py               Initialize DuckDB
│   ├── inspect_zindi.py
│   ├── preflight_enforce.py
│   ├── test_phase_1.py              Phase 1 integration test
│   ├── verify_competition_state.py
│   ├── verify_phase_b.py            Module verification
│   ├── write_oof_meta.py
│   └── zindian_audit.sh             Orchestrator audit script
│
├── templates/                       Per-competition templates
│   ├── SKILL_STATE_template.json
│   └── challenge_config_template.json
│
└── IDE bridges (Tool agnostic)
    ├── .github/instructions/zindian.md
    ├── .cursor/rules/zindian.md
    ├── .windsurf/rules/zindian.md
    └── .kiro/specs/zindian.md
```

---

## 🚀 Quick Start

### 1. Understand the Architecture (5 min)

```bash
# Read the master spec
cat AGENTS.md

# Read requirements
cat specs/requirements.md

# Read design
cat specs/design.md
```

### 2. Install & Verify (5 min)

```bash
# Activate venv (Unix)
source .venv/bin/activate

# Activate venv (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install dependencies from pinned file
pip install -r requirements.txt
```

### Manage Python dependencies (pinned)

This repository uses `requirements.in` plus `pip-compile` (from `pip-tools`) to produce a pinned `requirements.txt`. Developers should generate and commit `requirements.txt` after updating `requirements.in`.

```bash
# Install the compiler
pip install --upgrade pip-tools

# Generate pinned requirements.txt
pip-compile requirements.in --output-file requirements.txt

# Install the pinned environment
pip install -r requirements.txt
```

### 3. Initialize DuckDB Ledger (2 min)

```bash
python scripts/init_ledger.py
```

### 4. Run Automated Test Suite (5 min)

The repository includes a comprehensive `pytest` test suite covering 44+ test files with 160+ unit and integration tests.

```bash
# Run tests inside virtualenv
.venv\Scripts\pytest
```

### 5. Run Phase 1 Tests (5 min)

```bash
python scripts/test_phase_1.py
```

### 6. Review Code (90 min)

## 📚 Documentation

| Document | Purpose | Audience |
|----------|---------|----------|
| [AGENTS.md](AGENTS.md) | Master spec | Developers, integrators |
| [specs/requirements.md](specs/requirements.md) | What system must do | Architects, reviewers |
| [specs/design.md](specs/design.md) | How it works | Developers, code reviewers |
| [specs/tasks.md](specs/tasks.md) | Task checklist | Project managers |
| [CLEANUP_GUIDE.md](CLEANUP_GUIDE.md) | Framework cleanup | DevOps/repo maintainers |

---

## 🔐 Security

- ✅ Credentials in `.env` (not committed)
- ✅ Atomic state updates prevent corruption
- ✅ No hardcoded paths or competition data
- ✅ Guard exceptions on null config fields
- ✅ MD5 hash lock prevents data tampering

---

## 🧪 Testing

### Automated Tests (pytest)
- Comprehensive test framework with 160+ test cases.
- Run using: `.venv\Scripts\pytest` (Windows) or `.venv/bin/pytest` (Unix).
- Unit tests for: `state.py`, `config.py`, `ledger.py`, `cv.py`, `paths.py`.
- Skill verification tests for: anchor training, gating logic, SHAP audit, pseudo-labeling.

---

## 📈 Development Progress

### Completed ✅
- Architecture design
- Core modules (state, config, ledger, zindi_client, paths, schemas, cv)
- Phase 0-5 all 22 skills implemented and verified
- Tabula Bootstrapper CLI
- Comprehensive test suite (167 tests passing)
- Regression Support & Secondary Metrics (Wave 2) refactor completed with scale-invariant safety guards, continuous domain post-processing, and secondary diagnostic telemetry

### In Progress 🟡
- Multi-competition validation

## 📈 Regression Support & Secondary Metrics

The framework supports continuous prediction domains (regression tasks) with scale-invariant safety guards and rich auxiliary diagnostic tracking.

### Secondary Metrics Schema
For all regression model evaluations, the orchestrator computes and stores secondary diagnostics inside each OOF record under a nested `secondary_metrics` block to prevent schema bloat:
```json
"branch_variant-01_oof": {
  "scores": [...],
  "cv_strategy_id": "stratified",
  "seed": 42,
  "branch_name": "variant-01",
  "secondary_metrics": {
    "mae": 0.123,
    "mape": 4.56,
    "r2": 0.78
  }
}
```

### Scale-Invariant Gating (RMSE vs. RMSLE)
To ensure robust search decisions across arbitrary target distributions, the validation gates dynamically normalize thresholds:
- **RMSE/MAE (Continuous Original Scale):** Gate variance threshold scales by $\sigma_y^2$ (`target_std ** 2`) and gate margin scales by $\sigma_y$ (`target_std`).
- **RMSLE (Dimensionless Log-Ratio):** Evaluated against raw thresholds with no target standard deviation scaling, since log-space metrics are naturally scale-invariant.

---

## 🤝 Contributing

The framework is **specification-driven**. To add a new skill:

1. **Design** — Add to `specs/tasks.md`
2. **Implement** — Create `zindian/skills/skill_XX_*.py`
3. **Test** — Add to `tests/` or `scripts/`
4. **Document** — Update `specs/requirements.md`

### Skill Template

```python
"""Skill XX — Description"""
from zindian.config import ChallengeConfig
from zindian.state import SkillStateStore

def run(
    *,
    state_path: str = "SKILL_STATE.json",
    config_path: str = "challenge_config.json",
    **kwargs
):
    """
    Run Skill XX.
    
    Returns:
        Dict: {"status": "GO|ERROR", "result": ..., "message": "..."}
    """
    try:
        config = ChallengeConfig.load(config_path)
        state_store = SkillStateStore(Path(state_path))
        
        # YOUR LOGIC HERE
        
        state_store.update(dag_phase="phase_X_done")
        return {"status": "GO", "result": ...}
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}

if __name__ == "__main__":
    print(run())
```

---

## 🐛 Known Issues

| Issue | Status | Fix |
|-------|--------|-----|
| Zindi `select_a_challenge()` breaking | ✅ FIXED | Use `challenge_id=slug` param |
| Competition files in framework | 📋 NOTED | See [CLEANUP_GUIDE.md](CLEANUP_GUIDE.md) |

---

## 📞 Support

- **Architecture questions** → Read [AGENTS.md](AGENTS.md)
- **Code review** → Read [VALIDATION_SUMMARY.md](VALIDATION_SUMMARY.md)
- **Implementation questions** → Check specs/
- **Bug reports** → See [AUDIT_REPORT.md](AUDIT_REPORT.md) for known issues

---

## 📄 License

[Add your license here]

---

## 📅 Roadmap

| Phase | Timeline | Status |
|-------|----------|--------|
| 0-6 | ✅ Complete | Ready for review |
| 7 | 🔄 In Progress | Multi-competition Validation |

**MVP Target**: ~3 weeks (after Phase 5 completion)

---

**Last Updated**: June 13, 2026  
**Status**: Phase 0-6 Complete, Regression Support Refactor Completed ✅
