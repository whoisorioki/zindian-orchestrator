# Zindian Orchestrator — Consolidated Workspace Standard

This document defines the core conventions, architectural rules, and refactoring integrity steps required of all agents working in this repository. For CLI command reference, refer to [cli_integration_guide.md](file:///c:/Users/Adrian/Desktop/Agents/zindian-orchestrator/docs/cli_integration_guide.md).

---

## 1. Topography & Hygiene
* **Package Directory (`zindian/`)**: Contains the core pipeline scripts. All skill files are located under `zindian/skills/`.
* **Testing Directory (`tests/`)**: Flat layout containing unit and integration tests. Shared fixtures are declared in a single root-level `conftest.py`.
* **Git Hygiene**:
  - **Commit**: Source code, unpinned specifications (`requirements.in`), pinned requirements (`requirements.txt`), and templates.
  - **Ignore**: `.env` credentials, raw/processed datasets (`data/`), generated outputs (`submissions/`), active execution state (`SKILL_STATE.json`), and challenge configs (`challenge_config.json`).

---

## 2. Naming Conventions
* **Skills**: `skill_{NN}_{name}.py` where `NN` is a zero-padded, two-digit number (e.g., `skill_04_eda.py`). Exposed entry point is exactly `run(config: dict, state: dict) -> dict` returning the updated state.
* **Shared Skill Helpers**: Private modules prefixed with an underscore (e.g., `_lightgbm_shared.py`).
* **Tests**: `test_skill{NN}_{name}.py` or `test_{policy_or_feature}.py`.

---

## 3. Import & Dependency Isolation
* **No Cross-Skill Imports**: Skills must not import from other skill modules. The compatibility shim `skill_13_ensemble` importing `zindian.oracle_fusion_core` is the sole exception.
* **No AutoML**: Imports of libraries like `auto-sklearn`, `tpot`, `h2o`, `pycaret`, or `optuna.integration` are strictly prohibited.
* **No Hardcoded Competition Strings**: Column names, targets, metrics, and dataset identifiers must always be dynamically read from `challenge_config.json`.
* **Lockfile Integrity**: All imports must resolve to packages listed in `requirements.txt` (compiled from `requirements.in`).

---

## 4. Architectural Python Rules

### Safe State Access Patterns
Avoid `KeyError` crashes on fresh runs. Access dynamic or optional state keys exclusively via `.get()` with safe defaults:
```python
# CV Strategy Override
override_active = SKILL_STATE.get("cv_strategy_override", {}).get("active", False)
cv_strategy = SKILL_STATE["cv_strategy_override"]["override_strategy"] if override_active else config["cv_strategy"]["type"]

# Pseudo-Label Retraining & Anchor Challenge checks
retraining_active = SKILL_STATE.get("pseudo_label_result", {}).get("retraining_required", False)
challenge_active = SKILL_STATE.get("anchor_challenge", {}).get("active", False)

# Drift Threshold & Target Standard deviation
drift_threshold = SKILL_STATE.get("drift_threshold", config.get("drift_threshold", 0.05))
target_std = float((SKILL_STATE.get("eda", {}) or {}).get("target_std") or 0.0)

# Sidecar Recommendations
sidecar_recommendations = SKILL_STATE.get("sidecar_recommendations", [])
```

### OOF Output Contract & Namespace Isolation
Every OOF-generating skill must write record schemas via `write_oof_record()` from `zindian/state.py`.
```python
{
    "scores": oof_array.tolist(),
    "cv_strategy_id": cv_strategy_id,
    "seed": config["reproducibility"]["seed"],
    "branch_name": branch_name,
    "model_config": model_config_dict,
    "secondary_metrics": { "mae": float, "mape": float | None, "r2": float } # Regression only
}
```
* **Namespace Protection**: Retraining outputs must use the `_augmented` suffix (e.g., `branch_{name}_oof_augmented`). Retraining loops must never overwrite original OOF keys; doing so triggers a runtime abort. Rollbacks must purge only `_augmented` keys.

### SHAP Computation Rules
* SHAP values must be computed per-fold on validation fold splits only. Full-train SHAP is strictly forbidden.
* Select the positive class index or sum absolute values across classes for classification; use values directly for regression.
* **Single-Feature Fallback**: If features count is less than 2, skip SHAP ratio audit and log `SKILL_STATE["shap_audit_skipped_reason"] = "single_feature"`.

### Two-Mode Feature Aggregation
Target-dependent features must implement two execution modes to prevent validation leaks and inference shape mismatches:
```python
def compute_features(X, y, train_idx=None, mode="cv"):
    if mode == "cv":
        assert train_idx is not None
        X_fit, y_fit = X.iloc[train_idx], y.iloc[train_idx]
    else:
        X_fit, y_fit = X, y
    # Compute aggregates on (X_fit, y_fit) only
```

### Config Temporal Lock
`challenge_config.json` is strictly read-only post-Phase 1.
* Phase 1 allowed writers: `skill_01` (file hashes), `skill_02` (fingerprint), `skill_03` (policy filters), `skill_05` (cv strategy).
* **Sole post-Phase 1 Exception**: `skill_00` writing to `community_signals`.

### Seed Discipline
Enforce reproducible modeling by setting random seeds uniformly:
```python
seed = config["reproducibility"]["seed"]
random.seed(seed)
np.random.seed(seed)
model = LGBMClassifier(random_state=seed, ...)
```

---

## 5. Refactoring Integrity Workflow

All agents performing modifications must execute these steps sequentially to preserve pipeline correctness:

1. **Pre-Refactor Check**: Verify the Git state is clean and the test suite passes 100%. Read relevant specifications in `docs/source_of_truth.md`.
2. **Integrity Rule Validation**: Perform static checks to ensure the implementation adheres to safe state access patterns, seed discipline, import restrictions, and AutoML prohibitions.
3. **Incremental Implementation**: Modify code incrementally, starting with core utility levels. Maintain existing function contracts and entry-point signatures.
4. **Static Scan & Schema Audit**: Verify that output directories and generated schemas conform to expected types. Assert namespace protection on modified state writes.
5. **Test Verification**: Execute the policy contract tests followed by the full test suite locally. If introducing new logic, append unit tests under the `tests/` directory.
6. **Dependency Freshness Check**: Ensure any new imports are documented in `requirements.in` and compiled requirements match `requirements.txt` exactly.
7. **Audit Record**: Log refactoring changes and verification test results in `walkthrough.md` and the audit ledger before final check-in.
