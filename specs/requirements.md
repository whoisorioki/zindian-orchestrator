# Zindian Orchestrator — Requirements

## Purpose

An autonomous ML competition agent that can participate in any Zindi Africa competition. It adapts to each competition's unique rules, metric, and domain constraints rather than hardcoding assumptions.

---

## Functional Requirements

### REQ-01: Competition Agnosticism

The agent **MUST** read `challenge_config.json` before any data operation.

The agent **MUST NOT** hardcode assumptions from any specific competition.

Every skill's behaviour **MUST** be conditional on `challenge_config` fields:
- `metric` — determines CV scoring strategy
- `use_probabilities` — determines if predictions are raw or thresholded
- `domain` — determines if physical constraints apply (e.g., nighttime solar zeros)
- `automl_permitted` — almost always false on Zindi
- `allowed_external_data` — determines if external sources are permitted

Skills **MUST** guard every operation: `if config.get("domain") == "solar": apply_nighttime_zeros()`

---

### REQ-02: Data Integrity Lock

The agent **MUST** compute and lock the MD5 hash of the target column at Skill 01 (session start).

The agent **MUST** verify this hash before every data transformation.

The hash **MUST** be persisted to `SKILL_STATE.json` as `md5_target_hash`.

If the hash shifts between sessions, the agent **MUST** halt and alert: "Target column modified — integrity breach."

---

### REQ-03: Submission Governance

The agent **MUST** check `remaining_submissions` from Zindi API before every submit call.

The agent **MUST NOT** proceed to submit if `remaining_submissions <= 0`.

The agent **MUST** structure every submission comment as: `branch:X|oof_rmse:X|features:N|calib:X`

Example: `branch:feature_v2|oof_rmse:0.248|features:42|calib:isotonic`

---

### REQ-04: Branch Gating

The agent **MUST** compute out-of-fold (OOF) RMSE for every branch locally before submitting.

The agent **MUST** only submit a branch if OOF RMSE beats `anchor_oof_rmse` by **0.5% or more**.

Formula: `(anchor_oof - branch_oof) / anchor_oof >= 0.005`

Pruned branches (that fail the gate) **MUST** be logged to the DuckDB ledger with reason: `GATE_FAIL`.

---

### REQ-05: Reproducibility

Every notebook and script **MUST** run top-to-bottom on original (unmodified) data to reproduce its submission.

No hardcoded local paths. No manual interventions. No cached intermediates.

Every submission CSV **MUST** be regenerable from the original data by running the notebook end-to-end.

---

### REQ-06: Physical Domain Guards (Conditional)

Physical domain constraints (e.g., nighttime solar zeros, TOA clipping for radiation) **MUST** only apply when `challenge_config.domain` explicitly confirms it.

Generic tabular competitions **MUST NOT** have these constraints applied blindly.

Example guard:
```python
if config.get("domain") == "solar":
    predictions[nighttime_mask] = 0
else:
    # no constraint applied
```

---

### REQ-07: Probability Handling

If `challenge_config.use_probabilities` is true, the agent **MUST** output raw probabilities (0 to 1).

The agent **MUST NOT** apply any threshold, rounding, or binary conversion.

If `use_probabilities` is false, predictions **MUST** be treated as continuous values or discrete classes depending on the metric.

---

## Non-Functional Requirements

### NFR-01: Tool Agnosticism

The agent **MUST** run identically whether invoked from VS Code, Cursor, Windsurf, Kiro, OpenCode, or the terminal directly.

Tool-specific UI (e.g., Copilot chat sidebar) **MUST** not affect core logic.

Core logic lives in Python files and JSON state, not in IDE extensions.

---

### NFR-02: State Durability

`SKILL_STATE.json` **MUST** be updated after every state change (skill completion, phase transition, submission, etc.).

At session start, the agent **MUST** read `SKILL_STATE.json` to resume from where it left off.

`SKILL_STATE.json` is the single source of truth for:
- Current `dag_phase` (Phase 0–5)
- `md5_target_hash` (locked target column checksum)
- `anchor_oof_rmse` (baseline performance)
- `submissions_used_today` and `submissions_used_total` (budget tracking)
- `selected_submissions` (list of exactly 2 for private judging)

---

### NFR-03: Security & Credentials

Credentials (Zindi API keys, etc.) **MUST** only live in `.env`.

Credentials **MUST** never be hardcoded in Python files or notebooks.

Credentials **MUST** never be committed to git.

All credential access **MUST** use `os.getenv("ZINDI_API_KEY")` or similar.

---

## Derived Rules (From 10 Non-Negotiable Rules in AGENTS.md)

| Rule # | Principle | Validation |
|--------|-----------|-----------|
| 1 | Read `challenge_config.json` before any data | REQ-01 |
| 2 | Read/write `SKILL_STATE.json` at state change | NFR-02 |
| 3 | Check `remaining_submissions` before submit | REQ-03 |
| 4 | Lock MD5 hash of target at Skill 01 | REQ-02 |
| 5 | Submission comment format | REQ-03 |
| 6 | Gate every branch before submit | REQ-04 |
| 7 | Select exactly 2 submissions for private | REQ-03 (extension) |
| 8 | Never apply physical constraints blindly | REQ-06 |
| 9 | Never threshold probabilities | REQ-07 |
| 10 | Never use AutoML | REQ-01 (competition agnosticism) |

---

## Acceptance Criteria

The agent implementation is complete when:

- ✅ `zindian_orchestrator/state.py` reads/writes `SKILL_STATE.json` atomically
- ✅ `zindian_orchestrator/config.py` reads `challenge_config.json` and raises `ConfigNotPopulated` if required fields are null
- ✅ All skills check `config.get()` before applying domain-specific logic
- ✅ Skill 01 computes and locks MD5; Skill 02 verifies it
- ✅ `zindian_orchestrator/zindi_client.py` checks `remaining_submissions` before every submit call
- ✅ Every submission comment follows: `branch:X|oof_rmse:X|features:N|calib:X`
- ✅ Branch gating is enforced: no submit if OOF RMSE doesn't beat anchor by 0.5%
- ✅ All 5 phases (0–5) are executable and reach completion
- ✅ Final submission includes exactly 2 selections with rationale logged to `reports/`
