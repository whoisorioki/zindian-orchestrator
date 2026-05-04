# Zindian Orchestrator — Operator Guide (Cursor)

An autonomous ML competition agent for Zindi Africa competitions.

## Non-Negotiable Rules (always enforce)
1. Read `challenge_config.json` before touching any data.
2. Read and write `SKILL_STATE.json` at every state change.
3. Check `user.remaining_subimissions` before every Zindi submit call.
4. Lock MD5 hash of target column at Skill 01 — verify before every transform.
5. Submission comments must follow: `branch:X|oof_rmse:X|features:N|calib:X`
6. Gate every branch — only submit if OOF RMSE beats `anchor_oof_rmse` in `SKILL_STATE.json`.
7. Select exactly 2 submissions for private judging — log rationale in `reports/`.
8. Never apply physical domain constraints unless `challenge_config.domain` confirms it.
9. Never threshold predictions if `challenge_config.use_probabilities` is true.
10. Never use AutoML — `challenge_config.automl_permitted` is almost always false on Zindi.

## Phase map (what "next task" means)
- **Phase 0 — Foundation**: wiring + auth + ledger + skeleton configs (no ML)
- **Phase 1 — Integrity + Intake**: Skill 01 (MD5 lock), Skill 02 (rules intake), Skill 15 (reporter)
- **Phase 2 — Anchor Baseline**: EDA → baseline LightGBM → anchor submit (budget-checked)
- **Phase 3 — Features + Calibration**: governed feature work + SHAP + calibration
- **Phase 4 — Branch + Gate**: branches must pass gate before submission
- **Phase 5 — Fusion + Final Submit**: fusion only after GO, select exactly 2 for private

## Session-start prompt (paste at start of every session)
```
Read SKILL_STATE.json and challenge_config.json.
Tell me what phase we are in, what competition is active,
how many submissions remain today, and what the next task
is according to the phase map in AGENTS.md.
Then proceed in Plan mode.
```

When the plan is approved:
```
The plan looks good. Switch to Build mode and implement it.
After each file is created, update SKILL_STATE.json with the current dag_phase.
```
