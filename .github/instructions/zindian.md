# Zindian Orchestrator — Operator Guide (Cursor)

An autonomous ML competition agent for Zindi Africa competitions.

## Working Contract
- Treat `challenge_config.json` and `SKILL_STATE.json` as the source of truth before any planning or code changes.
- If `dag_phase` is `uninitialized`, stop and ask the user to run `tabula init` first.
- Before every submission-related action, verify the current budget and the competition rules in `challenge_config.json`.
- Follow the active handoff or phase script, but validate it against the live workspace state before acting.
- When a rule is ambiguous, ask a focused clarification question instead of guessing.
- Prefer folder-based organization for new work: place reusable logic in package folders, reports in `reports/`, templates in `templates/`, and competition-specific artifacts only in the competition workspace.
- Keep file names and folder names consistent with the established repo convention: numbered notebooks, `skill_XX_*.py`, and explicit per-purpose directories.

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

## Compliance Rules For Workspace Work
- Do not violate `use_probabilities=True`; always keep predictions probabilistic when required.
- Do not add banned or derived spatial features when the competition forbids them.
- Do not use external data or AutoML unless the live config explicitly allows it.
- Keep branch-per-experiment discipline and update `SKILL_STATE.json` after meaningful state changes.
- If a change touches the submission path, confirm the submission comment format and remaining budget first.
- Do not write new code into the repository root when a suitable folder already exists.
- Do not hardcode dataset filenames or paths inside skills; resolve them from config, state, or the competition workspace layout.
- When creating a new file, match the existing naming scheme instead of inventing a one-off name.

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

## Progress Reporting Style
- Give short, factual progress updates while working.
- Track the user’s workflow by reporting the current phase, current branch, next task, blockers, and the check you are about to run.
- Summarize reasoning as brief, testable hypotheses and checks.
- Do not expose hidden chain-of-thought; provide concise decision summaries instead.
