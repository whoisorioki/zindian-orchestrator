Skill 22 — Reproducibility & Integration Audit (reference)
==========================================================

Purpose
-------
- Audits the pipeline state, git branch tracking, dependencies, and outputs to ensure complete reproducibility.
- Performs AST imports scan over all skill scripts to verify the absence of forbidden AutoML libraries.
- Checks that `requirements.txt` is updated and fully contains all top-level requirements in `requirements.in`.
- Audits all OOF records in `SKILL_STATE.json` to ensure their `cv_strategy_id` matches the active CV strategy.

Primary implementation
----------------------
- `zindian/skills/skill_22_reproducibility_audit.py`

Inputs
------
- File paths of `requirements.in`, `requirements.txt`.
- Codebase scripts in `zindian/skills/`.
- `SKILL_STATE.json` and `challenge_config.json` for the active competition.

Outputs
-------
- Prints detailed report to stdout.
- Writes/updates `SKILL_STATE.json["reproducibility_audit"]` with `{ "success": bool, "timestamp": ISO_string }`.
- Returns exit code 0 if secure/reproducible, 1 otherwise.

Commands
--------
- Run audit:
  ```bash
  python3 -m zindian.skills.skill_22_reproducibility_audit
  ```
- Run audit for a specific competition slug:
  ```bash
  python3 -m zindian.skills.skill_22_reproducibility_audit <slug>
  ```

Audit findings & resolution status
----------------------------------
- **AST Scan Coverage**: The AST scan successfully blocks forbidden AutoML packages.
- **Requirements Sync Verification**: Correctly checks lockfile package presence, but may flag false positives/stale alerts if `st_mtime` gets reset during git checkouts or file copies.
- **OOF Tag Verification**: [RESOLVED] Checks that OOF records have a `cv_strategy_id` matching the active strategy. Normalized CV strategy prefixes (stripping `config:` or `override:`) in `_audit_oof_strategy_tags` before matching, completely preventing false-positive mismatches.

