Skill 20 — The Scientist
========================

Purpose
-------
- Turn ecological hypotheses and prior art into bounded, testable feature proposals.
- Run static validation and empirical validation before promoting hypotheses to Skill 07.

Primary implementation
----------------------
- `zindian/skills/skill_20_scientist.py`

Commands
--------
- Run the scientist:

  python3 -m zindian.skills.skill_20_scientist

What it writes
--------------
- `competitions/<slug>/reports/validated_hypotheses.json`
- `competitions/<slug>/reports/failed_hypotheses.json`

Current behavior notes
----------------------
- Reads `domain_hypotheses.json` and `ml_priorart.json` from the earlier research stages.
- Runs Stage 1 static validation and Stage 2 empirical validation.
- Writes rejected hypotheses to `failed_hypotheses.json` with `do_not_retry` flags so future runs can skip them.

Notes
-----
- This is the synthesis and validation stage of the research split.
- Keep it separate from the retrieval steps so evidence remains traceable.
