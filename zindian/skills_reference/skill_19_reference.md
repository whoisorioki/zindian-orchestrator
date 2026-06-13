Skill 19 — The Code Miner (reference)
====================================

Purpose
-------
- Search public competition writeups and public repositories for reusable ML prior art.
- Extract patterns that are relevant to the active competition without scraping restricted sources.

Primary implementation
----------------------
- `zindian/skills/skill_19_code_miner.py`

Commands
--------
- Run the code miner:

  python3 -m zindian.skills.skill_19_code_miner

- Run a dry run without external API calls:

  python3 -m zindian.skills.skill_19_code_miner --dry-run

What it writes
--------------
- `competitions/<slug>/reports/ml_priorart.json`
- `competitions/<slug>/reports/code_miner_report.md`

Current behavior notes
----------------------
- Uses Gemini Flash to structure search results into tricks, validation strategies, feature ideas, and warnings.
- Supports multiple domains, including geospatial, tabular, frog ecology, and all.
- Updates `SKILL_STATE.json` with the last run metadata.
- Does not query Semantic Scholar; that API is reserved for Skill 18.

Notes
-----
- This is the evidence-collection stage of the research split.
- It should stay separate from hypothesis synthesis to preserve auditability.
