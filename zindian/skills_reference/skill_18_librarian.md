Skill 18 — The Librarian
========================

Purpose
-------
- Retrieve and cache literature relevant to the active competition.
- Ground paper search in the actual TerraClimate variables, region, and temporal window.

Primary implementation
----------------------
- `zindian/skills/skill_18_librarian.py`

Commands
--------
- Run the librarian cache build:

  python3 -m zindian.skills.skill_18_librarian

What it writes
--------------
- `competitions/<slug>/reports/literature_cache.json`

Current behavior notes
----------------------
- Builds Semantic Scholar queries from TerraClimate variables and competition context.
- Deduplicates papers by `paperId` and keeps entries with abstracts or TLDRs.
- Caches the search results for downstream synthesis in Skill 19.
- This is the only skill that should call Semantic Scholar directly.

Notes
-----
- This is the evidence-collection stage of the research split.
- It should stay separate from hypothesis synthesis to preserve auditability.

Configuration
-------------
- Set the Semantic Scholar API key in your local `.env` as `SEMANTIC_SCHOLAR_API_KEY` (see [.env.example](.env.example)).
- The `zindian/clients/semantic_scholar.py` helper reads the env var and enforces a safe 1 req/sec throttle.
  Use this helper from the librarian (`skill_18_librarian.py`) to centralize API calls and caching.

Usage notes
-----------
- When authoring queries, run the librarian with `--dry-run` to validate query construction without issuing
  external requests. Once satisfied, run the full job with the key present in `.env`.
- Do not commit `.env` to source control.
