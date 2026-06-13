**Skill 03 — Legality Gate**

**Purpose:** Synthesize a competition-level `feature_policy` from `zindi_monitor.json` and `challenge_config.json`, then evaluate planned features producing PASS/WARN/BLOCK decisions. Writes `feature_policy.json` and `legality_report.md` and updates `SKILL_STATE.json` (without downgrading phase).

**Run:**
- `python3 -m zindian.skills.skill_03_legality` — runs against the active competition (reads reports and state from `competitions/<slug>/`).

**Key functions:**
- `synthesise_feature_policy(monitor_data, config, flagged_titles) -> dict` — returns policy keys: `allowed_sources`, `banned_transformations`, `lat_lon_permitted_as_feature`, `external_data_permitted`, `automl_permitted`, `use_probabilities`, `metric`, `output_format`, `synthesised_at`, `source_flags`.
- `check_planned_features(policy, planned_features) -> list[dict]` — for each feature returns `{name, status: PASS|WARN|BLOCK, reason, blocks}`. `blocks=True` indicates DAG should not progress.
- `run(slug=None, planned_features=None) -> dict` — orchestrates both steps, writes outputs, updates state.

**Inputs:**
- `competitions/<slug>/reports/zindi_monitor.json` (optional but recommended)
- `challenge_config.json` (via `ChallengeConfig.load()`)
- `competitions/<slug>/SKILL_STATE.json` (for `planned_features` fallback)

**Outputs:**
- `competitions/<slug>/reports/feature_policy.json`
- `competitions/<slug>/reports/legality_report.md`
- `competitions/<slug>/SKILL_STATE.json` fields updated: `legality_status`, `feature_policy_written`, `last_legality_checked` (and `dag_phase` only when advancing non-downgrading).

**Behavior & Safety:**
- Does not hardcode dataset names; builds policy from monitor and config.
- Treats spatial/derived spatial bans as blocking latitude/longitude usage.
- External data is blocked only when monitor flags `external_banned`.
- Avoids downgrading `dag_phase` — will only advance `dag_phase` when appropriate.

**Human gating:**
- If any checked feature returns `blocks=True`, the skill returns `status: BLOCKED` and the agent must prompt a human before proceeding to feature experiments or submissions.

**Notes:**
- If `planned_features` is not provided, the skill will attempt to read `planned_features` from `SKILL_STATE.json`, then fall back to `anchor_features` if present.
- Keep `zindi_monitor.json` up-to-date (run Skill 00) for most accurate policy extraction.
