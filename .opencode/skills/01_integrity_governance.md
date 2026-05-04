---

slug-aware: true
state-path: competitions/<slug>/SKILL_STATE.json


## description: "Skill 01 — Integrity Governance (MD5 lock + verify)"

## Preconditions

- `competitions/<slug>/challenge_config.json` exists and is valid.
- `competitions/<slug>/SKILL_STATE.json` exists.
- Data must download into `competitions/<slug>/data/raw/` and remain immutable.

## Steps

- Use the Zindi client to download the dataset for the active competition into `competitions/<slug>/data/raw/`.
- Identify the target column from competition rules (never guess).
- Compute MD5 hash of the **target column values** in training data.
- Write the hash to `competitions/<slug>/SKILL_STATE.json.md5_target_hash`.
- Set `dag_phase` to `phase_1_integrity_locked`.

## Invariants

- Before any transform/feature generation later, verify the MD5 matches.