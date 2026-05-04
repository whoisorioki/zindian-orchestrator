---

## description: "Skill 08 — Anchor Baseline (first governed submission)"

## Rules

- Check `user.remaining_subimissions` before submit.
- Comments must be **JSON string payloads** (v2): e.g. `{\"branch\":\"anchor\",\"oof\":0.34,\"features\":123,\"calib\":\"none\"}`.
- Only submit if branch beats `anchor_oof_`* gating rules once anchor exists.

## Output

- Train baseline → compute OOF metric → produce `submissions/sub_001_anchor.csv` under `competitions/<slug>/submissions/`.
- After submit: record LB score and rank in ledger and (when enabled) GitHub Issue table.