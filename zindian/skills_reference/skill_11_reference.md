Skill 11 — Branch Gate (reference)
=================================

Purpose
-------
- Promote the best passing feature variant to a new anchor branch.
- Block promotion when no variant passed the gate.

Primary implementation
----------------------
- `zindian/skills/skill_11_gate.py`

Commands
--------
- Run the promotion gate:

  python3 -m zindian.skills.skill_11_gate

What it writes
--------------
- Creates or checks out `anchor-v<N>` via git.
- Updates `competitions/<slug>/SKILL_STATE.json` with the promoted anchor metadata.

Current behavior notes
----------------------
- Evaluates the 5 promotion conditions for variant promotion.
- Reads `metric_direction` from `ChallengeConfig` ("maximize" or "minimize").
- Support Gate 1 overrides (`cv_strategy_override`) and anchor challenges.

Findings (issues / misalignments)
---------------------------------
- **Regression Gating Thresholds**: Gating must use scale-invariant thresholds for regression:
  - For RMSE (regression):
    - `effective_variance_threshold = config["variance_gate_threshold"] * (target_std ** 2)`
    - `effective_gate_margin = config["gate_margin"] * target_std`
  - For RMSLE (regression) and Classification:
    - `effective_variance_threshold = config["variance_gate_threshold"]`
    - `effective_gate_margin = config["gate_margin"]`
  - Unbiased target standard deviation `target_std` must be read from `state["eda"]["target_std"]`.
- **Directional Check Inversion**: For minimize metrics (RMSE/RMSLE), variant improvement means:
  `baseline_score - variant_score > effective_gate_margin`.
- **Safe State Access**: Baseline lookups must safely parse `pseudo_label_result` and `anchor_challenge` blocks to avoid KeyErrors before they are initialized.

Recommendations
---------------
- Implement inverted gating comparisons and target standard deviation scaling for regression metrics (RMSE).
- Ensure safe `.get()` access patterns are used on all optional keys in state.

