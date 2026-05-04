---
description: "Skill 09 — Granular Calibration (residual mean matching)"
---

## Formula
\n\(final\\_pred_{group} = pred + (\\frac{1}{n} \\sum_{i=1}^{n} (y_{true} - y_{pred}))_{group}\)\n+
## Rules
- Granularity is derived from available categorical columns; do not hardcode group keys.
- Never threshold predictions if `challenge_config.use_probabilities` is true.

