"""Verify all SoT v2.2 contract logic pathways exist in the modified files.
Uses whitespace-tolerant regex patterns to handle black auto-formatting."""

import re
import sys


def _check_text(text: str, pattern: str) -> bool:
    """Whitespace-tolerant exact match: collapse all whitespace before comparing."""
    return re.sub(r"\s+", "", pattern) in re.sub(r"\s+", "", text)


def _check_regex(text: str, pattern: str) -> bool:
    """Regex match with re.DOTALL."""
    return bool(re.search(pattern, text, re.DOTALL))


# Check _lightgbm_shared.py
with open("zindian/skills/_lightgbm_shared.py", encoding="utf-8") as f:
    src = f.read()

checks = [
    ("log1p for rmsle training", _check_text(src, "np.log1p(y[tr_idx])")),
    (
        "expm1 after clip for rmsle",
        _check_text(src, "np.expm1(np.clip(val_pred_flat, 0, None))"),
    ),
    (
        "domain bounds clipping for RMSE/MAE",
        _check_text(src, "np.clip(val_pred_flat, domain_bounds[0], domain_bounds[1])"),
    ),
    ("regression_metric parameter defined", _check_text(src, "regression_metric")),
    ("use_log1p flag from metric", _check_text(src, 'use_log1p = metric == "rmsle"')),
    (
        "Correct RMSLE score formula",
        _check_text(src, "np.sqrt(np.mean((np.log1p(y) - np.log1p(oof_probs)) ** 2))"),
    ),
]

# Check skill_08_anchor.py
with open("zindian/skills/skill_08_anchor.py", encoding="utf-8") as f:
    src08 = f.read()

checks += [
    (
        "regression_metric from config",
        _check_text(
            src08, 'config.get("metric") if task_type == "regression" else None'
        ),
    ),
    ("root_mean_squared_error in metric_map", "root_mean_squared_error" in src08),
    ("mean_absolute_error in metric_map", "mean_absolute_error" in src08),
    (
        "train_lightgbm_cv called with regression_metric",
        _check_text(src08, "regression_metric=regression_metric"),
    ),
]

# Check skill_11_gate.py
with open("zindian/skills/skill_11_gate.py", encoding="utf-8") as f:
    src11 = f.read()

checks += [
    (
        "SCALE_INVARIANT_METRICS (rmsle)",
        _check_text(src11, 'SCALE_INVARIANT_METRICS = frozenset({"rmsle"})'),
    ),
    (
        "SCALE_SENSITIVE_METRICS defined",
        _check_text(src11, "SCALE_SENSITIVE_METRICS = frozenset("),
    ),
    ("root_mean_squared_error in sensitive", "root_mean_squared_error" in src11),
    ("mean_absolute_error in sensitive", "mean_absolute_error" in src11),
    (
        "RMSLE returns raw thresholds",
        _check_text(src11, "return variance_gate_threshold, gate_margin, None"),
    ),
    (
        "variance scaling by target_std^2",
        _check_text(src11, "variance_gate_threshold * (target_std**2)"),
    ),
    ("margin scaling by target_std", _check_text(src11, "gate_margin * target_std")),
    (
        "degenerate target_std warning",
        "Degenerate target_std (0.0) in skill_11_gate" in src11,
    ),
    (
        "catch-all for unknown regression metrics",
        _check_text(src11, "metric not in SCALE_INVARIANT_METRICS"),
    ),
]

# Check docs/source_of_truth.md
with open("docs/source_of_truth.md", encoding="utf-8") as f:
    so_src = f.read()

checks += [
    ("SoT v2.2 version header", "2.2-Generalized-Regression" in so_src),
    (
        "SoT Regression Target Transformation Lifecycle section",
        "Regression Target Transformation Lifecycle" in so_src,
    ),
    ("rmsle in SoT lifecycle", '"rmsle"' in so_src),
    ("root_mean_squared_error in SoT lifecycle", "root_mean_squared_error" in so_src),
    ("mean_absolute_error in SoT lifecycle", "mean_absolute_error" in so_src),
]

all_pass = True
failures = []
for name, result in checks:
    status = "PASS" if result else "FAIL"
    if not result:
        all_pass = False
        failures.append(name)
    print(f"{status}: {name}")

print("\n=== Summary ===")
print(f"Total checks : {len(checks)}")
print(f"Passed       : {len(checks) - len(failures)}")
print(f"Failed       : {len(failures)}")
if failures:
    print("Failures:")
    for failure in failures:
        print(f"  - {failure}")
print(f"\nOverall: {'ALL PASS ✅' if all_pass else 'SOME FAILED ❌'}")
sys.exit(0 if all_pass else 1)
