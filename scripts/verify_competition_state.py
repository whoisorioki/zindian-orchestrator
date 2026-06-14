#!/usr/bin/env python3
"""
verify_competition_state.py
============================
Ground-truth verification script for ey-frogs competition state.
Reads directly from files — never from memory or handoff docs.

Run from: ~/projects/zindian_orchestrator
Usage:    python scripts/verify_competition_state.py

Checks:
  1. challenge_config.json — metric, use_probabilities, slug
  2. SKILL_STATE.json — dag_phase, anchor scores, branch, budget
  3. SampleSubmission.csv — expected format (columns, dtypes)
  4. features_train.csv — shape, columns, ws presence
  5. features_test.csv  — shape, columns, ws presence
  6. All submission CSVs — format vs SampleSubmission
  7. skill_08_anchor.py — Check 8 logic (use_probabilities aware)
  8. Git branch state
"""

import json
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

ROOT = Path(__file__).parent.parent

try:
    from zindian.paths import resolve_competition_paths
    paths = resolve_competition_paths()
    COMP_DIR = paths.competition_dir or (ROOT / "competitions" / "ey-frogs")
except Exception:
    COMP_DIR = ROOT / "competitions" / "ey-frogs"

DATA_RAW = COMP_DIR / "data" / "raw"
DATA_PROC = COMP_DIR / "data" / "processed"
SUBS_DIR = COMP_DIR / "submissions"
SKILLS = ROOT / "zindian" / "skills"

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

errors = []
warnings = []


def section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def ok(msg: str):
    print(f"  {PASS} {msg}")


def fail(msg: str):
    print(f"  {FAIL} {msg}")
    errors.append(msg)


def warn(msg: str):
    print(f"  {WARN} {msg}")
    warnings.append(msg)


# ── 1. challenge_config.json ───────────────────────────────────────────────────
section("1. challenge_config.json")

config_path = COMP_DIR / "challenge_config.json"
if not config_path.exists():
    fail(f"challenge_config.json not found at {config_path}")
    config = {}
else:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    ok(f"Found: {config_path}")

    slug = config.get("slug")
    metric = config.get("metric")
    use_probs = config.get("use_probabilities")
    daily_limit = config.get("daily_limit")
    allowed_ext = config.get("allowed_external_data")

    print(f"\n  slug              : {slug}")
    print(f"  metric            : {metric}")
    print(f"  use_probabilities : {use_probs}")
    print(f"  daily_limit       : {daily_limit}")
    print(f"  allowed_external  : {allowed_ext}")

    if slug == "ey-frogs":
        warn(
            "slug='ey-frogs' — internal folder name, not real Zindi slug. "
            "Should be 'ey-biodiversity-challenge'"
        )
    elif slug == "ey-biodiversity-challenge":
        ok("slug = 'ey-biodiversity-challenge' ✓")
    else:
        warn(f"slug='{slug}' — verify this is correct on Zindi")

    if metric is None:
        fail("metric is null — Skill 02 must populate this")
    else:
        ok(f"metric = '{metric}'")

    if use_probs is None:
        fail("use_probabilities is null")
    else:
        ok(f"use_probabilities = {use_probs}")

    if use_probs is True and metric in ("f1", "F1", "f1_score"):
        fail("CONFLICT: use_probabilities=True but metric=F1 requires hard 0/1 labels")
    if use_probs is False and metric in ("logloss", "log_loss", "auc"):
        fail(
            f"CONFLICT: use_probabilities=False but metric={metric} needs probabilities"
        )


# ── 2. SKILL_STATE.json ────────────────────────────────────────────────────────
section("2. SKILL_STATE.json")

state_path = COMP_DIR / "SKILL_STATE.json"
if not state_path.exists():
    fail(f"SKILL_STATE.json not found at {state_path}")
    state = {}
else:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    ok(f"Found: {state_path}")

    dag_phase = state.get("dag_phase")
    branch = state.get("current_git_branch")
    anchor_rmse = state.get("anchor_oof_rmse")
    anchor_auc = state.get("anchor_oof_auc")
    anchor_lb = state.get("anchor_lb_score")
    subs_today = state.get("submissions_used_today")
    subs_total = state.get("submissions_used_total")
    remaining = state.get("remaining_submissions")
    selected = state.get("selected_submissions", [])
    use_probs_st = state.get("use_probabilities")

    print(f"\n  dag_phase            : {dag_phase}")
    print(f"  current_git_branch   : {branch}")
    print(f"  anchor_oof_rmse      : {anchor_rmse}")
    print(f"  anchor_oof_auc       : {anchor_auc}")
    print(f"  anchor_lb_score      : {anchor_lb}")
    print(f"  submissions_today    : {subs_today}")
    print(f"  submissions_total    : {subs_total}")
    print(f"  remaining            : {remaining}")
    print(f"  selected_submissions : {selected}")

    if anchor_rmse is None:
        warn("anchor_oof_rmse is null — Phase 2 not completed in state")
    if anchor_lb is None:
        warn("anchor_lb_score is null — update after Zindi scores submission")
    if not selected:
        warn("selected_submissions is empty — must select 2 before deadline")
    elif len(selected) != 2:
        warn(f"selected_submissions has {len(selected)} entries — must be exactly 2")


# ── 3. SampleSubmission.csv ────────────────────────────────────────────────────
section("3. SampleSubmission.csv — Expected Format")

sample_path = DATA_RAW / "SampleSubmission.csv"
if not sample_path.exists():
    fail(f"SampleSubmission.csv not found at {sample_path}")
    sample = None
else:
    sample = pd.read_csv(sample_path)
    ok(f"Found: {sample_path}")
    print(f"\n  Shape   : {sample.shape}")
    print(f"  Columns : {list(sample.columns)}")
    print(f"  Dtypes  :\n{sample.dtypes.to_string()}")
    print("\n  First 3 rows:")
    print(sample.head(3).to_string(index=False))

    pred_col = [c for c in sample.columns if c.upper() != "ID"]
    if pred_col:
        vals = sample[pred_col[0]]
        print(f"\n  Prediction col '{pred_col[0]}' range: [{vals.min()}, {vals.max()}]")
        unique = set(vals.unique())
        if unique.issubset({0, 1, 0.0, 1.0}):
            ok("SampleSubmission values are 0/1 — hard labels expected")
        else:
            ok("SampleSubmission values are floats — probabilities expected")


# ── 4. features_train.csv ─────────────────────────────────────────────────────
section("4. features_train.csv")

ft_path = DATA_PROC / "features_train.csv"
if not ft_path.exists():
    warn(f"features_train.csv not found at {ft_path} — Phase 3 not started?")
else:
    ft = pd.read_csv(ft_path)
    ok(f"Found: {ft_path}")
    print(f"\n  Shape   : {ft.shape}")

    ws_cols = [c for c in ft.columns if "ws" in c.lower()]
    if ws_cols:
        fail(f"ws (wind speed) columns present — BANNED: {ws_cols}")
    else:
        ok("No ws columns found ✓")

    spatial_derived = [
        c
        for c in ft.columns
        if any(
            x in c.lower()
            for x in [
                "distance",
                "cluster",
                "bin",
                "h3",
                "poly",
                "interact",
                "lat_lon",
                "lon_lat",
            ]
        )
    ]
    if spatial_derived:
        fail(f"Derived spatial columns detected — BANNED: {spatial_derived}")
    else:
        ok("No derived spatial columns found ✓")

    print(f"\n  Columns ({len(ft.columns)}):")
    for c in ft.columns:
        print(f"    {c}")


# ── 5. features_test.csv ──────────────────────────────────────────────────────
section("5. features_test.csv")

ftest_path = DATA_PROC / "features_test.csv"
if not ftest_path.exists():
    warn(f"features_test.csv not found at {ftest_path}")
else:
    ftest = pd.read_csv(ftest_path)
    ok(f"Found: {ftest_path}")
    print(f"\n  Shape   : {ftest.shape}")

    ws_cols_t = [c for c in ftest.columns if "ws" in c.lower()]
    if ws_cols_t:
        fail(f"ws columns in test — BANNED: {ws_cols_t}")
    else:
        ok("No ws columns in test ✓")

    # Column parity with train (excluding target)
    if ft_path.exists():
        train_non_target = [
            c for c in ft.columns if c not in ("Occurrence Status", "Target")
        ]
        test_cols = list(ftest.columns)
        missing_in_test = set(train_non_target) - set(test_cols)
        extra_in_test = set(test_cols) - set(train_non_target)
        if missing_in_test:
            fail(f"Columns in train but not test: {missing_in_test}")
        if extra_in_test:
            warn(f"Columns in test but not train: {extra_in_test}")
        if not missing_in_test and not extra_in_test:
            ok("Train/test column parity ✓")


# ── 6. All submission CSVs ────────────────────────────────────────────────────
section("6. Submission Files vs SampleSubmission")

sub_files = sorted(SUBS_DIR.glob("*.csv")) if SUBS_DIR.exists() else []
if not sub_files:
    warn("No submission CSVs found in submissions/")
else:
    use_probs_cfg = config.get("use_probabilities", True)

    for sub_path in sub_files:
        sub = pd.read_csv(sub_path)
        sub_errors = []

        # Column check
        if sample is not None:
            if list(sub.columns) != list(sample.columns):
                sub_errors.append(
                    f"columns {list(sub.columns)} ≠ expected {list(sample.columns)}"
                )

            # Row count
            if len(sub) != len(sample):
                sub_errors.append(f"rows {len(sub)} ≠ expected {len(sample)}")

        # Value check
        pred_col = [c for c in sub.columns if c.upper() != "ID"]
        if pred_col:
            vals = sub[pred_col[0]]
            unique = set(vals.round(8).unique())

            if use_probs_cfg:
                # Expect floats
                if unique.issubset({0, 1, 0.0, 1.0}):
                    sub_errors.append(
                        "thresholded (0/1 only) but use_probabilities=True"
                    )
                if vals.min() < 0 or vals.max() > 1:
                    sub_errors.append(
                        f"values out of [0,1]: [{vals.min():.4f}, {vals.max():.4f}]"
                    )
            else:
                # Expect hard labels
                invalid = unique - {0, 1, 0.0, 1.0}
                if invalid:
                    sub_errors.append(
                        f"non-binary values found but use_probabilities=False: {sorted(invalid)[:3]}"
                    )

        if sub_errors:
            fail(f"{sub_path.name}: {'; '.join(sub_errors)}")
        else:
            ok(f"{sub_path.name} ({len(sub)} rows) — format valid")


# ── 7. skill_08_anchor.py — Check 8 logic ─────────────────────────────────────
section("7. skill_08_anchor.py — Check 8 (use_probabilities aware?)")

skill08 = SKILLS / "skill_08_anchor.py"
if not skill08.exists():
    fail(f"skill_08_anchor.py not found at {skill08}")
else:
    src = skill08.read_text(encoding="utf-8")

    # Check if Check 8 is use_probabilities aware
    if "use_probabilities" in src and "issubset" in src:
        # Check if it's inside a use_probabilities conditional
        lines = src.splitlines()
        check8_lines = [
            (i, line) for i, line in enumerate(lines, 1) if "issubset" in line
        ]
        for lineno, line in check8_lines:
            print(f"\n  Check 8 found at line {lineno}:")
            # Show context (5 lines before)
            start = max(0, lineno - 6)
            context = lines[start:lineno]
            for cl in context:
                print(f"    {cl}")

        # Determine if the issubset check is inside use_probabilities block
        if "if config.use_probabilities" in src and "issubset" in src:
            # Check if there's an else branch for hard labels
            if "use_probabilities=False" in src or "hard" in src.lower():
                ok("Check 8 appears use_probabilities aware (has else branch)")
            else:
                fail(
                    "Check 8 blocks thresholded submissions regardless of "
                    "use_probabilities — will block F1 submissions (hard 0/1 labels)"
                )
        else:
            fail(
                "Check 8 not inside use_probabilities conditional — "
                "will block correct F1 submissions"
            )
    else:
        warn("Could not find Check 8 (issubset) in skill_08_anchor.py")


# ── 8. Git branch state ────────────────────────────────────────────────────────
section("8. Git Branch State")

try:
    branch_out = subprocess.run(
        ["git", "branch", "-v"], capture_output=True, text=True, cwd=ROOT
    ).stdout
    print(branch_out)

    log_out = subprocess.run(
        ["git", "log", "--oneline", "-5"], capture_output=True, text=True, cwd=ROOT
    ).stdout
    print(log_out)
except Exception as e:
    warn(f"Git check failed: {e}")


# ── Summary ────────────────────────────────────────────────────────────────────
section("SUMMARY")
print(f"\n  Errors   : {len(errors)}")
for e in errors:
    print(f"    {FAIL} {e}")

print(f"\n  Warnings : {len(warnings)}")
for w in warnings:
    print(f"    {WARN} {w}")

if not errors:
    print(f"\n  {PASS} No blocking errors found.")
else:
    print(f"\n  {FAIL} Fix {len(errors)} error(s) before submitting.")

print()
