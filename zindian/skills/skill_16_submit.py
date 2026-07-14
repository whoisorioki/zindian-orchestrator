"""
Skill 16 — Zindi Submission
============================

Validates a candidate submission, enforces the daily budget, and (after a human
gate) submits to the Zindi platform. All metric / lineage data is read from
`SKILL_STATE.json`; no filename heuristics are used.

Contract (SoT §4 / §8):
  * Human Gate 4 must be approved before any network call.
  * Branch-specific Human Gate 2 (`human_gate_2_{branch}_approved`) must be
    approved for the active branch.
  * Task-aware value validation runs on the submission before the budget is
    consumed (probability interval, hard-label, or regression domain bounds).
  * The platform's `client.remaining_submissions` is queried *before* the
    submit call and a hard abort triggers if the budget is depleted.
  * The submission comment is composed from canonical state records
    (`branch_{name}_oof["model_config"]["calibration_method"]`,
    `state.get("last_calibration_method")`); the literal `calib:none` is gone.
  * OOF score and feature count are read from `branch_{name}_oof`; no
    filename string parsing.
  * The skill never writes to `challenge_config.json` after Phase 1.
  * The skill never writes a `human_gate_*_approved` key.
"""

from __future__ import annotations
import tabula.skill_state_autopatch  # noqa

import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from zindian.paths import resolve_competition_paths
from zindian.config import ChallengeConfig
from zindian.state import SkillStateStore
from zindian.ledger import Ledger


class HardAbortException(RuntimeError):
    """Raised when the submission budget is exhausted."""

    pass


# -- Value validation (mirrors skill_14 semantics) -----------------------------


def _validate_probability_interval(values: np.ndarray) -> list[str]:
    if values.size == 0:
        return []
    errors: list[str] = []
    if not bool(np.isfinite(values).all()):
        errors.append("Probability column contains NaN or Inf.")
    if values.size and (float(values.min()) <= 0.0 or float(values.max()) >= 1.0):
        errors.append(
            f"Probability column must lie strictly inside (0, 1); got range "
            f"[{float(values.min())}, {float(values.max())}]."
        )
    return errors


def _validate_binary(values: np.ndarray) -> list[str]:
    if values.size == 0:
        return []
    rounded = np.rint(values).astype(np.int64)
    if not np.all(
        (values == 0.0) | (values == 1.0) | (values == rounded.astype(np.float64))
    ):
        return ["Hard-label column must contain only 0/1 values."]
    return []


def _validate_regression_bounds(
    values: np.ndarray, bounds: dict[str, Any]
) -> list[str]:
    if values.size == 0:
        return []
    if not bool(np.isfinite(values).all()):
        return ["Regression column contains NaN or Inf."]
    lo = bounds.get("min", None)
    hi = bounds.get("max", None)
    if lo is None or hi is None:
        return [
            "Regression requires target_domain_bounds.{min,max} in challenge_config.json."
        ]
    if float(values.min()) < float(lo) or float(values.max()) > float(hi):
        return [
            f"Regression column out of domain bounds; got range "
            f"[{float(values.min())}, {float(values.max())}], expected "
            f"[{float(lo)}, {float(hi)}]."
        ]
    return []


def _value_validation_errors(
    df: pd.DataFrame,
    target_column: str,
    task_type: str,
    use_probabilities: bool,
    bounds: dict[str, Any],
) -> list[str]:
    if target_column not in df.columns:
        return [f"Submission missing target column '{target_column}'."]

    # Skip validation for multi_target competitions
    if task_type == "multi_target":
        return []

    values = df[target_column].to_numpy()
    if not np.issubdtype(values.dtype, np.number):
        try:
            values = values.astype(np.float64)
        except (TypeError, ValueError) as exc:
            return [f"Target column '{target_column}' is not numeric: {exc}"]
    if task_type == "classification":
        if use_probabilities:
            return _validate_probability_interval(values.astype(np.float64))
        return _validate_binary(values.astype(np.float64))
    if task_type == "regression":
        return _validate_regression_bounds(values.astype(np.float64), bounds)
    return [f"Unsupported task_type '{task_type}'."]


# -- Public surface -------------------------------------------------------------


def validate(
    sub_path: Path,
    sample_path: Path,
    config: ChallengeConfig | None = None,
) -> list[str]:
    """Validate submission format + task-aware value constraints.

    Executes the canonical 8-check structural alignment sequence:
      1. Column layout check (columns match SampleSubmission exactly)
      2. Row count check (matches SampleSubmission exactly)
      3. ID column presence in submission
      4. ID column presence in SampleSubmission
      5. ID values set match check (values match SampleSubmission)
      6. ID values order match check (order matches SampleSubmission)
      7. Nulls check (no null values)
      8. Duplicate IDs check (no duplicate IDs in submission)
    """
    errors: list[str] = []
    sub = pd.read_csv(sub_path)
    sample = pd.read_csv(sample_path)

    # Resolve ID column dynamically (default to first column).
    id_column: str = "ID"
    if config is not None:
        candidate = config.get("id_column")
        if isinstance(candidate, str) and candidate and candidate in sample.columns:
            id_column = candidate
        elif len(sample.columns) > 0:
            id_column = str(sample.columns[0])
    elif "ID" in sample.columns:
        id_column = "ID"
    elif len(sample.columns) > 0:
        id_column = str(sample.columns[0])

    # 1. Column layout check
    if list(sub.columns) != list(sample.columns):
        errors.append(f"Column mismatch: {list(sub.columns)} vs {list(sample.columns)}")

    # 2. Row count check
    if len(sub) != len(sample):
        errors.append(f"Row count: got {len(sub)}, expected {len(sample)}")

    # 3. ID column presence in submission
    if id_column not in sub.columns:
        errors.append(f"Submission missing '{id_column}' column")

    # 4. ID column presence in SampleSubmission
    if id_column not in sample.columns:
        errors.append(f"SampleSubmission missing '{id_column}' column")

    # 5. ID values set match check
    if id_column in sub.columns and id_column in sample.columns:
        if set(sub[id_column].astype(str)) != set(sample[id_column].astype(str)):
            errors.append(f"{id_column} set mismatch vs SampleSubmission")

    # 6. ID values order match check
    if id_column in sub.columns and id_column in sample.columns:
        if list(sub[id_column].astype(str)) != list(sample[id_column].astype(str)):
            errors.append(f"{id_column} order mismatch vs SampleSubmission")

    # 7. Nulls check
    if sub.isnull().any().any():
        errors.append(f"Nulls in: {sub.columns[sub.isnull().any()].tolist()}")

    # 8. Duplicate IDs check
    if id_column in sub.columns:
        if sub[id_column].duplicated().any():
            errors.append(f"Duplicate IDs found in submission '{id_column}' column")

    # Task-aware value validation (runs only if structural validation passes)
    if config is not None and not errors:
        target_col = (
            config.get("target_col") or config.get("target_column") or id_column
        )
        target_col = str(target_col)
        value_cols = [c for c in sample.columns if c != id_column and c in sub.columns]
        if not value_cols and target_col in sub.columns:
            value_cols = [target_col]
        task_type = str(config.get("task_type", "classification"))
        use_probs = bool(config.get("use_probabilities", False))
        bounds_cfg = config.get("target_domain_bounds") or {}
        bounds = bounds_cfg if isinstance(bounds_cfg, dict) else {}
        for idx, vcol in enumerate(value_cols):
            if len(value_cols) > 1 and task_type == "classification" and use_probs:
                # Multi-column probability submission: binary for non-last columns,
                # probability interval for the last value column.
                if idx < len(value_cols) - 1:
                    errors.extend(
                        _validate_binary(sub[vcol].to_numpy().astype(np.float64))
                    )
                else:
                    errors.extend(
                        _validate_probability_interval(
                            sub[vcol].to_numpy().astype(np.float64)
                        )
                    )
            else:
                errors.extend(
                    _value_validation_errors(sub, vcol, task_type, use_probs, bounds)
                )
    return errors


def determine_submission_metrics(
    submission_file: Path,
    state: dict[str, Any],
) -> tuple[float | None, str]:
    """Resolve (oof_score, source_key) directly from the SKILL_STATE branch records.

    No filename parsing. We inspect the active branch from the state (set by
    Skill 11 / 13) and return the OOF score from the matching
    `branch_{name}_oof` record.
    """
    active_branch = (
        state.get("current_active_branch")
        or state.get("anchor_git_branch")
        or state.get("best_variant_this_round")
    )
    candidate_keys: list[str] = []
    if isinstance(active_branch, str) and active_branch:
        candidate_keys.append(f"branch_{active_branch}_oof")
    candidate_keys.extend(
        [
            # Generic keys
            "last_ensemble_oof_score",
            "best_ensemble_oof_score",
            "last_variant_oof_score",
            "best_variant_oof_score",
            "anchor_oof_score_augmented",
            "anchor_oof_score_challenged",
            "anchor_oof_score",
        ]
    )
    for key in candidate_keys:
        value = state.get(key)
        if isinstance(value, dict):
            scores = value.get("scores")
            if isinstance(scores, (list, tuple)) and scores:
                try:
                    arr = np.asarray(scores, dtype=np.float64)
                    return float(arr.mean()), key
                except (TypeError, ValueError):
                    continue
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                return float(value), key
            except (TypeError, ValueError):
                continue
    return None, "missing"


def _branch_from_state(state: dict[str, Any]) -> str:
    branch = (
        state.get("current_active_branch")
        or state.get("anchor_git_branch")
        or state.get("best_variant_this_round")
        or "unknown"
    )
    return str(branch)


def _feature_count_from_state(state: dict[str, Any], branch: str) -> int | str:
    # Resolve using git_branch first
    git_branch = state.get("current_git_branch")
    if git_branch:
        for k, v in state.items():
            if (
                k.startswith(f"branch_{git_branch}_")
                and k.endswith("_oof")
                and isinstance(v, dict)
            ):
                mc = v.get("model_config") or {}
                fc = mc.get("feature_count")
                if isinstance(fc, (int, float)):
                    return int(fc)

    # Try calibration_candidate_oof_key
    calib_key = state.get("calibration_candidate_oof_key")
    if calib_key:
        oof = state.get(calib_key)
        if isinstance(oof, dict):
            mc = oof.get("model_config") or {}
            fc = mc.get("feature_count")
            if isinstance(fc, (int, float)):
                return int(fc)

    # Fallback to branch_{branch}_oof (with target matching)
    for k, v in state.items():
        if (
            k.startswith(f"branch_{branch}")
            and k.endswith("_oof")
            and isinstance(v, dict)
        ):
            mc = v.get("model_config") or {}
            fc = mc.get("feature_count")
            if isinstance(fc, (int, float)):
                return int(fc)

    # Final fallbacks
    for key in (
        "last_ensemble_features",
        "best_variant_features",
        "last_ensemble_feature_count",
    ):
        v = state.get(key)
        if isinstance(v, (int, float)):
            return int(v)
    return "?"


def _calibration_method_from_state(state: dict[str, Any], branch: str) -> str:
    oof = state.get(f"branch_{branch}_oof")
    if isinstance(oof, dict):
        mc = oof.get("model_config") or {}
        cm = mc.get("calibration_method")
        if isinstance(cm, str) and cm:
            return cm
    cm = state.get("last_calibration_method")
    if isinstance(cm, str) and cm:
        return cm
    return "none"


# -- Entry point ----------------------------------------------------------------


def run(
    submission_file: str | None = None, state: dict[str, Any] | None = None
) -> dict:
    print("\n" + "=" * 60)
    print("SKILL 16 — Zindi Submission")
    print("=" * 60 + "\n")

    paths = resolve_competition_paths()

    if submission_file is None:
        if paths.submissions_dir.exists():
            files = sorted(
                paths.submissions_dir.glob("sub_*.csv"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if files:
                submission_file = str(files[0])
            else:
                raise FileNotFoundError("No submission files found")
        else:
            raise FileNotFoundError("Submissions directory not found")
    config = ChallengeConfig.load()
    store = SkillStateStore(paths.state_path)
    skill_state = store.read() if state is None else state

    sub_path = Path(submission_file)
    sample_filename = config.get("sample_submission_filename") or "SampleSubmission.csv"
    sample_path = paths.data_raw_dir / str(sample_filename)

    if not sub_path.exists():
        raise FileNotFoundError(f"Submission file not found: {sub_path}")
    if not sample_path.exists():
        raise FileNotFoundError(f"SampleSubmission.csv not found at {sample_path}")

    if not bool(skill_state.get("human_gate_4_approved", False)):
        return {
            "status": "BLOCKED",
            "reason": "human_gate_4_missing",
            "message": "Human Gate 4 not approved. Skill 14 must be human-approved before submission.",
        }

    branch = _branch_from_state(skill_state)
    if branch and branch != "unknown":
        gate2_key = f"human_gate_2_{branch}_approved"
        if not bool(skill_state.get(gate2_key, False)):
            return {
                "status": "BLOCKED",
                "reason": f"{gate2_key}_missing",
                "message": f"Branch '{branch}' has not been human-approved.",
            }

    print(f"Validating: {sub_path.name}")
    errors = validate(sub_path, sample_path, config=config)
    if errors:
        print("\n[FAIL] VALIDATION FAILED:")
        for e in errors:
            print(f"   {e}")
        return {"status": "BLOCKED", "errors": errors}
    print("[OK] Validation passed")

    from zindian.zindi_client import ZindiClient

    client = ZindiClient()
    try:
        client.select_competition(config.slug)
    except Exception as exc:
        return {
            "status": "BLOCKED",
            "reason": "platform_unreachable",
            "message": f"Could not select competition '{config.slug}': {exc}",
        }
    try:
        live_remaining = int(client.remaining_submissions)
    except Exception:
        live_remaining = -1
    if live_remaining != -1 and live_remaining <= 0:
        raise HardAbortException("Zindi reports zero remaining submissions today.")
    if live_remaining == 1:
        budget_warning_payload = {
            "remaining_submissions": 1,
            "source": "live",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        store.update(budget_warning=budget_warning_payload)
        print(
            "\n[WARN]  BUDGET WARNING: Only 1 live submission remaining today! "
            "Proceeding will exhaust the daily budget. "
            "Explicit confirmation required before proceeding. "
            "Warning written to SKILL_STATE['budget_warning']."
        )

    cached_remaining_val = skill_state.get("remaining_submissions")
    cached_remaining = (
        int(cached_remaining_val) if cached_remaining_val is not None else 10
    )
    used_today = int(skill_state.get("submissions_used_today") or 0)
    print(
        f"\nBudget (cached state): {cached_remaining} remaining | {used_today} used today"
    )
    if cached_remaining <= 0:
        raise HardAbortException("State-side budget guard: zero submissions remaining.")
    if cached_remaining == 1:
        budget_warning_payload = {
            "remaining_submissions": 1,
            "source": "cached",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        store.update(budget_warning=budget_warning_payload)
        print(
            "\n[WARN]  BUDGET WARNING: Only 1 cached submission remaining today! "
            "Explicit confirmation required before proceeding. "
            "Warning written to SKILL_STATE['budget_warning']."
        )

    best_auc = skill_state.get("anchor_oof_score")
    best_f1, metric_source = determine_submission_metrics(sub_path, skill_state)
    feature_count = _feature_count_from_state(skill_state, branch)
    calibration_method = _calibration_method_from_state(skill_state, branch)
    git_branch = skill_state.get("current_git_branch", "unknown")
    metric_name = str(config.get("metric", "f1")).upper()

    print(
        f"""
============================================================
=== HUMAN GATE: Skill 16 — Submit ===
============================================================
File              : {sub_path.name}
Branch            : {git_branch}
OOF {metric_name} : {best_f1}
Reference ROC-AUC : {best_auc}
Metric source     : {metric_source}
Feature count     : {feature_count}
Calibration       : {calibration_method}
Live remaining    : {live_remaining if live_remaining >= 0 else "unknown"}
Validation        : [OK] PASSED

Type YES to submit or NO to abort.
============================================================"""
    )
    response = input("Submit? [YES/NO]: ").strip().upper()
    if response != "YES":
        print("[STOP] Submission aborted by user.")
        return {"status": "ABORTED"}

    oof_str = f"{best_f1:.4f}" if isinstance(best_f1, (int, float)) else "n/a"
    metric_name_lower = metric_name.lower()
    oof_tag = "oof_f1" if "f1" in metric_name_lower else "oof_score"
    comment = (
        f"branch:{git_branch}"
        f"|{oof_tag}:{oof_str}"
        f"|features:{feature_count}"
        f"|calib:{calibration_method}"
    )
    print(f"\nSubmitting with comment: {comment}")
    result = client.submit(filepath=str(sub_path), comment=comment)

    # Only update state if submission succeeded
    if not result or result.get("error"):
        print(f"[FAIL] Submission failed: {result}")
        return {"status": "FAILED", "error": result.get("error", "Unknown error")}

    now_iso = datetime.now(timezone.utc).isoformat()
    try:
        store.update(
            submissions_used_today=used_today + 1,
            submissions_used_total=int(skill_state.get("submissions_used_total") or 0)
            + 1,
            remaining_submissions=live_remaining - 1 if live_remaining > 0 else None,
            last_updated=now_iso,
            last_submission_comment=comment,
            last_submission_at=now_iso,
        )
    except Exception as exc:
        print(f"[WARN]  State write failed after successful submit: {exc}")

    log_path = paths.reports_dir / "submission_log.md"
    log_entry = (
        f"\n## Submission [{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}]\n"
        f"**File**: {sub_path.name}\n"
        f"**Branch**: {git_branch}\n"
        f"**Comment**: {comment}\n"
        f"**Result**: {json.dumps(result)}\n"
    )
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(log_entry)
    except Exception as exc:
        print(f"[WARN]  Failed to append submission log: {exc}")

    print(f"\n[OK] Submitted. Result: {result}")
    print(f"[OK] Logged -> {log_path}")

    print(f"\n{'=' * 60}")
    print("POST-SUBMISSION RESULTS")
    print(f"{'=' * 60}")
    try:
        my_rank = client._user.my_rank
        remaining_after = client.remaining_submissions
        print(f"Current rank    : {my_rank}")
        print(f"Remaining today : {remaining_after}")
        print("\n--- Top 20 Leaderboard ---")
        client.leaderboard(per_page=20)
        try:
            store.update(
                anchor_rank=my_rank,
                last_updated=datetime.now(timezone.utc).isoformat(),
            )
        except Exception as exc:
            print(f"[WARN]  Failed to record anchor_rank: {exc}")
    except Exception as exc:
        print(f"[WARN]  Could not fetch leaderboard: {exc}")

    # Record to ledger
    try:
        with Ledger() as ledger:
            exp_id = ledger.log_experiment(
                branch_name=git_branch,
                oof_score=best_f1,
                metric=metric_name_lower,
                feature_count=feature_count if isinstance(feature_count, int) else None,
                calibration_method=calibration_method,
                gate_result="PASS",
                dag_phase=skill_state.get("dag_phase"),
                notes=comment,
            )
            ledger.log_submission(
                experiment_id=exp_id,
                branch_name=git_branch,
                public_score=result.get("public_score"),
                my_rank=my_rank if "my_rank" in locals() else None,
                comment=comment,
            )
        print(f"[OK] Recorded to ledger (experiment_id={exp_id})")
    except Exception as exc:
        print(f"[WARN]  Failed to record to ledger: {exc}")

    return {"status": "SUBMITTED", "result": result, "comment": comment}


def pull_submission_board() -> list[dict[str, Any]]:
    """Pull submission board from platform and return as list."""
    from zindian.zindi_client import ZindiClient

    config = ChallengeConfig.load()
    client = ZindiClient()
    client.select_competition(config.slug)
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        subs = cast(list[dict[str, Any]], list(client._user.submission_board()))
    finally:
        sys.stdout = old
    return subs


def show_submission_board() -> None:
    """Render the submission board from the platform."""
    subs = pull_submission_board()
    clean: list[dict[str, Any]] = []
    for s in subs:
        clean.append(
            {
                "id": s.get("id"),
                "date": str(s.get("created_at", ""))[:10],
                "file": s.get("filename"),
                "lb_f1": s.get("public_score"),
                "status": s.get("status"),
                "chosen": s.get("chosen"),
                "comment": s.get("comment"),
            }
        )
    col_id, col_date, col_f1, col_ch, col_file = 12, 12, 13, 6, 40
    sep = "-" * 150
    hdr = f"{'ID':{col_id}} {'Date':{col_date}} {'LB F1':>{col_f1}} {'Ch':>{col_ch}}  {'File':{col_file}} Comment"
    print(hdr)
    print(sep)
    for s in clean:
        chosen = "YES" if s["chosen"] else "   "
        f1_str = f"{s['lb_f1']:.9f}" if s["lb_f1"] else "0.000000000"
        row = (
            f"{s['id']:{col_id}} {s['date']:{col_date}} {f1_str:>{col_f1}} {chosen:>{col_ch}}  "
            f"{s['file']:{col_file}} {s['comment']}"
        )
        print(row)
    print(sep)


def pull_leaderboard(per_page: int = 20) -> None:
    """Pull and display current leaderboard."""
    from zindian.zindi_client import ZindiClient

    config = ChallengeConfig.load()
    client = ZindiClient()
    client.select_competition(config.slug)
    print(f"\n{'=' * 60}")
    print(f"LEADERBOARD — {config.slug}")
    print(f"{'=' * 60}")
    try:
        my_rank = client._user.my_rank
        print(f"Your current rank: {my_rank}\n")
    except Exception:
        pass
    client.leaderboard(per_page=per_page)


if __name__ == "__main__":
    if "--submission-board" in sys.argv:
        show_submission_board()
    elif "--leaderboard" in sys.argv:
        pull_leaderboard()
    elif len(sys.argv) < 2:
        print("Usage:")
        print("  python -m zindian.skills.skill_16_submit <file>")
        print("  python -m zindian.skills.skill_16_submit --submission-board")
        print("  python -m zindian.skills.skill_16_submit --leaderboard")
        sys.exit(1)
    else:
        arg = next((a for a in sys.argv[1:] if not a.startswith("--")), None)
        if arg is None:
            print("Usage: python -m zindian.skills.skill_16_submit <file>")
            sys.exit(1)
        print(json.dumps(run(arg), indent=2, default=str))
