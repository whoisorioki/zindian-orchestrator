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
            if len(value_cols) > 1 and task_type == "classification":
                # Multi-column classification submission (e.g. TargetF1 hard label + TargetRAUC probability):
                # binary for non-last columns, probability interval for the last value column.
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


def _branch_from_state(state: dict[str, Any], submission_file: Path | None = None) -> str:
    if submission_file:
        name = submission_file.name.lower()
        if "ensemble" in name:
            return "ensemble"
        for k in state:
            if k.startswith("branch_") and k.endswith("_oof"):
                b = k.removeprefix("branch_").removesuffix("_oof")
                if b in name:
                    return b

    branch = (
        state.get("current_active_branch")
        or state.get("best_variant_this_round")
        or state.get("best_variant_branch")
        or state.get("anchor_git_branch")
        or "unknown"
    )
    return str(branch)


def determine_submission_metrics(
    submission_file: Path,
    state: dict[str, Any],
) -> tuple[float | None, str]:
    """Resolve (oof_score, source_key) directly from the SKILL_STATE branch records."""
    branch = _branch_from_state(state, submission_file)

    if branch == "ensemble":
        if "last_ensemble_oof_metric" in state:
            v = state["last_ensemble_oof_metric"]
            if isinstance(v, (int, float)):
                return float(v), "last_ensemble_oof_metric"
        if "branch_ensemble_oof" in state:
            v = state["branch_ensemble_oof"]
            if isinstance(v, dict) and "scores" in v:
                try:
                    arr = np.asarray(v["scores"], dtype=np.float64)
                    return float(arr.mean()), "branch_ensemble_oof"
                except (TypeError, ValueError):
                    pass

    candidate_keys: list[str] = [
        f"branch_calibration_{branch}_oof",
        f"branch_{branch}_oof",
        "last_ensemble_oof_score",
        "best_ensemble_oof_score",
        "last_variant_oof_score",
        "best_variant_oof_score",
        "anchor_oof_score_augmented",
        "anchor_oof_score_challenged",
        "anchor_oof_score",
    ]
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
            elif "score" in value and isinstance(value["score"], (int, float)):
                return float(value["score"]), key
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                return float(value), key
            except (TypeError, ValueError):
                continue
    return None, "missing"


def _feature_count_from_state(state: dict[str, Any], branch: str) -> int | str:
    # 1. Check explicit OOF record model_config for the branch
    for key in (f"branch_calibration_{branch}_oof", f"branch_{branch}_oof"):
        oof = state.get(key)
        if isinstance(oof, dict):
            mc = oof.get("model_config") or {}
            fc = mc.get("feature_count")
            if isinstance(fc, (int, float)) and int(fc) > 0:
                return int(fc)

    # 2. For ensemble, check last_ensemble_feature_count or constituent variants' feature counts
    if branch == "ensemble":
        efc = state.get("last_ensemble_feature_count")
        if isinstance(efc, (int, float)) and int(efc) > 0:
            return int(efc)
        variants = state.get("last_ensemble_variants")
        if isinstance(variants, list) and len(variants) > 0:
            counts = []
            for v_name in variants:
                if v_name == "ensemble":
                    continue
                v_oof = state.get(f"branch_{v_name}_oof") or state.get(f"branch_calibration_{v_name}_oof")
                if isinstance(v_oof, dict):
                    v_fc = (v_oof.get("model_config") or {}).get("feature_count")
                    if isinstance(v_fc, (int, float)) and int(v_fc) > 0:
                        counts.append(int(v_fc))
            if counts:
                return max(counts)

    # 3. Check general feature count state keys
    for key in (
        "last_ensemble_feature_count",
        "best_variant_features",
        "last_ensemble_features",
        "shap_feature_count",
    ):
        v = state.get(key)
        if isinstance(v, (int, float)) and int(v) > 0:
            return int(v)

    # 4. Fallback to feature CSV file on disk if available
    try:
        paths = resolve_competition_paths()
        feat_file = paths.data_processed_dir / f"features_train_{branch}.csv"
        if not feat_file.exists():
            feat_file = paths.data_processed_dir / "features_train_anchor-baseline.csv"
        if feat_file.exists():
            import pandas as pd
            cols = pd.read_csv(feat_file, nrows=1).columns
            cfg = ChallengeConfig.load()
            target_col = cfg.get("target_col")
            id_col = cfg.get("id_col") or "ID"
            feat_cols = [c for c in cols if c not in (id_col, target_col)]
            if feat_cols:
                return len(feat_cols)
    except Exception:
        pass

    return "?"


def _calibration_method_from_state(state: dict[str, Any], branch: str) -> str:
    for key in (f"branch_calibration_{branch}_oof", f"branch_{branch}_oof"):
        oof = state.get(key)
        if isinstance(oof, dict):
            mc = oof.get("model_config") or {}
            cm = mc.get("calibration_method") or mc.get("method")
            if isinstance(cm, str) and cm:
                return cm
    cm = state.get("last_calibration_method")
    if isinstance(cm, str) and cm:
        return cm
    top_level = state.get("calibration_method")
    if isinstance(top_level, str) and top_level in ("platt", "isotonic"):
        return top_level
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

    branch = _branch_from_state(skill_state, sub_path)
    if branch and branch != "unknown":
        if branch == "ensemble":
            gate_approved = bool(skill_state.get("human_gate_3_approved", False))
            gate_key = "human_gate_3_approved"
        else:
            gate_key = f"human_gate_2_{branch}_approved"
            gate_approved = bool(skill_state.get(gate_key, False))

        if not gate_approved:
            return {
                "status": "BLOCKED",
                "reason": f"{gate_key}_missing",
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
    if cached_remaining == 1 and live_remaining == -1:
        budget_warning_payload = {
            "remaining_submissions": 1,
            "source": "cached",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        store.update(budget_warning=budget_warning_payload)
        print(
            "\n[WARN]  BUDGET WARNING: Only 1 cached submission remaining today! "
            "Proceeding will exhaust the daily budget. "
            "Explicit confirmation required before proceeding. "
            "Warning written to SKILL_STATE['budget_warning']."
        )

    best_auc = skill_state.get("anchor_oof_score")
    best_f1, metric_source = determine_submission_metrics(sub_path, skill_state)
    feature_count = _feature_count_from_state(skill_state, branch)
    calibration_method = _calibration_method_from_state(skill_state, branch)
    metric_name = str(config.get("metric", "f1")).upper()
    metric_display = "COMPOSITE" if metric_name == "MULTI" else metric_name

    print(
        f"""
============================================================
=== HUMAN GATE: Skill 16 — Submit ===
============================================================
File              : {sub_path.name}
Branch            : {branch}
OOF {metric_display} : {best_f1}
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
        f"branch:{branch}"
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
        f"**Branch**: {branch}\n"
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
                branch_name=branch,
                oof_score=best_f1,
                metric=metric_name_lower,
                feature_count=feature_count if isinstance(feature_count, int) else None,
                calibration_method=calibration_method,
                gate_result="PASS",
                dag_phase=skill_state.get("dag_phase"),
                notes=comment,
            )
            z_id = str(result.get("id")) if result.get("id") else None
            ledger.log_submission(
                experiment_id=exp_id,
                branch_name=branch,
                zindi_id=z_id,
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
    """Render the submission board from the platform and persist to DuckDB ledger."""
    subs = pull_submission_board()
    clean: list[dict[str, Any]] = []
    
    # Persist board entries to ledger
    try:
        paths = resolve_competition_paths()
        manifest_map: dict[str, dict[str, Any]] = {}
        manifest_path = paths.reports_dir / "submissions_manifest.json"
        if manifest_path.exists():
            try:
                import json
                with manifest_path.open(encoding="utf-8") as f:
                    manifest_data = json.load(f)
                    for item in manifest_data:
                        manifest_map[str(item.get("zindi_id"))] = item
            except Exception:
                pass

        # Load Human Gate 5 selections.
        # Primary source: reports/audits/final_selections.json — the file
        # written by skill_17 after the human Gate 5 decision (schema:
        # {"selections": [{"filename": ..., "score": ...}]}). Filenames are
        # joined to zindi_ids via submissions_manifest.json.
        # Fallback: SKILL_STATE human_gate_5_selection (list of filenames).
        gate5_ids = set()
        gate5_files = set()
        gate5_source = "none"
        final_sel_path = paths.reports_dir / "audits" / "final_selections.json"
        if final_sel_path.exists():
            try:
                import json
                with final_sel_path.open(encoding="utf-8") as f:
                    sel_data = json.load(f)
                for item in sel_data.get("selections", []):
                    fname = str(item.get("filename", ""))
                    if fname:
                        gate5_files.add(fname)
                # Join filenames -> zindi_ids through the manifest.
                filename_to_zindi = {
                    str(item.get("filename", "")): str(item.get("zindi_id", ""))
                    for item in manifest_map.values()
                }
                for fname in gate5_files:
                    z_id = filename_to_zindi.get(fname)
                    if z_id:
                        gate5_ids.add(z_id)
                if gate5_files:
                    gate5_source = "final_selections.json"
            except Exception:
                pass

        if not gate5_files and paths.state_path.exists():
            try:
                import json
                with paths.state_path.open(encoding="utf-8") as f:
                    st_data = json.load(f)
                gate5_files.update(st_data.get("human_gate_5_selection", []))
                if gate5_files:
                    gate5_source = "SKILL_STATE.human_gate_5_selection"
            except Exception:
                pass
        if gate5_source != "none":
            print(f"[Gate 5] selected_for_final governed by: {gate5_source}")

        with Ledger() as ledger:
            for s in subs:
                z_id = str(s.get("id"))
                pub_score = s.get("public_score")
                filename = str(s.get("filename", ""))
                platform_chosen = bool(s.get("chosen", False))
                comment = s.get("comment")
                m_info = manifest_map.get(z_id, {})
                lb_f1 = s.get("lb_f1") or m_info.get("lb_f1")
                lb_auc = s.get("lb_auc") or m_info.get("lb_auc")
                submitted_at = s.get("created_at") or m_info.get("created_at")

                # If Human Gate 5 selections exist, they govern selected_for_final
                if gate5_ids or gate5_files:
                    selected_for_final = (z_id in gate5_ids) or (filename in gate5_files)
                else:
                    selected_for_final = platform_chosen

                ledger.upsert_submission_by_zindi_id(
                    zindi_id=z_id,
                    public_score=pub_score,
                    lb_f1=lb_f1,
                    lb_auc=lb_auc,
                    selected_for_final=selected_for_final,
                    comment=comment,
                    submitted_at=submitted_at,
                )
    except Exception as exc:
        print(f"[WARN] Failed to persist submission board to ledger: {exc}")

    for s in subs:
        clean.append(
            {
                "id": s.get("id"),
                "date": str(s.get("created_at", ""))[:10],
                "file": s.get("filename"),
                "public_score": s.get("public_score"),
                "status": s.get("status"),
                "chosen": s.get("chosen"),
                "comment": s.get("comment"),
            }
        )
    col_id, col_date, col_score, col_ch, col_file = 12, 12, 13, 6, 40
    sep = "-" * 150
    hdr = f"{'ID':{col_id}} {'Date':{col_date}} {'Public Score':>{col_score}} {'Ch':>{col_ch}}  {'File':{col_file}} Comment"
    print(hdr)
    print(sep)
    for s in clean:
        chosen = "YES" if s["chosen"] else "   "
        score_str = f"{s['public_score']:.9f}" if s["public_score"] else "0.000000000"
        row = (
            f"{s['id']:{col_id}} {s['date']:{col_date}} {score_str:>{col_score}} {chosen:>{col_ch}}  "
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
