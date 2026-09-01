"""Skill 15 - Reporter: Event Logger and Phase Summary Generator.

Phase 1. Logs pipeline events, generates phase summaries, and initialises
session-scoped log files.

Phase 2B / 3B. Generates consolidated Markdown branch-metric summaries from
SKILL_STATE.json via `run_phase_summary(phase)`.

Phase contract (SoT Phase 1):
    skill_01 -> skill_02 -> skill_03 -> skill_04 -> skill_05 -> skill_15

Reads:
    config["task_type"]          -- used for semantic mapping in event data
    state["dag_phase"]           -- current pipeline phase
    state["submissions_used_today"], state["submissions_used_total"]

Writes:
    state["last_reported"]       -- timestamp of last report generation
    reports/summaries/{phase}_summary.json -- per-phase summary files
    reports/summaries/phase_{2b,3b}_summary.md -- Phase 2B / 3B Markdown branch summaries
    reports/sessions/startup_*.jsonl -- session-scoped startup event logs

Does NOT write:
    - Does NOT write to long-term history_log.jsonl during initialisation
    - Startup events are routed to local session-scoped files
"""

from __future__ import annotations
import tabula.skill_state_autopatch  # noqa

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from zindian.config import ChallengeConfig
from zindian.ledger import Ledger
from zindian.paths import resolve_competition_paths
from zindian.state import SkillStateStore

# -- Session-scoped event logging ------------------------------------


def _log_startup_event(
    session_log_path: Path,
    event_data: Dict[str, Any],
) -> None:
    """Write a startup event to the session-scoped log file only.

    This replaces the old pattern of writing to history_log.jsonl during
    initialisation. Session-scoped logs are stored under reports/sessions/
    and never pollute the long-term history log.
    """
    session_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(session_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event_data) + "\n")


# -- Semantic metric mapping -----------------------------------------


def _metric_display_name(task_type: Optional[str]) -> str:
    """Map config.task_type to a human-readable metric label.

    This is the canonical mapping — never use config.domain for metric display.
    """
    mapping = {
        "classification": "Accuracy / LogLoss / AUC",
        "regression": "RMSE / MAE / R²",
        "ranking": "NDCG / MAP",
    }
    return mapping.get(task_type or "", "Unknown metric")


def _task_type_display(task_type: Optional[str]) -> str:
    """Return a human-readable task type label."""
    mapping = {
        "classification": "Classification",
        "regression": "Regression",
        "ranking": "Ranking",
    }
    return mapping.get(task_type or "", "Unknown")


# -- Entry point -----------------------------------------------------


def run(
    *,
    phase: str = "1",
    ledger_path: str | None = None,
    state_path: str | None = None,
    config_path: str | None = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Skill 15 — Reporter: Log pipeline events and generate phase summary.

    Args:
        phase: Pipeline phase identifier ("1", "2a", "2b", "3a", "3b", "4")
        ledger_path: Path to DuckDB experiments.db
        state_path: Path to SKILL_STATE.json
        config_path: Path to challenge_config.json

    Returns:
        Status dict with paths and counts.
    """
    try:
        paths = resolve_competition_paths()
        ledger_path = ledger_path or str(paths.reports_dir / "experiments.db")
        state_path = state_path or str(paths.state_path)
        config_path = config_path or str(paths.config_path)

        phase_clean = str(phase).lower().strip().replace("phase_", "")
        if not phase_clean:
            phase_clean = "1"

        # Load config
        try:
            config = ChallengeConfig.load(config_path)
        except Exception as e:
            return {
                "status": "ERROR",
                "message": f"Failed to load config: {e}",
            }

        # Load state
        state_store = SkillStateStore(Path(state_path))
        state = state_store.read()

        # Extract metric info from config.task_type (NOT config.domain)
        task_type = config.get("task_type")
        metric_label = _metric_display_name(task_type)
        task_label = _task_type_display(task_type)

        # Initialize ledger (creates DB if doesn't exist)
        with Ledger(ledger_path) as ledger:
            # Verify schema by querying
            try:
                experiments = ledger.query("SELECT COUNT(*) as count FROM experiments")
                exp_count = experiments[0]["count"] if experiments else 0
            except Exception:
                exp_count = 0

            try:
                submissions = ledger.query("SELECT COUNT(*) as count FROM submissions")
                sub_count = submissions[0]["count"] if submissions else 0
            except Exception:
                sub_count = 0

        # -- Session-scoped startup logging -------------------------
        # Route startup events to session-scoped files, NOT history_log.jsonl
        session_dir = paths.reports_dir / "sessions"
        session_dir.mkdir(parents=True, exist_ok=True)

        startup_event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "competition_initialized",
            "competition_id": config.slug,
            "task_type": task_type,
            "metric_label": metric_label,
            "metric": config.metric,
            "metric_direction": config.metric_direction,
            "cv_strategy_type": state.get("cv_strategy_type"),
            "cv_strategy_override": state.get("cv_strategy_override", False),
            "dag_phase": state.get("dag_phase"),
            # Fields not available at init stage — set to None
            "anchor_oof_score": None,
            "best_promoted_oof_score": None,
            "best_public_lb_score": None,
            "oof_to_lb_delta": None,
        }

        # Content-hash dedup: check if latest startup log has identical content (excluding timestamp)
        existing_logs = sorted(session_dir.glob("startup_*.jsonl"))
        latest_log = existing_logs[-1] if existing_logs else None
        use_existing = False

        if latest_log is not None:
            try:
                with open(latest_log, "r", encoding="utf-8") as lf:
                    first_line = lf.readline().strip()
                if first_line:
                    latest_event = json.loads(first_line)
                    # Compare ignoring timestamp
                    clean_latest = {
                        k: v for k, v in latest_event.items() if k != "timestamp"
                    }
                    clean_new = {
                        k: v for k, v in startup_event.items() if k != "timestamp"
                    }
                    if clean_latest == clean_new:
                        use_existing = True
                        session_log_path = latest_log
            except Exception:
                pass

        if not use_existing:
            session_start = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            session_log_path = session_dir / f"startup_{session_start}.jsonl"
            _log_startup_event(session_log_path, startup_event)

            # Enforce rolling 14-file window
            all_logs = sorted(session_dir.glob("startup_*.jsonl"))
            if len(all_logs) > 14:
                for old_log in all_logs[:-14]:
                    try:
                        old_log.unlink()
                    except Exception:
                        pass

        print("=" * 60)
        print("SKILL 15 — Central Reporter (Initialization)")
        print("=" * 60)
        print(f"Connecting to ledger: {ledger_path}")
        print(f"Found {exp_count} experiments, {sub_count} submissions.")
        print(f"Session log initialised at: {session_log_path}")

        # -- Generate phase summary report --------------------------
        report = {
            "phase": f"phase_{phase_clean}",
            "competition": config.slug,
            "task_type": task_type,
            "task_type_label": task_label,
            "metric": config.metric,
            "metric_label": metric_label,
            "metric_direction": config.metric_direction,
            "domain": config.domain,
            "daily_limit": config.daily_limit,
            "use_probabilities": config.use_probabilities,
            "automl_permitted": config.automl_permitted,
            "ledger": {
                "path": str(ledger_path),
                "experiments_table_rows": exp_count,
                "submissions_table_rows": sub_count,
            },
            "state": {
                "dag_phase": state.get("dag_phase"),
                "md5_target_hash": state.get("md5_target_hash"),
                "submissions_used_today": state.get("submissions_used_today"),
                "submissions_used_total": state.get("submissions_used_total"),
            },
            "session_log": str(session_log_path),
            "timestamp": state_store.read()["last_updated"],
            "status": "INITIALIZED",
        }

        # Append integrity audit if available
        integrity_path = paths.reports_dir / "integrity_audit.json"
        if integrity_path.exists():
            try:
                integrity_data = json.loads(integrity_path.read_text())
                report["integrity_audit"] = integrity_data
            except Exception:
                pass

        # Write Markdown summary exclusively to summaries/ directory with embedded JSON metadata
        summaries_dir = paths.reports_dir / "summaries"
        summaries_dir.mkdir(parents=True, exist_ok=True)

        # Call run_phase_summary(phase_clean) to generate the Markdown file
        run_phase_summary(phase_clean)

        # Consolidate: append report JSON to the summaries/ Markdown file
        md_filename = f"phase_{phase_clean}_summary.md"
        md_path = summaries_dir / md_filename
        report_path = md_path
        if md_path.exists():
            md_content = md_path.read_text(encoding="utf-8")
            if "## Raw Metadata" in md_content:
                md_content = md_content.split("## Raw Metadata")[0]
            fenced_json = (
                f"\n\n## Raw Metadata\n```json\n{json.dumps(report, indent=2)}\n```\n"
            )
            updated_md = md_content.rstrip() + fenced_json

            md_path.write_text(updated_md, encoding="utf-8")

        # Clean up any legacy standalone JSON summary files in summaries/
        for old_json_name in (f"phase_{phase_clean}_summary.json", f"{phase_clean}_summary.json"):
            old_json_file = summaries_dir / old_json_name
            if old_json_file.exists():
                try:
                    old_json_file.unlink()
                except Exception:
                    pass

        def _rel(p) -> str:
            if not p:
                return ""
            try:
                return str(Path(p).resolve().relative_to(paths.root.resolve()))
            except Exception:
                return str(p)

        # -- Phase transition event (session-scoped) ----------------
        phase_event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "phase_1_summary_generated",
            "competition_id": config.slug,
            "task_type": task_type,
            "metric_label": metric_label,
            "report_path": _rel(report_path),
        }
        _log_startup_event(session_log_path, phase_event)

        return {
            "status": "GO",
            "ledger_path": _rel(ledger_path),
            "experiments_count": exp_count,
            "submissions_count": sub_count,
            "phase_1_summary_path": _rel(report_path),
            "session_log": _rel(session_log_path),
            "message": "Session log initialised, phase summary generated.",
        }

    except Exception as e:
        import traceback

        # Log exception to session-scoped file if paths are available
        try:
            _paths = resolve_competition_paths()
            _session_dir = _paths.reports_dir / "sessions"
            _session_dir.mkdir(parents=True, exist_ok=True)
            _error_log = _session_dir / "skill_15_error.jsonl"
            _log_startup_event(
                _error_log,
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event_type": "skill_15_error",
                    "competition_id": (
                        config.slug if "config" in locals() else "unknown"
                    ),
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
        except Exception:
            pass

        return {
            "status": "ERROR",
            "message": f"Skill 15 failed: {str(e)}",
            "traceback": traceback.format_exc(),
        }


# -- Phase summary --------------------------------


def run_phase_summary(phase: str = "1") -> Dict[str, Any]:
    """
    Generate a consolidated Markdown summary of phase metrics and artifacts.

    Reads SKILL_STATE.json and writes reports/phase_{phase}_summary.md.
    Safe to call multiple times -- overwrites the previous report.

    Args:
        phase: One of "1", "2a", "2b", "3a", "3b", "4" (case-insensitive).

    Returns:
        Status dict with report path and key metric counts.
    """
    import numpy as np

    phase = phase.lower().strip()
    valid_phases = ("1", "2a", "2b", "3a", "3b", "4")
    if phase not in valid_phases:
        return {
            "status": "ERROR",
            "message": f"Unknown phase '{phase}'. Supported phases: {valid_phases}.",
        }

    paths = resolve_competition_paths()
    state_store = SkillStateStore(paths.state_path)
    state = state_store.read()

    try:
        config = ChallengeConfig.load(str(paths.config_path))
        competition = config.slug or "unknown"
        metric_name = str(config.get("metric", "score")).lower()
        metric_direction = str(config.get("metric_direction", "maximize")).lower()
        task_type = str(config.get("task_type", "classification")).lower()
    except Exception:
        competition = "unknown"
        metric_name = "score"
        metric_direction = "maximize"
        task_type = "classification"

    now = datetime.now(timezone.utc).isoformat()
    lines: list[str] = []

    if phase == "1":
        lines += [
            "# Phase 1 Integrity Intake & CV Strategy Summary",
            "",
            f"**Competition:** {competition}  ",
            f"**Metric:** `{metric_name}` ({metric_direction})  ",
            f"**Task Type:** {task_type}  ",
            f"**Generated:** {now}  ",
            "",
            "## Data Integrity & Preflight",
            "",
            f"- **Target Column:** `{config.get('target', 'target')}`",
            f"- **Target Hash (MD5):** `{state.get('md5_target_hash', 'N/A')}`",
            f"- **DAG Phase State:** `{state.get('dag_phase', 'N/A')}`",
            "",
            "## EDA & Dataset Diagnostics",
            "",
        ]
        eda = state.get("eda", {}) or {}
        if eda:
            target_name = config.get("target", "target")
            target_std = eda.get(f"{target_name}_std") or eda.get("target_std") or "N/A"
            lines.append(f"- **Target Std Dev:** `{target_std}`")
            lines.append(f"- **Train Shape:** `{eda.get('train_shape', 'N/A')}`")
            lines.append(f"- **Test Shape:** `{eda.get('test_shape', 'N/A')}`")
            lines.append(
                f"- **Missing Value Cells:** `{eda.get('null_cells_count', 'N/A')}`"
            )
        else:
            lines.append("_No detailed EDA diagnostics in state._")

        lines += [
            "",
            "## Policy & Compliance Constraints",
            "",
            f"- **AutoML Permitted:** `{config.get('automl_permitted', False)}`",
            f"- **Allowed External Data:** `{config.get('allowed_external_data', False)}`",
        ]
        banned = state.get("banned_features", []) or config.get("banned_features", [])
        if banned:
            lines.append(
                f"- **Banned Features:** {', '.join(f'`{b}`' for b in banned)}"
            )
        else:
            lines.append("- **Banned Features:** None declared")

        lines += [
            "",
            "## Cross-Validation Strategy",
            "",
            f"- **Strategy Type:** `{state.get('cv_strategy_type', config.get('cv_strategy', {}).get('type', 'Unknown'))}`",
            f"- **CV Strategy ID:** `{state.get('cv_strategy_id', 'N/A')}`",
        ]
        cv_split_info = state.get("cv_split_summary", {})
        if cv_split_info:
            lines.append(f"- **Folds:** `{cv_split_info.get('n_splits', 5)}`")
            lines.append(
                f"- **Group Column:** `{cv_split_info.get('group_col', 'spatial_cluster')}`"
            )

    elif phase == "2a":
        lines += [
            "# Phase 2A Feature Engineering & Pre-processing Summary",
            "",
            f"**Competition:** {competition}  ",
            f"**Generated:** {now}  ",
            "",
            "## Data Cleaning & Transformation",
            "",
            f"- **Pre-processing Imputation:** `{state.get('preprocessing_strategy', 'train_median_impute')}`",
            f"- **Cleaned Dataset Path:** `{state.get('cleaned_data_path', 'data/processed/')}`",
            "",
            "## Feature Extraction & Policy Verification",
            "",
        ]
        features_added = state.get("engineered_feature_count")
        if features_added is not None:
            lines.append(f"- **Engineered Features Count:** `{features_added}`")
        else:
            lines.append("_Feature extraction summary pending._")

    elif phase == "2b":
        lines += [
            "# Phase 2B Branch Metrics & Candidate Gating Summary",
            "",
            f"**Competition:** {competition}  ",
            f"**Metric:** `{metric_name}` ({metric_direction})  ",
            f"**Task type:** {task_type}  ",
            f"**Generated:** {now}  ",
            "",
        ]

        # Anchor baseline
        anchor = state.get("anchor_oof_score")
        anchor_branch = state.get("anchor_git_branch", "anchor-baseline")
        if anchor is not None:
            lines.append(f"**Anchor baseline ({anchor_branch}):** `{anchor:.6f}`  ")
            lines.append("")

        # Collect all branch OOF records from state
        branch_rows: list[dict] = []
        for key, val in state.items():
            if not (key.startswith("branch_") and key.endswith("_oof")):
                continue
            if not isinstance(val, dict):
                continue
            branch_name = val.get("branch_name") or key.removeprefix(
                "branch_"
            ).removesuffix("_oof")
            model_cfg = val.get("model_config") or {}
            fold_scores_raw = model_cfg.get("fold_scores") or []
            cv_id = val.get("cv_strategy_id", "")
            if fold_scores_raw and len(fold_scores_raw) > 1:
                variance = float(np.var(fold_scores_raw, ddof=1))
                mean_fold = float(np.mean(fold_scores_raw))
            else:
                variance = None
                mean_fold = None
            branch_rows.append(
                {
                    "branch": branch_name,
                    "cv_id": cv_id,
                    "mean_fold": mean_fold,
                    "variance": variance,
                    "fold_scores": fold_scores_raw,
                }
            )

        gate_result = state.get("gate_result", {})
        gate_summary = state.get("gate_summary", "")
        best_branch = state.get("best_variant_branch") or state.get(
            "best_variant_this_round"
        )

        lines.append("## Branch OOF Records")
        lines.append("")
        if not branch_rows:
            lines.append("_No branch OOF records found in SKILL_STATE._")
        else:
            lines.append(
                "| Branch | CV Strategy | Mean Fold Score | Fold Variance (ddof=1) | Fold Scores |"
            )
            lines.append(
                "|--------|-------------|-----------------|----------------------|-------------|"
            )
            for row in branch_rows:
                mean_s = (
                    f"{row['mean_fold']:.6f}" if row["mean_fold"] is not None else "N/A"
                )
                var_s = (
                    f"{row['variance']:.6g}" if row["variance"] is not None else "N/A"
                )
                fs = (
                    ", ".join(f"{s:.4f}" for s in row["fold_scores"])
                    if row["fold_scores"]
                    else "N/A"
                )
                lines.append(
                    f"| `{row['branch']}` | {row['cv_id']} | {mean_s} | {var_s} | {fs} |"
                )
        lines.append("")

        if best_branch:
            lines.append(f"**Promoted branch:** `{best_branch}`  ")
        if gate_summary:
            lines.append(f"**Gate summary:** {gate_summary}  ")
        if isinstance(gate_result, dict):
            gate_pass = gate_result.get("gate", "")
            gate_reason = gate_result.get("reason", "")
            if gate_pass:
                lines.append(f"**Gate result:** `{gate_pass}`  ")
            if gate_reason:
                lines.append(f"**Gate reason:** {gate_reason}  ")
        lines.append("")

        # Carbon footprint section
        lines.append("## Carbon Footprint (Phase 1-2B)")
        lines.append("")
        carbon_total = 0.0
        carbon_rows = []
        for key, val in state.items():
            if key.startswith("telemetry.") and isinstance(val, dict):
                skill = key.removeprefix("telemetry.")
                carbon_kg = val.get("carbon_kg_estimate")
                if carbon_kg:
                    carbon_total += carbon_kg
                    carbon_rows.append(
                        {
                            "skill": skill,
                            "carbon_kg": carbon_kg,
                            "duration_sec": val.get("duration_sec", 0),
                            "method": val.get("tracker_method", "unknown"),
                        }
                    )
        if carbon_rows:
            lines.append(f"**Total carbon footprint:** `{carbon_total:.6f} kg CO₂e`  ")
            lines.append("")
            lines.append("| Skill | Duration (s) | Carbon (kg CO₂e) | Method |")
            lines.append("|-------|--------------|------------------|--------|")
            for row in carbon_rows:
                lines.append(
                    f"| {row['skill']} | {row['duration_sec']:.2f} | {row['carbon_kg']:.6f} | {row['method']} |"
                )
        else:
            lines.append("_No carbon tracking data available._")
        lines.append("")

    elif phase == "3a":
        lines += [
            "# Phase 3A Generalization & Leakage Audit Summary",
            "",
            f"**Competition:** {competition}  ",
            f"**Generated:** {now}  ",
            "",
            "## SHAP Leakage Audit",
            "",
        ]
        shap_top = state.get("shap_top_features", [])
        shap_count = state.get("shap_feature_count")
        shap_skipped = state.get("shap_audit_skipped_reason")

        if shap_skipped:
            lines.append(f"**Audit Skipped:** `{shap_skipped}`  ")
        else:
            if shap_count is not None:
                lines.append(f"**Features Audited:** {shap_count}  ")
            if shap_top:
                lines.append("")
                lines.append("**Top 10 SHAP Features:**")
                for i, feat in enumerate(shap_top[:10], 1):
                    lines.append(f"{i}. `{feat}`")

        lines += [
            "",
            "## Feature Pruning",
            "",
        ]
        pruning_delta = state.get("pruning_delta_f1")
        pruning_pass = state.get("pruning_pass")
        if pruning_delta is not None:
            lines.append(f"**Pruning Performance Delta:** `{pruning_delta:+.6f}`  ")
        if pruning_pass is not None:
            lines.append(
                f"**Pruning Gate Status:** `{'PASS' if pruning_pass else 'PRUNE'}`  "
            )

    elif phase == "3b":
        lines += [
            "# Phase 3B Calibration & Oracle Fusion Summary",
            "",
            f"**Competition:** {competition}  ",
            f"**Generated:** {now}  ",
            "",
            "## OOF Calibration",
            "",
        ]
        cal_method = state.get("calibration_method")
        cal_branch = state.get("calibration_candidate_branch")
        cal_cv_id = state.get("calibration_oof_cv_strategy_id")
        cal_at = state.get("calibration_written_at")

        if cal_method:
            lines.append(f"- **Method:** `{cal_method}`")
        if cal_branch:
            lines.append(f"- **Source Branch:** `{cal_branch}`")
        if cal_cv_id:
            lines.append(f"- **CV Strategy ID:** `{cal_cv_id}`")
        if cal_at:
            lines.append(f"- **Written At:** `{cal_at}`")
        if not cal_method:
            lines.append("_No calibration applied in this round._")

        lines += [
            "",
            "## Ensemble Diversity & Fusion",
            "",
        ]
        fusion = state.get("oracle_fusion_summary", {}) or {}
        if fusion:
            lines.append(
                f"- **Ensemble OOF Score:** `{fusion.get('ensemble_oof_score', 'N/A')}`"
            )
            lines.append(
                f"- **Candidates Fused:** `{fusion.get('candidate_count', 0)}`"
            )
        else:
            lines.append("_Oracle fusion summary pending._")

    elif phase == "4":
        lines += [
            "# Phase 4 Governance & Submission Summary",
            "",
            f"**Competition:** {competition}  ",
            f"**Generated:** {now}  ",
            "",
            "## Submission Integrity",
            "",
            f"- **Submissions Used Today:** `{state.get('submissions_used_today', 0)}`",
            f"- **Submissions Remaining Today:** `{state.get('remaining_submissions', 5)}`",
            f"- **Best Public LB Score:** `{state.get('anchor_lb_score', 'N/A')}`",
            "",
            "## Reproducibility Audit",
            "",
            f"- **Current Git Branch:** `{state.get('current_git_branch', 'N/A')}`",
            f"- **Reproducibility Audit Status:** `{state.get('audit_status', 'PASSED')}`",
        ]

    # Print standardized header for summary run
    print("=" * 60)
    print(f"SKILL 15 — Phase {phase.upper()} Reporter")
    print("=" * 60)

    # Write report exclusively to summaries/
    summaries_dir = paths.reports_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    report_filename = f"phase_{phase}_summary.md"
    report_path = summaries_dir / report_filename
    md_content = "\n".join(lines) + "\n"
    report_path.write_text(md_content, encoding="utf-8")

    # Update SKILL_STATE
    state_store.update(last_reported=now)

    print(f"  [OK] Phase {phase.upper()} summary -> {report_path}")
    return {
        "status": "OK",
        "phase": phase,
        "report_path": str(report_path),
    }


def _write_json_summary(
    phase: str,
    paths: Any,
    state: dict,
    include_keys: list[str],
) -> Dict[str, Any]:
    """Write a lightweight JSON summary for any phase and consolidate into Markdown."""
    phase = phase.lower().strip()

    # Build JSON report
    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "competition": (
            state.get("competition", "unknown")
            if state.get("competition")
            else (
                ChallengeConfig.load().slug if paths.config_path.exists() else "unknown"
            )
        ),
        "dag_phase": state.get("dag_phase"),
    }
    for key in include_keys:
        if key in state:
            report[key] = state[key]

    # Write Markdown summary exclusively to summaries/ with embedded JSON metadata
    summaries_dir = paths.reports_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    md_filename = f"phase_{phase}_summary.md"
    md_path = summaries_dir / md_filename

    # If Markdown file doesn't exist, create it via run_phase_summary
    if not md_path.exists():
        run_phase_summary(phase)

    if md_path.exists():
        md_content = md_path.read_text(encoding="utf-8")
        if "## Raw Metadata" in md_content:
            md_content = md_content.split("## Raw Metadata")[0]

        fenced_json = (
            f"\n\n## Raw Metadata\n```json\n{json.dumps(report, indent=2)}\n```\n"
        )
        updated_md = md_content.rstrip() + fenced_json

        md_path.write_text(updated_md, encoding="utf-8")
        print(f"  [OK] Consolidated {md_filename} with JSON metadata")

    # Clean up any legacy standalone JSON summary files in summaries/
    for old_json_name in (f"phase_{phase}_summary.json", f"{phase}_summary.json"):
        old_json_file = summaries_dir / old_json_name
        if old_json_file.exists():
            try:
                old_json_file.unlink()
            except Exception:
                pass

    return {"status": "OK", "path": str(md_path)}


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
    if result.get("status") != "GO":
        exit(1)
