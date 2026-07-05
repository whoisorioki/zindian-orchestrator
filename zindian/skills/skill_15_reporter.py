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
    reports/{phase}_summary.json -- per-phase summary files
    reports/phase_{2b,3b}_summary.md -- Phase 2B / 3B Markdown branch summaries

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
    ledger_path: str | None = None,
    state_path: str | None = None,
    config_path: str | None = None,
) -> Dict[str, Any]:
    """
    Skill 15 — Reporter: Log pipeline events and generate phase summary.

    Args:
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
        session_start = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        session_log_path = session_dir / f"startup_{session_start}.jsonl"

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
        _log_startup_event(session_log_path, startup_event)

        # -- Generate phase summary report --------------------------
        report_path = paths.reports_dir / "phase_1_summary.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "phase": "phase_1_integrity_intake",
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

        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )

        # Append integrity audit if available
        integrity_path = paths.reports_dir / "integrity_audit.json"
        if integrity_path.exists():
            try:
                integrity_data = json.loads(integrity_path.read_text())
                report["integrity_audit"] = integrity_data
                report_path.write_text(
                    json.dumps(report, indent=2, sort_keys=False) + "\n",
                    encoding="utf-8",
                )
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


# -- Phase 2B / 3B Markdown summary --------------------------------


def run_phase_summary(phase: str = "2b") -> Dict[str, Any]:
    """
    Generate a consolidated Markdown summary of branch metrics for Phase 2B or 3B.

    Reads SKILL_STATE.json and writes reports/phase_{phase}_summary.md.
    Safe to call multiple times -- overwrites the previous report.

    Args:
        phase: One of "2b" or "3b" (case-insensitive).

    Returns:
        Status dict with report path and key metric counts.
    """
    import numpy as np

    phase = phase.lower().strip()
    if phase not in ("2b", "3b"):
        return {
            "status": "ERROR",
            "message": f"Unknown phase '{phase}'. Use '2b' or '3b'.",
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

    if phase == "2b":
        lines += [
            "# Phase 2B Branch Metrics Summary",
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
            # scores field holds OOF predictions, not fold scores -- check model_config
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

        # Gate results
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

    elif phase == "3b":
        lines += [
            "# Phase 3B SHAP + Calibration Summary",
            "",
            f"**Competition:** {competition}  ",
            f"**Generated:** {now}  ",
            "",
        ]

        # SHAP section
        shap_top = state.get("shap_top_features", [])
        shap_count = state.get("shap_feature_count")
        pruning_delta = state.get("pruning_delta_f1")
        pruning_pass = state.get("pruning_pass")
        shap_skipped = state.get("shap_audit_skipped_reason")

        lines.append("## SHAP Audit")
        lines.append("")
        if shap_skipped:
            lines.append(f"**Skipped:** `{shap_skipped}`  ")
        else:
            if shap_count is not None:
                lines.append(f"**Features audited:** {shap_count}  ")
            if pruning_delta is not None:
                lines.append(f"**Pruning delta:** `{pruning_delta:+.6f}`  ")
            if pruning_pass is not None:
                lines.append(
                    f"**Pruning gate:** `{'PASS' if pruning_pass else 'PRUNE'}`  "
                )
            if shap_top:
                lines.append("")
                lines.append("**Top SHAP features:**")
                for i, feat in enumerate(shap_top[:10], 1):
                    lines.append(f"{i}. `{feat}`")
        lines.append("")

        # Calibration section
        cal_method = state.get("calibration_method")
        cal_branch = state.get("calibration_candidate_branch")
        cal_cv_id = state.get("calibration_oof_cv_strategy_id")
        cal_at = state.get("calibration_written_at")

        lines.append("## Calibration")
        lines.append("")
        if cal_method:
            lines.append(f"**Method:** `{cal_method}`  ")
        if cal_branch:
            lines.append(f"**Source branch:** `{cal_branch}`  ")
        if cal_cv_id:
            lines.append(f"**CV strategy ID:** `{cal_cv_id}`  ")
        if cal_at:
            lines.append(f"**Written at:** {cal_at}  ")
        lines.append("")

        # Carbon footprint section (full pipeline)
        lines.append("## Carbon Footprint (Full Pipeline)")
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
                            "peak_memory_mb": val.get("peak_memory_mb", 0),
                            "method": val.get("tracker_method", "unknown"),
                            "hardware": val.get("hardware_type", "unknown"),
                            "region": val.get("region", "unknown"),
                        }
                    )
        if carbon_rows:
            lines.append(f"**Total carbon footprint:** `{carbon_total:.6f} kg CO₂e`  ")
            lines.append(
                f"**Tracking method:** `{carbon_rows[0]['method'] if carbon_rows else 'unknown'}`  "
            )
            lines.append(
                f"**Hardware:** `{carbon_rows[0]['hardware'] if carbon_rows else 'unknown'}`  "
            )
            lines.append(
                f"**Region:** `{carbon_rows[0]['region'] if carbon_rows else 'unknown'}`  "
            )
            lines.append("")
            lines.append("| Skill | Duration (s) | Peak RAM (MB) | Carbon (kg CO₂e) |")
            lines.append("|-------|--------------|---------------|------------------|")
            for row in carbon_rows:
                lines.append(
                    f"| {row['skill']} | {row['duration_sec']:.2f} | {row['peak_memory_mb']:.2f} | {row['carbon_kg']:.6f} |"
                )
        else:
            lines.append("_No carbon tracking data available._")
        lines.append("")

    # Write report
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    report_filename = f"phase_{phase}_summary.md"
    report_path = paths.reports_dir / report_filename
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Update SKILL_STATE
    state_store.update(last_reported=now)

    print(f"[OK] Phase {phase.upper()} summary -> {report_path}")
    return {
        "status": "OK",
        "phase": phase,
        "report_path": str(report_path),
        "branch_count": len(branch_rows) if phase == "2b" else None,
    }


def _write_json_summary(
    phase: str,
    paths: Any,
    state: dict,
    include_keys: list[str],
) -> Dict[str, Any]:
    """Write a lightweight JSON summary for any phase."""
    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "competition": state.get("competition", "unknown"),
        "dag_phase": state.get("dag_phase"),
    }
    for key in include_keys:
        if key in state:
            report[key] = state[key]

    report_path = paths.reports_dir / f"{phase}_summary.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return {"status": "OK", "path": str(report_path)}


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
    if result.get("status") != "GO":
        exit(1)
