"""Skill 15 — Reporter: Event Logger and Phase Summary Generator.

Phase 1. Logs pipeline events, generates phase summaries, and initialises
session-scoped log files.

Phase contract (SoT §Phase 1):
    skill_01 → skill_02 → skill_03 → skill_04 → skill_05 → skill_15

Reads:
    config["task_type"]          — used for semantic mapping in event data
    state["dag_phase"]           — current pipeline phase
    state["submissions_used_today"], state["submissions_used_total"]

Writes:
    state["last_reported"]       — timestamp of last report generation
    reports/{phase}_summary.json — per-phase summary files

Does NOT write:
    - Does NOT write to long-term history_log.jsonl during initialisation
    - Startup events are routed to local session-scoped files
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from zindian.config import ChallengeConfig
from zindian.ledger import Ledger
from zindian.paths import resolve_competition_paths
from zindian.state import SkillStateStore

# ── Session-scoped event logging ────────────────────────────────────


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


# ── Semantic metric mapping ─────────────────────────────────────────


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


# ── Entry point ─────────────────────────────────────────────────────


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
        ledger = Ledger(ledger_path)

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

        ledger.close()

        # ── Session-scoped startup logging ─────────────────────────
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

        # ── Generate phase summary report ──────────────────────────
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

        # ── Phase transition event (session-scoped) ────────────────
        phase_event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "phase_1_summary_generated",
            "competition_id": config.slug,
            "task_type": task_type,
            "metric_label": metric_label,
            "report_path": str(report_path),
        }
        _log_startup_event(session_log_path, phase_event)

        return {
            "status": "GO",
            "ledger_path": str(ledger_path),
            "experiments_count": exp_count,
            "submissions_count": sub_count,
            "phase_1_summary_path": str(report_path),
            "session_log": str(session_log_path),
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


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
    if result.get("status") != "GO":
        exit(1)
