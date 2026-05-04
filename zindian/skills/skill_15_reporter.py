"""Skill 15 — Reporter: Initialize DuckDB Ledger and Generate Reports"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from zindian.config import ChallengeConfig
from zindian.ledger import Ledger
from zindian.paths import resolve_competition_paths
from zindian.state import SkillStateStore


def run(
    *,
    ledger_path: str | None = None,
    state_path: str | None = None,
    config_path: str | None = None,
) -> Dict[str, Any]:
    """
    Skill 15 — Reporter: Initialize DuckDB ledger and generate phase summary.
    
    Args:
        ledger_path: Path to DuckDB experiments.db
        state_path: Path to SKILL_STATE.json
        config_path: Path to challenge_config.json
    
    Returns:
        Status dict: {
            "status": "GO|ERROR",
            "ledger_path": "...",
            "experiments_count": N,
            "submissions_count": N,
            "message": "..."
        }
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
        
        # Initialize ledger (creates DB if doesn't exist)
        ledger = Ledger(ledger_path)
        
        # Verify schema by querying
        try:
            experiments = ledger.query("SELECT COUNT(*) as count FROM experiments")
            exp_count = experiments[0]["count"] if experiments else 0
        except Exception as e:
            exp_count = 0
        
        try:
            submissions = ledger.query("SELECT COUNT(*) as count FROM submissions")
            sub_count = submissions[0]["count"] if submissions else 0
        except Exception as e:
            sub_count = 0
        
        ledger.close()
        
        # Generate Phase 1 summary report
        report_path = paths.reports_dir / "phase_1_summary.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            "phase": "phase_1_integrity_intake",
            "competition": config.slug,
            "metric": config.metric,
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
            "timestamp": state_store.read()["last_updated"],
            "status": "INITIALIZED",
        }
        
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=False) + "\n",
            encoding="utf-8"
        )
        
        # Log integrity audit summary if available
        integrity_path = paths.reports_dir / "integrity_audit.json"
        if integrity_path.exists():
            try:
                integrity_data = json.loads(integrity_path.read_text())
                report["integrity_audit"] = integrity_data
                report_path.write_text(
                    json.dumps(report, indent=2, sort_keys=False) + "\n",
                    encoding="utf-8"
                )
            except:
                pass
        
        return {
            "status": "GO",
            "ledger_path": str(ledger_path),
            "experiments_count": exp_count,
            "submissions_count": sub_count,
            "phase_1_summary_path": str(report_path),
            "message": "DuckDB ledger initialized and Phase 1 summary generated",
        }
    
    except Exception as e:
        import traceback
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
