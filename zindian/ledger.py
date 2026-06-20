from __future__ import annotations

import duckdb
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .paths import resolve_competition_paths


class Ledger:
    """DuckDB wrapper for experiments and submissions tracking."""

    def __init__(self, path: str | None = None):
        """Initialize DuckDB connection to ledger."""
        if path is None:
            comp_paths = resolve_competition_paths()
            self.path = comp_paths.reports_dir / "experiments.db"
        else:
            self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.path))
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create tables and add any missing columns to existing tables."""
        # Create sequences first
        try:
            self.conn.execute(
                "CREATE SEQUENCE IF NOT EXISTS experiments_id_seq START 1"
            )
        except Exception:
            pass

        try:
            self.conn.execute(
                "CREATE SEQUENCE IF NOT EXISTS submissions_id_seq START 1"
            )
        except Exception:
            pass

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id       INTEGER PRIMARY KEY DEFAULT nextval('experiments_id_seq'),
                branch_name         VARCHAR NOT NULL,
                oof_score           FLOAT,
                metric              VARCHAR,
                oof_rmse            FLOAT,
                feature_count       INTEGER,
                calibration_method  VARCHAR,
                gate_result         VARCHAR,
                gate_reason         VARCHAR,
                md5_target_hash     VARCHAR,
                dag_phase           VARCHAR,
                notes               VARCHAR,
                created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Safe migration: add columns that may be absent in pre-existing databases.
        for col_def in (
            "oof_score FLOAT",
            "metric VARCHAR",
        ):
            col_name = col_def.split()[0]
            try:
                self.conn.execute(
                    f"ALTER TABLE experiments ADD COLUMN IF NOT EXISTS {col_def}"
                )
            except Exception:
                # DuckDB < 0.8 does not support IF NOT EXISTS on ALTER TABLE;
                # fall back to checking the column list manually.
                try:
                    cols = [
                        r[0]
                        for r in self.conn.execute(
                            "PRAGMA table_info('experiments')"
                        ).fetchall()
                    ]
                    if col_name not in cols:
                        self.conn.execute(
                            f"ALTER TABLE experiments ADD COLUMN {col_def}"
                        )
                except Exception:
                    pass  # Column already exists — safe to ignore.

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS submissions (
                submission_id       INTEGER PRIMARY KEY DEFAULT nextval('submissions_id_seq'),
                experiment_id       INTEGER,
                branch_name         VARCHAR NOT NULL,
                submission_rank     INTEGER,
                public_score        FLOAT,
                private_score       FLOAT,
                my_rank             INTEGER,
                selected_for_final  BOOLEAN DEFAULT FALSE,
                selection_rationale VARCHAR,
                comment             VARCHAR,
                submitted_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            )
        """)

        self.conn.commit()

    def log_experiment(
        self,
        *,
        branch_name: str,
        oof_score: Optional[float] = None,
        metric: Optional[str] = None,
        oof_rmse: Optional[float] = None,  # deprecated alias — use oof_score instead
        feature_count: Optional[int] = None,
        calibration_method: Optional[str] = None,
        gate_result: str = "PENDING",
        gate_reason: Optional[str] = None,
        md5_target_hash: Optional[str] = None,
        dag_phase: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> int:
        """
        Log an experiment (branch training run).

        Args:
            oof_score:  Generic primary OOF metric value (F1, AUC, RMSE, etc.).
                        Preferred over the legacy oof_rmse parameter.
            metric:     Name of the metric stored in oof_score (e.g. "f1", "rmse", "auc").
            oof_rmse:   Deprecated. Alias for oof_score when metric is 'rmse'.
                        If oof_score is None and oof_rmse is set, oof_score is populated
                        from oof_rmse and metric is forced to 'rmse'.

        Returns: experiment_id
        """
        # Backward-compat alias resolution
        if oof_score is None and oof_rmse is not None:
            oof_score = oof_rmse
            metric = metric or "rmse"

        cursor = self.conn.execute(
            """
            INSERT INTO experiments (
                branch_name, oof_score, metric, oof_rmse,
                feature_count, calibration_method,
                gate_result, gate_reason, md5_target_hash, dag_phase, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING experiment_id
        """,
            [
                branch_name,
                oof_score,
                metric,
                oof_rmse,  # also write to legacy column for any direct SQL queries
                feature_count,
                calibration_method,
                gate_result,
                gate_reason,
                md5_target_hash,
                dag_phase,
                notes,
            ],
        )

        row = cursor.fetchone()
        if row is None:
            raise RuntimeError("Failed to insert experiment")
        experiment_id = row[0]
        self.conn.commit()
        return experiment_id

    def log_submission(
        self,
        *,
        experiment_id: int,
        branch_name: str,
        submission_rank: Optional[int] = None,
        public_score: Optional[float] = None,
        private_score: Optional[float] = None,
        my_rank: Optional[int] = None,
        selected_for_final: bool = False,
        selection_rationale: Optional[str] = None,
        comment: Optional[str] = None,
    ) -> int:
        """
        Log a submission to Zindi.

        Returns: submission_id
        """
        cursor = self.conn.execute(
            """
            INSERT INTO submissions (
                experiment_id, branch_name, submission_rank, public_score,
                private_score, my_rank, selected_for_final, selection_rationale, comment
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING submission_id
        """,
            [
                experiment_id,
                branch_name,
                submission_rank,
                public_score,
                private_score,
                my_rank,
                selected_for_final,
                selection_rationale,
                comment,
            ],
        )

        row = cursor.fetchone()
        if row is None:
            raise RuntimeError("Failed to insert submission")
        submission_id = row[0]
        self.conn.commit()
        return submission_id

    def get_experiment(self, experiment_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve experiment by ID."""
        cursor = self.conn.execute(
            "SELECT * FROM experiments WHERE experiment_id = ?", [experiment_id]
        )
        row = cursor.fetchone()
        if row is None:
            return None

        cols = [desc[0] for desc in cursor.description]
        return dict(zip(cols, row))

    def get_best_experiment(self) -> Optional[Dict[str, Any]]:
        """Get experiment with best OOF score respecting config metric_direction."""
        _comp_paths = resolve_competition_paths()
        config_path = (
            _comp_paths.competition_dir / "challenge_config.json"
            if _comp_paths.competition_dir is not None
            else _comp_paths.config_path
        )
        try:
            with open(config_path) as f:
                config = json.load(f)
            metric_direction = config.get("metric_direction", "minimize")
        except Exception:
            metric_direction = "minimize"

        # Use oof_score (generic column); fall back to oof_rmse for legacy rows.
        order = "ASC" if metric_direction == "minimize" else "DESC"
        cursor = self.conn.execute(f"""
            SELECT * FROM experiments
            WHERE COALESCE(oof_score, oof_rmse) IS NOT NULL
            ORDER BY COALESCE(oof_score, oof_rmse) {order}
            LIMIT 1
            """)
        row = cursor.fetchone()
        if row is None:
            return None
        cols = [desc[0] for desc in cursor.description]
        return dict(zip(cols, row))

    def get_passed_experiments(self) -> List[Dict[str, Any]]:
        """Get all experiments that passed the gate, ordered by oof_score.

        Direction-aware: minimize metrics ordered ASC, maximize metrics ordered DESC.
        Falls back to oof_rmse for legacy rows that predate the oof_score column.
        """
        from zindian.config import ChallengeConfig

        try:
            cfg = ChallengeConfig.load()
            order = (
                "ASC"
                if cfg.get("metric_direction", "minimize") == "minimize"
                else "DESC"
            )
        except Exception:
            order = "ASC"
        cursor = self.conn.execute(f"""
            SELECT * FROM experiments
            WHERE gate_result = 'PASS'
            ORDER BY COALESCE(oof_score, oof_rmse) {order}
            """)
        cols = [desc[0] for desc in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]

    def get_failed_experiments(self) -> List[Dict[str, Any]]:
        """Get all experiments that failed the gate."""
        cursor = self.conn.execute(
            "SELECT * FROM experiments WHERE gate_result = 'FAIL' ORDER BY created_at DESC"
        )
        cols = [desc[0] for desc in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]

    def get_submissions(self) -> List[Dict[str, Any]]:
        """Get all submissions."""
        cursor = self.conn.execute(
            "SELECT * FROM submissions ORDER BY submitted_at DESC"
        )
        cols = [desc[0] for desc in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]

    def get_selected_submissions(self) -> List[Dict[str, Any]]:
        """Get submissions selected for final private judging."""
        cursor = self.conn.execute(
            "SELECT * FROM submissions WHERE selected_for_final = TRUE ORDER BY public_score DESC"
        )
        cols = [desc[0] for desc in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]

    def update_submission_final_selection(
        self, submission_id: int, selected: bool = True, rationale: Optional[str] = None
    ) -> None:
        """Mark submission as selected (or deselected) for final private judging."""
        self.conn.execute(
            """UPDATE submissions
               SET selected_for_final = ?, selection_rationale = ?
               WHERE submission_id = ?""",
            [selected, rationale, submission_id],
        )
        self.conn.commit()

    def update_gate_result(
        self, experiment_id: int, gate_result: str, gate_reason: Optional[str] = None
    ) -> None:
        """Update gate result for an experiment."""
        self.conn.execute(
            """UPDATE experiments
               SET gate_result = ?, gate_reason = ?
               WHERE experiment_id = ?""",
            [gate_result, gate_reason, experiment_id],
        )
        self.conn.commit()

    def query(
        self, sql: str, params: Optional[List[Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute arbitrary query and return results as list of dicts."""
        cursor = self.conn.execute(sql, params or [])
        cols = [desc[0] for desc in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]

    def close(self) -> None:
        """Close database connection."""
        try:
            self.conn.execute("CHECKPOINT")
        except Exception:
            pass
        self.conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self):
        """Ensure connection is closed on object deletion."""
        try:
            self.close()
        except Exception:
            pass
