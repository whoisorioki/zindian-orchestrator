from __future__ import annotations

import duckdb
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .paths import resolve_competition_paths


class Ledger:
    """DuckDB wrapper for experiments and submissions tracking."""
    
    def __init__(self, path: str | None = None):
        """Initialize DuckDB connection to ledger."""
        if path is None:
            self.path = resolve_competition_paths().reports_dir / "experiments.db"
        else:
            self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.path))
        self._ensure_schema()
    
    def _ensure_schema(self) -> None:
        """Create tables if they don't exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id       INTEGER PRIMARY KEY DEFAULT nextval('experiments_id_seq'),
                branch_name         VARCHAR NOT NULL,
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
        
        # Create sequences if they don't exist
        try:
            self.conn.execute("CREATE SEQUENCE experiments_id_seq")
        except:
            pass
        
        try:
            self.conn.execute("CREATE SEQUENCE submissions_id_seq")
        except:
            pass
        
        self.conn.commit()
    
    def log_experiment(
        self,
        *,
        branch_name: str,
        oof_rmse: Optional[float] = None,
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
        
        Returns: experiment_id
        """
        cursor = self.conn.execute("""
            INSERT INTO experiments (
                branch_name, oof_rmse, feature_count, calibration_method,
                gate_result, gate_reason, md5_target_hash, dag_phase, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING experiment_id
        """, [
            branch_name, oof_rmse, feature_count, calibration_method,
            gate_result, gate_reason, md5_target_hash, dag_phase, notes
        ])
        
        experiment_id = cursor.fetchone()[0]
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
        cursor = self.conn.execute("""
            INSERT INTO submissions (
                experiment_id, branch_name, submission_rank, public_score,
                private_score, my_rank, selected_for_final, selection_rationale, comment
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING submission_id
        """, [
            experiment_id, branch_name, submission_rank, public_score,
            private_score, my_rank, selected_for_final, selection_rationale, comment
        ])
        
        submission_id = cursor.fetchone()[0]
        self.conn.commit()
        return submission_id
    
    def get_experiment(self, experiment_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve experiment by ID."""
        cursor = self.conn.execute(
            "SELECT * FROM experiments WHERE experiment_id = ?",
            [experiment_id]
        )
        row = cursor.fetchone()
        if row is None:
            return None
        
        cols = [desc[0] for desc in cursor.description]
        return dict(zip(cols, row))
    
    def get_best_experiment(self) -> Optional[Dict[str, Any]]:
        """Get experiment with lowest OOF RMSE."""
        cursor = self.conn.execute(
            "SELECT * FROM experiments WHERE oof_rmse IS NOT NULL ORDER BY oof_rmse ASC LIMIT 1"
        )
        row = cursor.fetchone()
        if row is None:
            return None
        
        cols = [desc[0] for desc in cursor.description]
        return dict(zip(cols, row))
    
    def get_passed_experiments(self) -> List[Dict[str, Any]]:
        """Get all experiments that passed the gate."""
        cursor = self.conn.execute(
            "SELECT * FROM experiments WHERE gate_result = 'PASS' ORDER BY oof_rmse ASC"
        )
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
        self,
        submission_id: int,
        selected: bool = True,
        rationale: Optional[str] = None
    ) -> None:
        """Mark submission as selected (or deselected) for final private judging."""
        self.conn.execute(
            """UPDATE submissions 
               SET selected_for_final = ?, selection_rationale = ?
               WHERE submission_id = ?""",
            [selected, rationale, submission_id]
        )
        self.conn.commit()
    
    def update_gate_result(
        self,
        experiment_id: int,
        gate_result: str,
        gate_reason: Optional[str] = None
    ) -> None:
        """Update gate result for an experiment."""
        self.conn.execute(
            """UPDATE experiments
               SET gate_result = ?, gate_reason = ?
               WHERE experiment_id = ?""",
            [gate_result, gate_reason, experiment_id]
        )
        self.conn.commit()
    
    def query(self, sql: str, params: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        """Execute arbitrary query and return results as list of dicts."""
        cursor = self.conn.execute(sql, params or [])
        cols = [desc[0] for desc in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]
    
    def close(self) -> None:
        """Close database connection."""
        self.conn.close()
    
    def __del__(self):
        """Ensure connection is closed on object deletion."""
        try:
            self.close()
        except:
            pass
