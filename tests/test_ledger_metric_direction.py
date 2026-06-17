"""Test ledger.get_best_experiment() respects metric_direction from config."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from zindian.ledger import Ledger


def test_get_best_experiment_minimize():
    """Test get_best_experiment with minimize metric (RMSE)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        config_path = Path(tmpdir) / "challenge_config.json"

        # Create config with minimize metric
        config = {"metric": "rmse", "metric_direction": "minimize"}
        config_path.write_text(json.dumps(config))

        # Mock resolve_competition_paths to return our temp dir
        with patch("zindian.ledger.resolve_competition_paths") as mock_paths:
            mock_paths.return_value.reports_dir = Path(tmpdir)
            mock_paths.return_value.competition_dir = Path(tmpdir)

            ledger = Ledger(str(db_path))

            # Log experiments with different RMSE scores
            ledger.log_experiment(
                branch_name="high_rmse", oof_rmse=0.8, gate_result="PASS"
            )
            ledger.log_experiment(
                branch_name="low_rmse", oof_rmse=0.5, gate_result="PASS"
            )
            ledger.log_experiment(
                branch_name="mid_rmse", oof_rmse=0.6, gate_result="PASS"
            )

            # Best should be lowest RMSE
            best = ledger.get_best_experiment()
            assert best is not None
            assert best["branch_name"] == "low_rmse"
            assert best["oof_rmse"] == 0.5


def test_get_best_experiment_maximize():
    """Test get_best_experiment with maximize metric (F1)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        config_path = Path(tmpdir) / "challenge_config.json"

        # Create config with maximize metric
        config = {"metric": "f1_score", "metric_direction": "maximize"}
        config_path.write_text(json.dumps(config))

        # Mock resolve_competition_paths to return our temp dir
        with patch("zindian.ledger.resolve_competition_paths") as mock_paths:
            mock_paths.return_value.reports_dir = Path(tmpdir)
            mock_paths.return_value.competition_dir = Path(tmpdir)

            ledger = Ledger(str(db_path))

            # Log experiments with different F1 scores (stored in oof_rmse field)
            ledger.log_experiment(
                branch_name="low_f1", oof_rmse=0.5, gate_result="PASS"
            )
            ledger.log_experiment(
                branch_name="high_f1", oof_rmse=0.9, gate_result="PASS"
            )
            ledger.log_experiment(
                branch_name="mid_f1", oof_rmse=0.7, gate_result="PASS"
            )

            # Best should be highest F1
            best = ledger.get_best_experiment()
            assert best is not None
            assert best["branch_name"] == "high_f1"
            assert best["oof_rmse"] == 0.9


def test_get_best_experiment_default_minimize():
    """Test get_best_experiment defaults to minimize when metric_direction missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        config_path = Path(tmpdir) / "challenge_config.json"

        # Create config without metric_direction
        config = {"metric": "rmse"}
        config_path.write_text(json.dumps(config))

        with patch("zindian.ledger.resolve_competition_paths") as mock_paths:
            mock_paths.return_value.reports_dir = Path(tmpdir)
            mock_paths.return_value.competition_dir = Path(tmpdir)

            ledger = Ledger(str(db_path))

            ledger.log_experiment(branch_name="exp1", oof_rmse=0.8, gate_result="PASS")
            ledger.log_experiment(branch_name="exp2", oof_rmse=0.5, gate_result="PASS")

            # Should default to minimize
            best = ledger.get_best_experiment()
            assert best is not None
            assert best["branch_name"] == "exp2"
            assert best["oof_rmse"] == 0.5
