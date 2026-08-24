import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from zindian.orchestrator import run_skill, SKILL_REGISTRY


def test_externalized_logging_captures_stdout_and_stderr():
    """Test that run_skill redirects stdout/stderr to a local file in the logs/ directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Mock the path resolver to return our temp directory as competition_dir
        with patch("zindian.paths.resolve_competition_paths") as mock_paths:
            mock_paths_obj = MagicMock()
            mock_paths_obj.competition_dir = tmp_path
            mock_paths.return_value = mock_paths_obj

            # Create a mock skill module
            import types

            dummy_mod = types.ModuleType("dummy_skill_test")

            def dummy_run(**kwargs):
                print("Stdout log message")
                sys.stderr.write("Stderr error message\n")
                return {"status": "SUCCESS"}

            dummy_mod.run = dummy_run  # type: ignore[attr-defined]

            # Inject the dummy skill into SKILL_REGISTRY
            SKILL_REGISTRY["dummy_skill_test"] = (
                "Dummy skill for logging test",
                dummy_mod,
            )

            try:
                # Execute the skill
                result = run_skill("dummy_skill_test")
                assert result.get("status") == "SUCCESS"

                # Verify log file is created
                log_file = tmp_path / "logs" / "dummy_skill_test.log"
                assert log_file.exists()

                # Verify log file content
                log_content = log_file.read_text(encoding="utf-8")
                assert "Stdout log message" in log_content
                assert "Stderr error message" in log_content

            finally:
                # Cleanup registry
                SKILL_REGISTRY.pop("dummy_skill_test", None)
