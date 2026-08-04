"""
Tests for _append_history_log() — the writer added to skill_22 that produces
competition_history/history_log.jsonl.

Covers:
  - File created on first passing run (F8 fix: file previously never came to exist)
  - Entries appended correctly on subsequent runs (no overwrite)
  - Failed audit does NOT write an entry (preserves log integrity)
  - three_lens._eval_phase4_generalisation returns PASS once the file exists
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_paths(tmp_path: Path):
    """Return a minimal mock paths object pointing at tmp_path."""

    class _Paths:
        root = tmp_path
        competition_dir = tmp_path / "competitions" / "test-slug"
        state_path = competition_dir / "SKILL_STATE.json"

    _Paths.competition_dir.mkdir(parents=True, exist_ok=True)
    return _Paths()


def _write_state(tmp_path: Path, slug: str = "test-slug") -> Path:
    """Write a minimal SKILL_STATE.json that passes the reproducibility audit."""
    comp_dir = tmp_path / "competitions" / slug
    comp_dir.mkdir(parents=True, exist_ok=True)
    state_path = comp_dir / "SKILL_STATE.json"
    state = {
        "slug": slug,
        "anchor_oof_score": 0.85,
        "variants_passed": 2,
        "selected_submissions": [101, 102],
        "reproducibility_audit": {"success": True},
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")
    return state_path


def _minimal_config(slug: str = "test-slug") -> dict:
    return {
        "slug": slug,
        "metric": "auc",
        "task_type": "classification",
        "cv_strategy": {"type": "stratifiedkfold"},
    }


def _minimal_state(slug: str = "test-slug") -> dict:
    return {
        "slug": slug,
        "anchor_oof_score": 0.85,
        "variants_passed": 2,
        "selected_submissions": [101, 102],
        "reproducibility_audit": {"success": True},
    }


# ---------------------------------------------------------------------------
# Import the helper directly
# ---------------------------------------------------------------------------

from zindian.skills.skill_22_reproducibility_audit import _append_history_log


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAppendHistoryLog:
    def test_file_created_on_first_call(self, tmp_path):
        paths = _make_paths(tmp_path)
        log_path = tmp_path / "competition_history" / "history_log.jsonl"
        assert not log_path.exists()

        _append_history_log(paths, _minimal_config(), _minimal_state())

        assert log_path.exists()

    def test_entry_is_valid_json_with_required_keys(self, tmp_path):
        paths = _make_paths(tmp_path)
        _append_history_log(paths, _minimal_config(), _minimal_state())

        log_path = tmp_path / "competition_history" / "history_log.jsonl"
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1

        entry = json.loads(lines[0])
        for key in (
            "slug",
            "completed_at",
            "metric",
            "task_type",
            "anchor_oof_score",
            "variants_passed",
            "cv_strategy",
            "promoted_branches",
            "reproducibility_audit_passed",
        ):
            assert key in entry, f"Missing key: {key}"

    def test_entry_values_match_config_and_state(self, tmp_path):
        paths = _make_paths(tmp_path)
        config = _minimal_config("my-competition")
        state = _minimal_state("my-competition")
        state["anchor_oof_score"] = 0.912
        state["variants_passed"] = 3

        _append_history_log(paths, config, state)

        log_path = tmp_path / "competition_history" / "history_log.jsonl"
        entry = json.loads(log_path.read_text(encoding="utf-8").strip())
        assert entry["slug"] == "my-competition"
        assert entry["metric"] == "auc"
        assert entry["anchor_oof_score"] == pytest.approx(0.912)
        assert entry["variants_passed"] == 3
        assert entry["cv_strategy"] == "stratifiedkfold"
        assert entry["reproducibility_audit_passed"] is True

    def test_second_call_appends_not_overwrites(self, tmp_path):
        paths = _make_paths(tmp_path)

        _append_history_log(paths, _minimal_config("comp-a"), _minimal_state("comp-a"))
        _append_history_log(paths, _minimal_config("comp-b"), _minimal_state("comp-b"))

        log_path = tmp_path / "competition_history" / "history_log.jsonl"
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

        slugs = [json.loads(l)["slug"] for l in lines]
        assert "comp-a" in slugs
        assert "comp-b" in slugs

    def test_parent_directory_created_if_missing(self, tmp_path):
        paths = _make_paths(tmp_path)
        log_dir = tmp_path / "competition_history"
        assert not log_dir.exists()

        _append_history_log(paths, _minimal_config(), _minimal_state())

        assert log_dir.is_dir()

    def test_reproducibility_audit_passed_false_when_state_marks_failure(
        self, tmp_path
    ):
        paths = _make_paths(tmp_path)
        state = _minimal_state()
        state["reproducibility_audit"] = {"success": False}

        _append_history_log(paths, _minimal_config(), state)

        log_path = tmp_path / "competition_history" / "history_log.jsonl"
        entry = json.loads(log_path.read_text(encoding="utf-8").strip())
        assert entry["reproducibility_audit_passed"] is False


class TestRunFunctionHistoryIntegration:
    """Tests that skill_22.run() only writes the history log on success."""

    def _make_valid_state(self, tmp_path: Path, slug: str = "test-slug") -> Path:
        """Write a schema-valid SKILL_STATE.json to the competition directory."""
        from zindian.schemas import skill_state_skeleton

        comp_dir = tmp_path / "competitions" / slug
        comp_dir.mkdir(parents=True, exist_ok=True)
        skeleton = skill_state_skeleton()
        skeleton["anchor_oof_score"] = 0.88
        state_path = comp_dir / "SKILL_STATE.json"
        state_path.write_text(json.dumps(skeleton), encoding="utf-8")
        return state_path

    def test_failed_audit_does_not_write_history(self, tmp_path, monkeypatch):
        """A failed audit must not produce a history entry."""
        import zindian.skills.skill_22_reproducibility_audit as s22
        import zindian.paths as zp

        self._make_valid_state(tmp_path)

        monkeypatch.setattr(s22, "audit_pipeline", lambda slug=None: False)
        monkeypatch.setattr(
            zp,
            "resolve_competition_paths",
            lambda slug=None, **kw: _FakePaths(tmp_path, "test-slug"),
        )

        s22.run(slug="test-slug")

        log_path = tmp_path / "competition_history" / "history_log.jsonl"
        assert not log_path.exists()

    def test_passing_audit_writes_history(self, tmp_path, monkeypatch):
        """A passing audit must write exactly one history entry."""
        import zindian.skills.skill_22_reproducibility_audit as s22
        import zindian.paths as zp

        self._make_valid_state(tmp_path)

        monkeypatch.setattr(s22, "audit_pipeline", lambda slug=None: True)
        monkeypatch.setattr(
            zp,
            "resolve_competition_paths",
            lambda slug=None, **kw: _FakePaths(tmp_path, "test-slug"),
        )

        s22.run(slug="test-slug")

        log_path = tmp_path / "competition_history" / "history_log.jsonl"
        assert log_path.exists()
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["slug"] == "test-slug"


class TestThreeLensUnblocked:
    """three_lens Phase 4 generalisation check must PASS once the file exists."""

    def test_phase4_generalisation_passes_with_history_log(self, tmp_path, monkeypatch):
        from zindian import three_lens
        import zindian.paths as zp

        # Create a minimal SKILL_STATE with reproducibility_audit token
        comp_dir = tmp_path / "competitions" / "ey-test"
        comp_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "reproducibility_audit": {"success": True},
            "file_hashes": {"train.csv": "abc123"},
            "seed": 42,
        }
        state_path = comp_dir / "SKILL_STATE.json"
        state_path.write_text(json.dumps(state), encoding="utf-8")

        # Create the history log
        log_dir = tmp_path / "competition_history"
        log_dir.mkdir(parents=True)
        entry = {
            "slug": "ey-test",
            "completed_at": "2026-01-01T00:00:00+00:00",
            "metric": "auc",
            "task_type": "classification",
            "anchor_oof_score": 0.85,
            "variants_passed": 1,
            "cv_strategy": "stratifiedkfold",
            "promoted_branches": [],
            "reproducibility_audit_passed": True,
        }
        (log_dir / "history_log.jsonl").write_text(
            json.dumps(entry) + "\n", encoding="utf-8"
        )

        monkeypatch.setattr(
            zp,
            "resolve_competition_paths",
            lambda slug=None, **kw: _FakePaths(tmp_path, "ey-test"),
        )

        config = {
            "slug": "ey-test",
            "workspace_root": str(tmp_path),
            "metric": "auc",
            "task_type": "classification",
        }
        result = three_lens._eval_phase4_generalisation(config, state)
        assert result.verdict == "PASS", f"Unexpected FAIL findings: {result.findings}"

    def test_phase4_generalisation_fails_without_history_log(
        self, tmp_path, monkeypatch
    ):
        from zindian import three_lens
        import zindian.paths as zp

        comp_dir = tmp_path / "competitions" / "ey-test"
        comp_dir.mkdir(parents=True, exist_ok=True)
        state = {"reproducibility_audit": {"success": True}}
        state_path = comp_dir / "SKILL_STATE.json"
        state_path.write_text(json.dumps(state), encoding="utf-8")

        monkeypatch.setattr(
            zp,
            "resolve_competition_paths",
            lambda slug=None, **kw: _FakePaths(tmp_path, "ey-test"),
        )

        config = {
            "slug": "ey-test",
            "workspace_root": str(tmp_path),
        }
        result = three_lens._eval_phase4_generalisation(config, state)
        assert result.verdict == "FAIL"
        assert any("history_log" in f for f in result.findings)


class TestIdempotentWriteHistoryLog:
    def test_rerun_updates_entry_in_place(self, tmp_path):
        from zindian.skills.skill_22_reproducibility_audit import (
            write_history_log_entry,
        )

        config = _minimal_config("comp-x")
        state1 = _minimal_state("comp-x")
        state1["anchor_oof_score"] = 0.80

        write_history_log_entry(tmp_path, config, state1)

        log_path = tmp_path / "competition_history" / "history_log.jsonl"
        lines1 = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines1) == 1
        entry1 = json.loads(lines1[0])
        assert entry1["anchor_oof_score"] == 0.80

        # Rerun with updated score
        state2 = _minimal_state("comp-x")
        state2["anchor_oof_score"] = 0.88
        write_history_log_entry(tmp_path, config, state2)

        lines2 = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines2) == 1  # Updated in place, no duplicate line
        entry2 = json.loads(lines2[0])
        assert entry2["anchor_oof_score"] == 0.88

    def test_different_competitions_preserved(self, tmp_path):
        from zindian.skills.skill_22_reproducibility_audit import (
            write_history_log_entry,
        )

        write_history_log_entry(
            tmp_path, _minimal_config("comp-1"), _minimal_state("comp-1")
        )
        write_history_log_entry(
            tmp_path, _minimal_config("comp-2"), _minimal_state("comp-2")
        )
        write_history_log_entry(
            tmp_path, _minimal_config("comp-1"), _minimal_state("comp-1")
        )

        log_path = tmp_path / "competition_history" / "history_log.jsonl"
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        slugs = [json.loads(line)["slug"] for line in lines]
        assert slugs == ["comp-2", "comp-1"]


# ---------------------------------------------------------------------------
# Shared fake paths helper
# ---------------------------------------------------------------------------


class _FakePaths:
    def __init__(self, root: Path, slug: str):
        self.root = root
        self.competition_dir = root / "competitions" / slug
        self.state_path = self.competition_dir / "SKILL_STATE.json"
