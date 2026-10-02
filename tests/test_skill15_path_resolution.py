"""skill_15 path-resolution and live-tree safety.

Covers three things established on 2026-10-02:

1. The explicit-paths path (``42097d5``) must never call the resolver.
2. The ERROR path must also never call the resolver when the caller passed
   explicit paths. It did until this commit -- the happy path was fixed and
   the error path was left writing ``skill_15_error.jsonl`` into the live
   competition tree.
3. The no-argument path (what a real Phase 4 run uses) MUST still call the
   resolver and write under the resolved reports dir. Commit ``42097d5``
   changed production code with no test on this branch, so it is pinned here.

Binding note: ``skill_15_reporter`` does ``from zindian.paths import
resolve_competition_paths``, binding the name into its own module
namespace. Patching ``zindian.paths.resolve_competition_paths`` therefore
does NOT intercept skill_15's calls -- these tests patch the skill module's
attribute, which is what actually governs the code path under test.
"""

from pathlib import Path

import json
import pytest

# Captured at import so the live-tree check below can distinguish a file this
# test session touched from one left by the earlier incident.
_TEST_START = __import__("time").time()

from zindian.skills import skill_15_reporter as s15


def _write_state_and_config(comp_dir):
    """Minimal but VALID competition tree.

    The config must satisfy ChallengeConfig.load's required-key set --
    otherwise skill_15 returns at its inner config-load handler
    (skill_15_reporter.py:161) and never exercises the error path under test.
    """
    reports = comp_dir / "reports"
    (reports / "sessions").mkdir(parents=True, exist_ok=True)
    (reports / "summaries").mkdir(parents=True, exist_ok=True)
    state = comp_dir / "SKILL_STATE.json"
    state.write_text(
        '{"dag_phase": "phase_3b_gating", "anchor_oof_score": 0.8}', encoding="utf-8"
    )
    config = comp_dir / "challenge_config.json"
    config.write_text(
        json.dumps(
            {
                "name": "tmp-test",
                "slug": "tmp-test",
                "metric": "f1",
                "metric_direction": "maximize",
                "submission_format": "csv",
                "use_probabilities": True,
                "daily_limit": 5,
                "total_limit": 50,
                "public_split_pct": 0.5,
                "private_split_pct": 0.5,
                "team_allowed": True,
                "code_review_tier": "none",
                "allowed_external_data": True,
                "automl_permitted": False,
                "data_modality": "tabular",
                "domain": "health",
                "task_type": "classification",
            }
        ),
        encoding="utf-8",
    )
    return {
        "state": str(state),
        "config": str(config),
        "reports": str(reports),
        "comp_dir": comp_dir,
    }


class _Tripwire:
    """Raises if skill_15 reaches the global resolver."""

    def __init__(self):
        self.calls = 0

    def __call__(self, *a, **kw):
        self.calls += 1
        raise AssertionError("skill_15 called resolve_competition_paths() unexpectedly")


@pytest.fixture
def explicit(tmp_path):
    """Explicit-path args plus a tripwire over the resolver."""
    paths = _write_state_and_config(tmp_path / "comp")
    trip = _Tripwire()
    monkey = pytest.MonkeyPatch()
    monkey.setattr(s15, "resolve_competition_paths", trip)
    try:
        yield paths, trip
    finally:
        monkey.undo()


def test_explicit_paths_never_resolve(explicit):
    """Happy path: all three explicit -> resolver untouched."""
    p, trip = explicit
    s15.run(
        ledger_path=str(Path(p["comp_dir"]) / "reports" / "experiments.db"),
        state_path=p["state"],
        config_path=p["config"],
    )
    assert trip.calls == 0, "happy path must not reach the resolver"


def test_error_path_does_not_leak_into_live_tree(explicit):
    """Failure path must honour explicit paths too -- the 42097d5 gap.

    Point ledger_path at a nonexistent directory so the run raises inside
    skill_15, then assert the error log lands in the tmp reports dir and
    that no resolver call was made.
    """
    p, trip = explicit
    # A directory cannot be opened as a DuckDB file -> the run raises inside
    # skill_15 and takes the error path.
    bad_ledger = Path(p["comp_dir"]) / "reports" / "ledger_is_a_dir"
    bad_ledger.mkdir(parents=True, exist_ok=True)
    s15.run(
        ledger_path=str(bad_ledger),
        state_path=p["state"],
        config_path=p["config"],
    )
    assert trip.calls == 0, "ERROR path leaked to the resolver"
    err = Path(p["reports"]) / "sessions" / "skill_15_error.jsonl"
    assert err.exists(), "error log should be written under explicit reports dir"

    # The live tree already contains a skill_15_error.jsonl from the 2026-10-02
    # incident, so assert it was NOT TOUCHED rather than that it is absent --
    # absence would be a false pass on a pristine tree.
    repo_live = Path(__file__).resolve().parent.parent / "competitions"
    for stray in repo_live.glob("*/reports/sessions/skill_15_error.jsonl"):
        assert stray.stat().st_mtime < _TEST_START, (
            f"ERROR path wrote into the live tree: {stray}"
        )


def test_no_argument_path_still_uses_resolver(monkeypatch, tmp_path):
    """Live-behaviour regression: the no-arg path MUST still resolve.

    This is what a real Phase 4 run does. If someone 'fixes' the leak by
    removing the resolver call unconditionally, this test fails.
    """
    comp_dir = tmp_path / "live-run"
    reports = Path(_write_state_and_config(comp_dir)["reports"])
    calls = {"n": 0}

    class _FakePaths:
        pass

    def fake_resolve(*a, **kw):
        calls["n"] += 1
        p = _FakePaths()
        p.root = comp_dir
        p.competition_dir = comp_dir
        p.state_path = comp_dir / "SKILL_STATE.json"
        p.config_path = comp_dir / "challenge_config.json"
        p.reports_dir = reports
        return p

    monkeypatch.setattr(s15, "resolve_competition_paths", fake_resolve)
    s15.run(phase="1")
    assert calls["n"] == 1, "no-arg path must call the resolver exactly once"
    assert reports.exists()