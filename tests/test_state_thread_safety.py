"""
Tests for SkillStateStore thread safety (F2 fix).

Confirms that concurrent read-modify-write operations on SkillStateStore
do not produce lost updates when the deep-research daemon thread and the
main pipeline thread both call store.update() / store.increment() / etc.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path


from zindian.state import SkillStateStore
from zindian.schemas import skill_state_skeleton


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store(tmp_path: Path) -> SkillStateStore:
    state_path = tmp_path / "SKILL_STATE.json"
    state_path.write_text(json.dumps(skill_state_skeleton()), encoding="utf-8")
    return SkillStateStore(state_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConcurrentUpdates:
    def test_no_lost_writes_across_20_threads(self, tmp_path):
        """Each thread writes a unique key; all 20 must survive."""
        store = _make_store(tmp_path)
        n = 20
        errors: list[Exception] = []

        def _worker(i: int) -> None:
            try:
                store.update(**{f"probe_{i}": i})
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        state = store.read()
        for i in range(n):
            assert f"probe_{i}" in state, f"probe_{i} was lost"
            assert state[f"probe_{i}"] == i

    def test_no_lost_increments_across_50_threads(self, tmp_path):
        """50 threads each increment the same counter; final value must be 50."""
        store = _make_store(tmp_path)
        n = 50
        errors: list[Exception] = []

        def _worker() -> None:
            try:
                store.increment("counter")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_worker) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        state = store.read()
        assert state["counter"] == n, (
            f"Expected counter={n}, got {state.get('counter')} — "
            f"{n - state.get('counter', 0)} increment(s) were lost"
        )

    def test_daemon_thread_pattern_matches_orchestrator(self, tmp_path):
        """
        Mirror the exact run_deep_research pattern:
          - daemon thread calls store.update(sidecar_done=True)
          - main thread calls store.update(main_done=True) concurrently
          - both keys must survive in the final state
        """
        store = _make_store(tmp_path)
        barrier = threading.Barrier(2)  # synchronise start for maximum contention
        errors: list[Exception] = []

        def _daemon_worker() -> None:
            try:
                barrier.wait()
                store.update(sidecar_done=True)
            except Exception as exc:
                errors.append(exc)

        t = threading.Thread(target=_daemon_worker, daemon=True)
        t.start()

        barrier.wait()  # main thread starts at the same moment
        try:
            store.update(main_done=True)
        except Exception as exc:
            errors.append(exc)

        t.join(timeout=5.0)
        assert not t.is_alive(), "Daemon thread did not finish in time"
        assert not errors, f"Thread errors: {errors}"

        state = store.read()
        assert state.get("sidecar_done") is True, "sidecar_done key was lost"
        assert state.get("main_done") is True, "main_done key was lost"

    def test_append_selected_no_duplicates_under_concurrency(self, tmp_path):
        """
        Multiple threads calling append_selected with the same ID must not
        produce duplicates (the de-dup guard inside the lock must hold).
        """
        store = _make_store(tmp_path)
        n = 10
        errors: list[Exception] = []

        def _worker() -> None:
            try:
                store.append_selected(999)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_worker) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        state = store.read()
        selected = state.get("selected_submissions", [])
        assert (
            selected.count(999) == 1
        ), f"Expected exactly 1 entry for ID 999, got {selected.count(999)}"

    def test_lock_is_class_level(self, tmp_path):
        """
        SkillStateStore instances must share a class-level lock so that separate
        instantiations (e.g. main thread vs daemon thread) synchronize state writes.
        """
        path_a = tmp_path / "a" / "SKILL_STATE.json"
        path_b = tmp_path / "b" / "SKILL_STATE.json"
        path_a.parent.mkdir(parents=True)
        path_b.parent.mkdir(parents=True)
        path_a.write_text(json.dumps(skill_state_skeleton()), encoding="utf-8")
        path_b.write_text(json.dumps(skill_state_skeleton()), encoding="utf-8")

        store_a = SkillStateStore(path_a)
        store_b = SkillStateStore(path_b)

        # Locks must be the same class-level lock object
        assert store_a._lock is store_b._lock, (
            "SkillStateStore instances do not share the class-level lock. "
            "Separate instantiations in daemon threads would fail to synchronize."
        )
