"""
Tests for Ledger in-process thread safety (F3 fix).

Verifies that concurrent calls to log_experiment, log_submission, update_gate_result,
and update_submission_final_selection across multiple threads do not cause database
locking errors, row corruption, or lost writes.
"""

from __future__ import annotations

import threading
import pytest
from pathlib import Path

from zindian.ledger import Ledger


def test_concurrent_log_experiment_no_corruption(tmp_path: Path):
    """
    Spawn 10 threads each calling ledger.log_experiment() concurrently.
    All 10 experiments must be logged without locking errors or lost writes.
    """
    db_path = tmp_path / "experiments.db"
    ledger = Ledger(str(db_path))

    errors: list[Exception] = []

    def _worker(thread_idx: int):
        try:
            ledger.log_experiment(
                branch_name=f"branch_thread_{thread_idx}",
                oof_score=0.80 + (thread_idx * 0.01),
                metric="auc",
                feature_count=10 + thread_idx,
                gate_result="PASS" if thread_idx % 2 == 0 else "FAIL",
                notes=f"Logged by thread {thread_idx}",
            )
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Concurrent log_experiment raised errors: {errors}"

    passed = ledger.get_passed_experiments()
    failed = ledger.get_failed_experiments()

    # Total experiments should equal 10
    total_logged = len(passed) + len(failed)
    assert total_logged == 10, f"Expected 10 logged experiments, got {total_logged}"

    ledger.close()


def test_daemon_thread_and_main_thread_concurrent_writes(tmp_path: Path):
    """
    Simulate run_deep_research pattern: daemon thread logs an experiment while
    main thread logs a submission simultaneously.
    """
    db_path = tmp_path / "experiments.db"
    ledger = Ledger(str(db_path))

    exp_id = ledger.log_experiment(
        branch_name="anchor-v1",
        oof_score=0.85,
        metric="f1",
        gate_result="PASS",
    )

    errors: list[Exception] = []

    def _daemon_task():
        try:
            for i in range(5):
                ledger.log_experiment(
                    branch_name=f"sidecar_research_branch_{i}",
                    oof_score=0.86 + (i * 0.001),
                    metric="f1",
                    gate_result="PENDING",
                    notes="Deep research sidecar findings",
                )
        except Exception as e:
            errors.append(e)

    def _main_task():
        try:
            for i in range(5):
                ledger.log_submission(
                    experiment_id=exp_id,
                    branch_name="anchor-v1",
                    submission_rank=i + 1,
                    public_score=0.850 + (i * 0.001),
                    comment=f"Main thread submission {i}",
                )
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=_daemon_task)
    t2 = threading.Thread(target=_main_task)

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert not errors, f"Concurrent daemon and main thread operations failed: {errors}"

    subs = ledger.get_submissions()
    assert len(subs) == 5, f"Expected 5 submissions, got {len(subs)}"

    ledger.close()


def test_concurrent_updates_and_queries(tmp_path: Path):
    """
    Test concurrent update_gate_result and update_submission_final_selection
    alongside queries.
    """
    db_path = tmp_path / "experiments.db"
    ledger = Ledger(str(db_path))

    exp_id = ledger.log_experiment(
        branch_name="test_update_branch",
        oof_score=0.75,
        metric="auc",
        gate_result="PENDING",
    )
    sub_id = ledger.log_submission(
        experiment_id=exp_id,
        branch_name="test_update_branch",
        public_score=0.76,
    )

    errors: list[Exception] = []

    def _updater():
        try:
            ledger.update_gate_result(exp_id, "PASS", "Manually verified")
            ledger.update_submission_final_selection(sub_id, True, "Best public score")
        except Exception as e:
            errors.append(e)

    def _reader():
        try:
            _ = ledger.get_experiment(exp_id)
            _ = ledger.get_selected_submissions()
        except Exception as e:
            errors.append(e)

    t_up = threading.Thread(target=_updater)
    t_rd = threading.Thread(target=_reader)

    t_up.start()
    t_rd.start()
    t_up.join()
    t_rd.join()

    assert not errors, f"Concurrent updates/reads raised errors: {errors}"

    exp = ledger.get_experiment(exp_id)
    assert exp is not None
    assert exp["gate_result"] == "PASS"

    sel_subs = ledger.get_selected_submissions()
    assert len(sel_subs) == 1
    assert sel_subs[0]["submission_id"] == sub_id

    ledger.close()
