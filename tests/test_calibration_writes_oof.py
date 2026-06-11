import json
from pathlib import Path

import numpy as np

from zindian.skills import skill_09_calibration as calib


def test_calibration_writes_oof(tmp_path, monkeypatch):
    proc = tmp_path / "data" / "processed"
    proc.mkdir(parents=True)

    # create train features with target and 2-fold stratification support
    (proc / "features_train.csv").write_text("ID,target\n1,0\n2,1\n3,0\n4,1\n", encoding="utf-8")

    state_payload = {
        "competition": "cmp",
        "md5_target_hash": None,
        "anchor_oof_f1": 0.1,
        "anchor_oof_rmse": None,
        "anchor_lb_score": 0.05,
        "submissions_used_today": 0,
        "submissions_used_total": 0,
        "remaining_submissions": 10,
        "dag_phase": "phase_3_features",
        "selected_submissions": [],
        "last_updated": None,
        "best_variant_this_round": "variant-001",
        "human_gate_2_variant-001_approved": True,
        "branch_variant-001_oof": {
            "scores": [0.2, 0.8, 0.3, 0.7],
            "cv_strategy_id": "config:stratified",
            "seed": 42,
            "branch_name": "variant-001",
            "model_config": {"name": "fixture"},
        },
    }
    (proc.parent / "SKILL_STATE.json").write_text(json.dumps(state_payload), encoding="utf-8")

    # create dummy test probs
    (proc / "test_probs_variant-001.csv").write_text("ID,Pred\n5,0.3\n6,0.6\n", encoding="utf-8")

    class SimplePaths:
        def __init__(self, proc_dir: Path):
            self.data_processed_dir = proc_dir
            self.reports_dir = proc_dir / "reports"
            self.competition_dir = proc_dir.parent
            self.state_path = proc_dir.parent / "SKILL_STATE.json"

    monkeypatch.setattr(calib, "resolve_competition_paths", lambda require_competition=True: SimplePaths(proc))

    class FakeCfg:
        def __init__(self):
            self._data = {"reproducibility": {"seed": 42}, "cv_strategy": {"type": "stratified", "n_splits": 2}, "task_type": "classification"}
        def get(self, key, default=None):
            if key == "target_column":
                return "target"
            return default

    monkeypatch.setattr(calib, "ChallengeConfig", type("C", (), {"load": staticmethod(lambda: FakeCfg())}))

    calls = {}

    def fake_write_oof_record(store, **kwargs):
        calls["called"] = True
        calls["kwargs"] = kwargs
        return kwargs

    monkeypatch.setattr(calib, "write_oof_record", fake_write_oof_record)

    res = calib.run(method="isotonic", dry_run=False)
    assert res["status"] == "OK"
    assert calls.get("called", False) is True
    assert calls["kwargs"]["branch_name"] == "calibration_variant-001"
    assert isinstance(calls["kwargs"]["scores"], list)
