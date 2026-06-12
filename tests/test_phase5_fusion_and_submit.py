import json
from pathlib import Path


from zindian.skills import skill_13_oracle_fusion as fusion
from zindian.skills import skill_16_submit as submitter


class SimplePaths:
    def __init__(self, proc_dir: Path):
        self.data_processed_dir = proc_dir
        self.reports_dir = proc_dir / "reports"
        # competition_dir should be the root containing 'data'
        self.competition_dir = proc_dir.parent.parent
        self.data_raw_dir = self.competition_dir / "data" / "raw"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        (proc_dir.parent / "submissions").mkdir(parents=True, exist_ok=True)
        self.state_path = proc_dir.parent / "SKILL_STATE.json"


def test_fusion_dry_run(tmp_path, monkeypatch):
    proc = tmp_path / "data" / "processed"
    proc.mkdir(parents=True)

    # train with target
    (proc / "features_train.csv").write_text("ID,occ\n1,0\n2,1\n", encoding="utf-8")

    # test probs for branch candidate
    (proc / "test_probs_variant-1.csv").write_text(
        "ID,Pred\n3,0.3\n4,0.6\n", encoding="utf-8"
    )

    # minimal but schema-valid state
    from zindian.schemas import skill_state_skeleton

    state = skill_state_skeleton()
    state.update(
        {
            "anchor_oof_f1": 0.1,
            "anchor_lb_score": 0.05,
            "competition": "cmp",
            "human_gate_2_variant-1_approved": True,
            "branch_variant-1_oof": {
                "scores": [0.2, 0.8],
                "cv_strategy_id": "config:stratifiedkfold",
                "seed": 42,
                "branch_name": "variant-1",
                "model_config": {"name": "fixture"},
            },
        }
    )
    (proc.parent / "SKILL_STATE.json").write_text(json.dumps(state), encoding="utf-8")

    monkeypatch.setattr(
        fusion,
        "resolve_competition_paths",
        lambda require_competition=True: SimplePaths(proc),
    )

    class FakeCfg:
        def get(self, key, default=None):
            if key == "target_column":
                return "occ"
            return default

    monkeypatch.setattr(
        fusion,
        "ChallengeConfig",
        type("C", (), {"load": staticmethod(lambda: FakeCfg())}),
    )

    res = fusion.run(dry_run=True)
    assert res["status"] == "DRY_RUN"


def test_submission_validate_and_determine(tmp_path):
    # create sample and a correct submission
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    sample = raw / "SampleSubmission.csv"
    sample.write_text("ID,Prediction\n3,0\n4,1\n", encoding="utf-8")

    sub = tmp_path / "sub.csv"
    sub.write_text("ID,Prediction\n3,0\n4,1\n", encoding="utf-8")

    errors = submitter.validate(sub, sample)
    assert not errors

    # mismatch case
    bad = tmp_path / "bad.csv"
    bad.write_text("IDX,Pred\n3,0\n4,1\n", encoding="utf-8")
    errs = submitter.validate(bad, sample)
    assert errs
