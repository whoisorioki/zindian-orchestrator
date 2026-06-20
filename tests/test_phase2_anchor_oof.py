import json
import numpy as np
from pathlib import Path

from zindian.skills import skill_08_anchor as anchor


def make_anchor_comp(tmp_path: Path):
    comp = tmp_path / "competitions" / "anchor-cmp"
    comp.mkdir(parents=True)
    # minimal features_train/test and sample submission
    train = comp / "data" / "processed"
    raw = comp / "data" / "raw"
    train.mkdir(parents=True)
    raw.mkdir(parents=True)
    # features_train.csv
    ft = train / "features_train.csv"
    ft.write_text("ID,Occurrence Status,feat1\n1,1,0.1\n2,0,0.2\n", encoding="utf-8")
    (train / "features_test.csv").write_text(
        "ID,feat1\n3,0.3\n4,0.4\n", encoding="utf-8"
    )
    # sample submission
    (raw / "SampleSubmission.csv").write_text(
        "ID,Prediction\n3,0\n4,0\n", encoding="utf-8"
    )
    # SKILL_STATE skeleton
    from zindian.schemas import skill_state_skeleton

    (comp / "SKILL_STATE.json").write_text(
        json.dumps(skill_state_skeleton()), encoding="utf-8"
    )
    # challenge_config minimal
    cfg = {
        "name": "anchor-test",
        "slug": "anchor-cmp",
        "metric": "f1_score",
        "metric_direction": "maximize",
        "submission_format": None,
        "use_probabilities": False,
        "daily_limit": None,
        "total_limit": None,
        "public_split_pct": None,
        "private_split_pct": None,
        "team_allowed": None,
        "code_review_tier": None,
        "allowed_external_data": True,
        "automl_permitted": False,
        "data_modality": "tabular",
        "domain": None,
        "reproducibility": {"seed": 42},
    }
    (comp / "challenge_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return comp


def test_anchor_writes_oof(tmp_path, monkeypatch):
    make_anchor_comp(tmp_path)
    monkeypatch.chdir(tmp_path)

    # Mock compute_oof_predictions to avoid heavy training
    def fake_compute(train, test, config, target_col, n_splits=5, random_seed=None):
        oof = np.array([0.9, 0.1])
        testp = np.array([0.8, 0.2])
        return oof, testp, 0.1, 0.95, 0.6, 0.5

    monkeypatch.setattr(anchor, "compute_oof_predictions", fake_compute)

    # Replace Ledger with a lightweight fake to avoid DB operations
    class FakeLedger:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.close()

        def log_experiment(self, *a, **k):
            return 1

        def close(self):
            return None

    monkeypatch.setattr(anchor, "Ledger", FakeLedger)

    anchor.run(submit=False)
    state_path = Path("competitions") / "anchor-cmp" / "SKILL_STATE.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    # Branch name used is 'anchor-baseline'
    key = "branch_anchor-baseline_oof"
    assert key in state
    rec = state[key]
    assert rec["branch_name"] == "anchor-baseline"
    assert "scores" in rec and isinstance(rec["scores"], list)
