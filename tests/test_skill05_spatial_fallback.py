import os
import json
import pandas as pd
from pathlib import Path
from zindian.skills import skill_05_cv


def test_run_falls_back_on_sparse_spatial(tmp_path, monkeypatch):
    # Create competition structure
    # Create competition under repo root so resolver finds it via COMPETITION_SLUG
    repo_root = os.getcwd()
    comp = Path(repo_root) / "competitions" / "tmpcomp"
    comp.mkdir(parents=True, exist_ok=True)
    # Use template to create a schema-valid SKILL_STATE.json for the test
    ss_template = Path(os.getcwd()) / "templates" / "SKILL_STATE_template.json"
    ss = {}
    if ss_template.exists():
        ss = json.loads(ss_template.read_text(encoding="utf-8"))
    ss.update({"dag_phase": "phase_1_complete", "eda": {}})
    # Ensure legacy/required keys exist
    ss.setdefault("anchor_oof_f1", None)
    (comp / "SKILL_STATE.json").write_text(json.dumps(ss), encoding="utf-8")

    # challenge_config with spatial signal present and minority_ratio small
    # Use the repository template to produce a full, schema-valid config
    template_path = Path(os.getcwd()) / "templates" / "challenge_config_template.json"
    base = {}
    if template_path.exists():
        base = json.loads(template_path.read_text(encoding="utf-8"))
    base.update({
        "name": "tmpcomp",
        "slug": "tmpcomp",
        "metric": "f1_score",
        "metric_direction": "maximize",
        "submission_format": "csv",
        "use_probabilities": False,
        "daily_limit": 10,
        "total_limit": 100,
        "public_split_pct": 20,
        "private_split_pct": 80,
        "team_allowed": True,
        "code_review_tier": None,
        "allowed_external_data": False,
        "automl_permitted": False,
        "data_modality": "tabular",
        "domain": "test",
        "task_type": "classification",
        "target_col": "target",
        "minority_ratio": 0.10,
        "spatial_signal": {"present": True},
        "latitude_column": "Latitude",
        "longitude_column": "Longitude",
    })
    (comp / "challenge_config.json").write_text(json.dumps(base), encoding="utf-8")

    # features_train.csv with fewer rows than N_SPLITS (N_SPLITS=5)
    df = pd.DataFrame({
        "ID": [1,2,3,4],
        "Latitude": [0.0, 1.0, 2.0, 3.0],
        "Longitude": [0.0, 1.0, 2.0, 3.0],
        "target": [0,1,0,1],
        "f1": [0.1,0.2,0.3,0.4]
    })
    data_dir = comp / "data" / "processed"
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_dir / "features_train.csv", index=False)

    # Point the resolver to this competition
    monkeypatch.setenv("COMPETITION_SLUG", "tmpcomp")
    # Run CV architect and expect fallback to StratifiedKFold due to small spatial sample
    res = skill_05_cv.run(strategy="compare")
    assert res["strategy_chosen"] in ("StratifiedKFold", "KFold")
