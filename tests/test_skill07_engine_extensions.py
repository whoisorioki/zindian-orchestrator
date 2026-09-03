"""Unit tests for v2.9 skill_07 feature engineering engine extensions:

- rolling_aggregates (long-memory spatial-temporal windows)
- static_bins (cohort discretization with custom edges)
- cascaded execution order (interactions referencing Stage 1 base outputs)
- target leakage guards
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from zindian.skills.skill_07_features import build_hypothesis_features


def _make_sample_frames():
    train = pd.DataFrame(
        {
            "location_id": ["loc1", "loc1", "loc1", "loc2", "loc2"],
            "date": [
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
                "2020-01-01",
                "2020-01-02",
            ],
            "age": [4.0, 10.0, 35.0, 70.0, 2.0],
            "tavg": [20.0, 22.0, 24.0, 15.0, 18.0],
            "target": [0, 1, 0, 1, 0],
        }
    )
    test = pd.DataFrame(
        {
            "location_id": ["loc1", "loc2"],
            "date": ["2020-01-04", "2020-01-03"],
            "age": [50.0, 4.0],
            "tavg": [25.0, 19.0],
        }
    )
    return train, test


def test_rolling_aggregates_computation():
    train, test = _make_sample_frames()
    fe_cfg = {
        "rolling_aggregates": [
            {
                "column": "tavg",
                "group_by": "location_id",
                "sort_by": "date",
                "window": 2,
                "min_periods": 1,
                "function": "mean",
                "name": "tavg_2d_mean",
            }
        ]
    }
    tr_out, te_out = build_hypothesis_features(
        train, test, mode="inference", merged_fe_cfg=fe_cfg
    )

    assert "tavg_2d_mean" in tr_out.columns
    assert "tavg_2d_mean" in te_out.columns

    # Verify loc1 rolling mean on train:
    # 2020-01-01 -> 20.0
    # 2020-01-02 -> (20+22)/2 = 21.0
    # 2020-01-03 -> (22+24)/2 = 23.0
    loc1_tr_means = tr_out[tr_out["location_id"] == "loc1"]["tavg_2d_mean"].tolist()
    assert loc1_tr_means == [20.0, 21.0, 23.0]


def test_static_bins_cohort_discretization():
    train, test = _make_sample_frames()
    fe_cfg = {
        "static_bins": [
            {
                "column": "age",
                "edges": [0, 5, 15, 64, 120],
                "labels": [0, 1, 2, 3],
                "name": "age_cohort",
            }
        ]
    }
    tr_out, te_out = build_hypothesis_features(
        train, test, mode="inference", merged_fe_cfg=fe_cfg
    )

    assert "age_cohort" in tr_out.columns
    assert "age_cohort" in te_out.columns

    # Verify age bin indices: 4.0 -> 0 (<5), 10.0 -> 1 (5-15), 35.0 -> 2 (15-64), 70.0 -> 3 (>64)
    assert tr_out["age_cohort"].tolist() == [0.0, 1.0, 2.0, 3.0, 0.0]


def test_cascaded_interaction_with_stage1_outputs():
    train, test = _make_sample_frames()
    fe_cfg = {
        "rolling_aggregates": [
            {
                "column": "tavg",
                "group_by": "location_id",
                "sort_by": "date",
                "window": 2,
                "min_periods": 1,
                "function": "mean",
                "name": "tavg_2d_mean",
            }
        ],
        "static_bins": [
            {
                "column": "age",
                "edges": [0, 5, 15, 64, 120],
                "labels": [0, 1, 2, 3],
                "name": "age_cohort",
            }
        ],
        "interactions": [
            ["age", "tavg_2d_mean"],
            ["age_cohort", "tavg_2d_mean"],
        ],
    }
    tr_out, te_out = build_hypothesis_features(
        train, test, mode="inference", merged_fe_cfg=fe_cfg
    )

    assert "age_x_tavg_2d_mean" in tr_out.columns
    assert "age_cohort_x_tavg_2d_mean" in tr_out.columns

    # Verify age_x_tavg_2d_mean for first row: 4.0 * 20.0 = 80.0
    assert tr_out["age_x_tavg_2d_mean"].iloc[0] == 80.0


def test_rolling_aggregates_target_leakage_guard(monkeypatch):
    train, test = _make_sample_frames()

    class FakeCfg:
        @staticmethod
        def load():
            class C:
                _data = {"target_col": "target"}

            return C()

    monkeypatch.setattr("zindian.config.ChallengeConfig.load", FakeCfg.load)

    fe_cfg = {
        "rolling_aggregates": [
            {
                "column": "target",
                "group_by": "location_id",
                "sort_by": "date",
                "window": 2,
            }
        ]
    }
    with pytest.raises(ValueError, match="Target column"):
        build_hypothesis_features(train, test, mode="inference", merged_fe_cfg=fe_cfg)


def test_ai4eac_variant_sidecar_end_to_end():
    import json
    from pathlib import Path

    variant_file = (
        Path(__file__).parents[1]
        / "competitions"
        / "climate-risk-health-prediction-challenge"
        / "variants"
        / "ai4eac-longmemory-cohorts.json"
    )
    assert variant_file.exists(), f"Sidecar missing: {variant_file}"

    with open(variant_file, "r") as f:
        sidecar = json.load(f)

    train = pd.DataFrame(
        {
            "location": ["loc1", "loc1", "loc1", "loc2", "loc2"],
            "date": [
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
                "2020-01-01",
                "2020-01-02",
            ],
            "age": [4.0, 10.0, 35.0, 70.0, 2.0],
            "gender": [0, 1, 0, 1, 0],
            "avg_temperature": [20.0, 22.0, 24.0, 15.0, 18.0],
            "max_temperature": [25.0, 27.0, 29.0, 20.0, 23.0],
            "min_temperature": [15.0, 17.0, 19.0, 10.0, 13.0],
            "precipitation": [0.1, 0.5, 0.0, 2.1, 1.2],
            "latitude": [10.0, 10.0, 10.0, 12.0, 12.0],
            "longitude": [30.0, 30.0, 30.0, 32.0, 32.0],
            "is_climate_sensitive": [0, 1, 0, 1, 0],
        }
    )
    test = pd.DataFrame(
        {
            "location": ["loc1", "loc2"],
            "date": ["2020-01-04", "2020-01-03"],
            "age": [50.0, 4.0],
            "gender": [1, 0],
            "avg_temperature": [25.0, 19.0],
            "max_temperature": [30.0, 24.0],
            "min_temperature": [20.0, 14.0],
            "precipitation": [0.0, 0.8],
            "latitude": [10.0, 12.0],
            "longitude": [30.0, 32.0],
        }
    )

    tr_out, te_out = build_hypothesis_features(
        train,
        test,
        mode="inference",
        merged_fe_cfg=sidecar.get("feature_engineering", {}),
    )

    whitelist = sidecar.get("feature_columns", [])
    for col in whitelist:
        assert col in tr_out.columns, f"Missing column in train: {col}"
        assert col in te_out.columns, f"Missing column in test: {col}"

    assert not tr_out[whitelist].isna().any().any()
    assert not te_out[whitelist].isna().any().any()


def test_cohort_only_sidecar_end_to_end():
    import json
    from pathlib import Path

    variant_file = (
        Path(__file__).parents[1]
        / "competitions"
        / "climate-risk-health-prediction-challenge"
        / "variants"
        / "cohort-only.json"
    )
    assert variant_file.exists(), f"Sidecar missing: {variant_file}"

    with open(variant_file, "r") as f:
        sidecar = json.load(f)

    train, test = _make_sample_frames()
    train["gender"] = [0, 1, 0, 1, 0]
    test["gender"] = [1, 0]
    train["avg_temperature"] = [20.0, 22.0, 24.0, 15.0, 18.0]
    test["avg_temperature"] = [25.0, 19.0]
    train["max_temperature"] = [25.0, 27.0, 29.0, 20.0, 23.0]
    test["max_temperature"] = [30.0, 24.0]
    train["min_temperature"] = [15.0, 17.0, 19.0, 10.0, 13.0]
    test["min_temperature"] = [20.0, 14.0]
    train["precipitation"] = [0.1, 0.5, 0.0, 2.1, 1.2]
    test["precipitation"] = [0.0, 0.8]
    train["latitude"] = [10.0, 10.0, 10.0, 12.0, 12.0]
    test["latitude"] = [10.0, 12.0]
    train["longitude"] = [30.0, 30.0, 30.0, 32.0, 32.0]
    test["longitude"] = [30.0, 32.0]
    train["location"] = train["location_id"]
    test["location"] = test["location_id"]

    tr_out, te_out = build_hypothesis_features(
        train,
        test,
        mode="inference",
        merged_fe_cfg=sidecar.get("feature_engineering", {}),
    )

    whitelist = sidecar.get("feature_columns", [])
    for col in whitelist:
        assert col in tr_out.columns, f"Missing column in train: {col}"
        assert col in te_out.columns, f"Missing column in test: {col}"

    assert "age_cohort" in whitelist
    assert "age" not in whitelist


def test_climate_longmemory_only_sidecar_end_to_end():
    import json
    from pathlib import Path

    variant_file = (
        Path(__file__).parents[1]
        / "competitions"
        / "climate-risk-health-prediction-challenge"
        / "variants"
        / "climate-longmemory-only.json"
    )
    assert variant_file.exists(), f"Sidecar missing: {variant_file}"

    with open(variant_file, "r") as f:
        sidecar = json.load(f)

    train, test = _make_sample_frames()
    train["gender"] = [0, 1, 0, 1, 0]
    test["gender"] = [1, 0]
    train["avg_temperature"] = [20.0, 22.0, 24.0, 15.0, 18.0]
    test["avg_temperature"] = [25.0, 19.0]
    train["max_temperature"] = [25.0, 27.0, 29.0, 20.0, 23.0]
    test["max_temperature"] = [30.0, 24.0]
    train["min_temperature"] = [15.0, 17.0, 19.0, 10.0, 13.0]
    test["min_temperature"] = [20.0, 14.0]
    train["precipitation"] = [0.1, 0.5, 0.0, 2.1, 1.2]
    test["precipitation"] = [0.0, 0.8]
    train["latitude"] = [10.0, 10.0, 10.0, 12.0, 12.0]
    test["latitude"] = [10.0, 12.0]
    train["longitude"] = [30.0, 30.0, 30.0, 32.0, 32.0]
    test["longitude"] = [30.0, 32.0]
    train["location"] = train["location_id"]
    test["location"] = test["location_id"]

    tr_out, te_out = build_hypothesis_features(
        train,
        test,
        mode="inference",
        merged_fe_cfg=sidecar.get("feature_engineering", {}),
    )

    whitelist = sidecar.get("feature_columns", [])
    for col in whitelist:
        assert col in tr_out.columns, f"Missing column in train: {col}"
        assert col in te_out.columns, f"Missing column in test: {col}"

    assert "age" in whitelist
    assert "max_temperature_90d_mean" in whitelist
    assert "precipitation_30d_mean" in whitelist


def test_temporal_anomaly_only_sidecar_end_to_end():
    import json
    from pathlib import Path

    variant_file = (
        Path(__file__).parents[1]
        / "competitions"
        / "climate-risk-health-prediction-challenge"
        / "variants"
        / "temporal-anomaly-only.json"
    )
    assert variant_file.exists(), f"Sidecar missing: {variant_file}"

    with open(variant_file, "r") as f:
        sidecar = json.load(f)

    train, test = _make_sample_frames()
    train["gender"] = [0, 1, 0, 1, 0]
    test["gender"] = [1, 0]
    train["avg_temperature"] = [20.0, 22.0, 24.0, 15.0, 18.0]
    test["avg_temperature"] = [25.0, 19.0]
    train["max_temperature"] = [25.0, 27.0, 29.0, 20.0, 23.0]
    test["max_temperature"] = [30.0, 24.0]
    train["min_temperature"] = [15.0, 17.0, 19.0, 10.0, 13.0]
    test["min_temperature"] = [20.0, 14.0]
    train["precipitation"] = [0.1, 0.5, 0.0, 2.1, 1.2]
    test["precipitation"] = [0.0, 0.8]
    train["latitude"] = [10.0, 10.0, 10.0, 12.0, 12.0]
    test["latitude"] = [10.0, 12.0]
    train["longitude"] = [30.0, 30.0, 30.0, 32.0, 32.0]
    test["longitude"] = [30.0, 32.0]
    train["location"] = train["location_id"]
    test["location"] = test["location_id"]

    tr_out, te_out = build_hypothesis_features(
        train,
        test,
        mode="inference",
        merged_fe_cfg=sidecar.get("feature_engineering", {}),
    )

    whitelist = sidecar.get("feature_columns", [])
    for col in whitelist:
        assert col in tr_out.columns, f"Missing column in train: {col}"
        assert col in te_out.columns, f"Missing column in test: {col}"

    assert "date_sin_doy" in whitelist
    assert "date_cos_doy" in whitelist


def test_macro_stress_geofence_sidecar_json_validity():
    """Verify macro-stress-geofence sidecar JSON passes 3-stage feature engine extraction."""
    variant_file = (
        Path(__file__).parents[1]
        / "competitions"
        / "climate-risk-health-prediction-challenge"
        / "variants"
        / "macro-stress-geofence.json"
    )
    assert variant_file.exists(), f"Sidecar missing: {variant_file}"

    with open(variant_file, "r") as f:
        sidecar = json.load(f)

    train, test = _make_sample_frames()
    train["gender"] = [0, 1, 0, 1, 0]
    test["gender"] = [1, 0]
    train["avg_temperature"] = [20.0, 22.0, 24.0, 15.0, 18.0]
    test["avg_temperature"] = [25.0, 19.0]
    train["max_temperature"] = [25.0, 27.0, 29.0, 20.0, 23.0]
    test["max_temperature"] = [30.0, 24.0]
    train["min_temperature"] = [15.0, 17.0, 19.0, 10.0, 13.0]
    test["min_temperature"] = [20.0, 14.0]
    train["precipitation"] = [0.1, 0.5, 0.0, 2.1, 1.2]
    test["precipitation"] = [0.0, 0.8]
    train["latitude"] = [10.0, 10.0, 10.0, 12.0, 12.0]
    test["latitude"] = [10.0, 12.0]
    train["longitude"] = [30.0, 30.0, 30.0, 32.0, 32.0]
    test["longitude"] = [30.0, 32.0]
    train["location"] = train["location_id"]
    test["location"] = test["location_id"]

    # External proxy columns loaded by macro_stress_extractor
    train["spei_6"] = [-0.5, 1.2, -1.8, 0.4, -0.1]
    test["spei_6"] = [-0.2, 0.9]
    train["viirs_radiance_lag1m"] = [1.5, 2.0, 0.8, 12.0, 5.0]
    test["viirs_radiance_lag1m"] = [1.8, 8.0]
    train["spei_6_is_imputed"] = [0, 0, 0, 0, 0]
    test["spei_6_is_imputed"] = [0, 0]
    train["viirs_radiance_lag1m_is_imputed"] = [0, 0, 0, 0, 0]
    test["viirs_radiance_lag1m_is_imputed"] = [0, 0]

    tr_out, te_out = build_hypothesis_features(
        train,
        test,
        mode="inference",
        merged_fe_cfg=sidecar.get("feature_engineering", {}),
    )

    whitelist = sidecar.get("feature_columns", [])
    for col in whitelist:
        assert col in tr_out.columns, f"Missing column in train: {col}"
        assert col in te_out.columns, f"Missing column in test: {col}"





