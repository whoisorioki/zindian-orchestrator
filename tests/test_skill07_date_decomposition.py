"""Unit tests for the skill_07 feature_engineering extension:

- date_decomposition op (calendar parts + cyclical sin/cos encodings)
- drop_columns / drop_source return-path column removal
- protected-column drop guard (target / id)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zindian.skills.skill_07_features import build_hypothesis_features


def _make_frames():
    train = pd.DataFrame(
        {
            "ID": ["a", "b", "c", "d"],
            "age": [30.0, 2.0, 69.0, 45.0],
            "event_date": [
                "2008-01-15",
                "2008-03-20",
                "2008-07-04",
                "2008-12-31",
            ],
            "unused": [1.0, 2.0, 3.0, 4.0],
        }
    )
    test = train.drop(columns=["age"]).copy()
    return train, test


def _fake_config_cls(payload: dict):
    class FakeCfg:
        _data = payload

        @staticmethod
        def load():
            return FakeCfg()

        def get(self, k, d=None):
            return self._data.get(k, d)

    return FakeCfg


def test_date_decomposition_emits_expected_columns():
    train, test = _make_frames()
    fe_cfg = {
        "date_decomposition": [
            {
                "column": "event_date",
                "parts": ["month", "day_of_year", "sin_doy", "cos_doy"],
            }
        ]
    }
    tr_out, te_out = build_hypothesis_features(
        train, test, mode="inference", merged_fe_cfg=fe_cfg
    )

    for part in ("month", "day_of_year", "sin_doy", "cos_doy"):
        assert f"event_date_{part}" in tr_out.columns
        assert f"event_date_{part}" in te_out.columns

    # Calendar parts are exact
    assert tr_out["event_date_month"].tolist() == [1.0, 3.0, 7.0, 12.0]
    assert tr_out["event_date_day_of_year"].tolist() == [15.0, 80.0, 186.0, 366.0]

    # Cyclical parts bounded to [-1, 1] on both frames
    for df in (tr_out, te_out):
        for part in ("sin_doy", "cos_doy"):
            assert df[f"event_date_{part}"].abs().max() <= 1.0 + 1e-9


def test_date_decomposition_drop_source_removes_raw_column():
    train, test = _make_frames()
    fe_cfg = {
        "date_decomposition": [
            {
                "column": "event_date",
                "parts": ["month", "sin_doy", "cos_doy"],
                "drop_source": True,
            }
        ]
    }
    tr_out, te_out = build_hypothesis_features(
        train, test, mode="inference", merged_fe_cfg=fe_cfg
    )
    assert "event_date" not in tr_out.columns
    assert "event_date" not in te_out.columns
    # Derived columns survive
    assert "event_date_sin_doy" in tr_out.columns
    # Non-dropped columns survive
    assert "age" in tr_out.columns


def test_drop_columns_generic_block():
    train, test = _make_frames()
    fe_cfg = {"drop_columns": ["unused"]}
    tr_out, te_out = build_hypothesis_features(
        train, test, mode="inference", merged_fe_cfg=fe_cfg
    )
    assert "unused" not in tr_out.columns
    assert "unused" not in te_out.columns
    assert "age" in tr_out.columns


def test_drop_columns_protected_target_raises(monkeypatch):
    train, test = _make_frames()
    train["target"] = [0, 1, 0, 1]
    test = test.copy()

    payload = {"target_col": "target"}
    monkeypatch.setattr(
        "zindian.config.ChallengeConfig.load",
        staticmethod(_fake_config_cls(payload).load),
    )

    fe_cfg = {"drop_columns": ["target"]}
    with pytest.raises(ValueError, match="protected columns"):
        build_hypothesis_features(
            train, test, mode="inference", merged_fe_cfg=fe_cfg
        )


def test_date_decomposition_on_target_raises(monkeypatch):
    train, test = _make_frames()
    train["target"] = [0, 1, 0, 1]

    payload = {"target_col": "target"}
    monkeypatch.setattr(
        "zindian.config.ChallengeConfig.load",
        staticmethod(_fake_config_cls(payload).load),
    )

    fe_cfg = {
        "date_decomposition": [
            {"column": "target", "parts": ["month"], "drop_source": True}
        ]
    }
    with pytest.raises(ValueError, match="date_decomposition"):
        build_hypothesis_features(
            train, test, mode="inference", merged_fe_cfg=fe_cfg
        )


def test_date_decomposition_coerce_handles_bad_dates():
    train, test = _make_frames()
    train.loc[0, "event_date"] = "not-a-date"
    fe_cfg = {
        "date_decomposition": [
            {
                "column": "event_date",
                "parts": ["month", "sin_doy"],
                "error_policy": "coerce",
            }
        ]
    }
    tr_out, te_out = build_hypothesis_features(
        train, test, mode="inference", merged_fe_cfg=fe_cfg
    )
    assert np.isnan(tr_out["event_date_month"].iloc[0])
    assert np.isnan(tr_out["event_date_sin_doy"].iloc[0])
    # Other rows parse normally
    assert tr_out["event_date_month"].iloc[1] == 3.0


def test_seasonal_deathdate_sidecar_shape_matches_convention():
    """The V1 sidecar's derived column names must match the engine's
    `{col}_{part}` output convention exactly."""
    import json
    from pathlib import Path

    sidecar = json.loads(
        (
            Path(__file__).resolve().parent.parent
            / "competitions"
            / "climate-risk-health-prediction-challenge"
            / "variants"
            / "seasonal-deathdate.json"
        ).read_text()
    )
    fe = sidecar["feature_engineering"]
    dd = fe["date_decomposition"][0]
    col = dd["column"]

    engine_expected = {f"{col}_{p}" for p in dd["parts"]}
    declared = set(sidecar["feature_columns"])
    derived_declared = {c for c in declared if c.startswith(f"{col}_")}

    assert derived_declared == engine_expected
    if dd.get("drop_source", False):
        assert col not in declared, "raw ordinal must be excluded from the whitelist when drop_source=true"
        assert col in fe.get("drop_columns", [])
    else:
        assert col in declared, "raw ordinal must be included in the whitelist when drop_source=false"