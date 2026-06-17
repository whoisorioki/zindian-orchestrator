"""
Tests for build_hypothesis_features() in skill_07.

The function is config-driven — it only produces derived columns when the
feature_engineering block in challenge_config.json specifies them.
With no config (or an empty feature_engineering block), it returns the
input frames unchanged. These tests verify:

  1. With a feature_engineering config containing interactions/polynomials/
     conditions, the expected derived columns are produced.
  2. Target-dependent bin features respect the two-mode contract:
       mode="cv"  — requires train_idx; raises if absent
       mode="inference" — uses full training set
  3. With no feature_engineering config, input frames are returned unchanged.
"""

import pandas as pd
import numpy as np
import pytest

from zindian.skills.skill_07_features import build_hypothesis_features

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_frames():
    cols = ["col_a", "col_b", "col_c"]
    n = 4
    train = pd.DataFrame({c: np.linspace(0.1, 1.0, n) for c in cols})
    test = pd.DataFrame({c: np.linspace(1.0, 2.0, n) for c in cols})
    return train, test


def _patch_config(monkeypatch, fe_cfg, target_col="y"):
    """
    Patch ChallengeConfig at source (zindian.config) so the local import
    inside build_hypothesis_features receives the test config.
    The function does:
        from zindian.config import ChallengeConfig
        cfg = ChallengeConfig.load()._data
    Patching zindian.config.ChallengeConfig is the correct intercept point.
    """
    import zindian.config as zconfig

    _data = {"feature_engineering": fe_cfg, "target_col": target_col}

    class _FakeCfg:
        def __init__(self):
            self._data = _data

        def get(self, key, default=None):
            return _data.get(key, default)

    class _FakeChallengeConfig:
        @staticmethod
        def load():
            return _FakeCfg()

    monkeypatch.setattr(zconfig, "ChallengeConfig", _FakeChallengeConfig)


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_build_hypothesis_features_adds_columns(monkeypatch):
    """Config-declared interaction and polynomial columns are added to both frames."""
    train, test = _make_frames()

    fe_cfg = {
        "polynomials": ["col_a"],
        "interactions": [["col_a", "col_b"], ["col_b", "col_c"]],
        "ratios": [["col_a", "col_c"]],
        "conditions": [
            {"column": "col_b", "operator": "lt", "value": 0.5, "name": "col_b_low"}
        ],
        "target_dependent_bins": [],
        "aliases": {},
    }
    _patch_config(monkeypatch, fe_cfg)

    tr, te = build_hypothesis_features(train.copy(), test.copy(), mode="inference")

    expected = [
        "col_a_sq",  # polynomial
        "col_a_x_col_b",  # interaction
        "col_b_x_col_c",  # interaction
        "col_a_div_col_c",  # ratio
        "col_b_low",  # condition
    ]
    for col in expected:
        assert col in tr.columns, f"Missing in train: {col}"
        assert col in te.columns, f"Missing in test: {col}"


def test_target_dependent_feature_modes(monkeypatch):
    """Two-mode contract: mode='cv' requires train_idx; mode='inference' uses full set."""
    train, test = _make_frames()
    target_array = np.array([1.0, 0.0, 1.0, 0.0])

    fe_cfg = {
        "polynomials": [],
        "interactions": [],
        "ratios": [],
        "conditions": [],
        "target_dependent_bins": [
            {"column": "col_a", "q": 2, "name": "col_a_bin_mean"}
        ],
        "aliases": {},
    }
    _patch_config(monkeypatch, fe_cfg)

    # 1. Missing train_idx in mode="cv" raises ValueError
    with pytest.raises(ValueError, match="train_idx must be provided"):
        build_hypothesis_features(
            train, test, mode="cv", target_array=target_array, train_idx=None
        )

    # 2. Correct computation in mode="cv" with train_idx
    train_idx = np.array([0, 1])
    tr_cv, te_cv = build_hypothesis_features(
        train.copy(),
        test.copy(),
        mode="cv",
        target_array=target_array,
        train_idx=train_idx,
    )
    assert "col_a_bin_mean" in tr_cv.columns
    assert "col_a_bin_mean" in te_cv.columns

    # 3. Correct computation in mode="inference"
    tr_inf, te_inf = build_hypothesis_features(
        train.copy(), test.copy(), mode="inference", target_array=target_array
    )
    assert "col_a_bin_mean" in tr_inf.columns
    assert "col_a_bin_mean" in te_inf.columns


def test_no_config_returns_frames_unchanged(monkeypatch):
    """With an empty feature_engineering config, frames come back unmodified."""
    train, test = _make_frames()

    _patch_config(
        monkeypatch,
        {
            "polynomials": [],
            "interactions": [],
            "ratios": [],
            "conditions": [],
            "target_dependent_bins": [],
            "aliases": {},
        },
    )

    tr, te = build_hypothesis_features(train.copy(), test.copy(), mode="inference")
    assert list(tr.columns) == list(train.columns)
    assert list(te.columns) == list(test.columns)
