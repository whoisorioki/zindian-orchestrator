"""
Tests for PCA column exclusion from the SHAP audit frame (F1 fix).

PCA components in the persisted features_train_{variant}.csv are inference-fit
(full-train). Passing them to _compute_shap_audit inflates SHAP importance and
corrupts the pruning_pass gate decision. The fix excludes pca_* columns from the
audit frame in both the single-target and multi-target paths.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch


from zindian.schemas import skill_state_skeleton
from zindian.state import SkillStateStore
import zindian.skills.skill_10_shap as s10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_state(tmp_path: Path) -> SkillStateStore:
    state_path = tmp_path / "SKILL_STATE.json"
    state = skill_state_skeleton()
    state_path.write_text(json.dumps(state), encoding="utf-8")
    return SkillStateStore(state_path)


def _make_frame(n_rows: int = 50, include_pca: bool = True) -> pd.DataFrame:
    """Build a minimal feature frame with optional pca_* columns."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "feature_a": rng.normal(size=n_rows),
            "feature_b": rng.normal(size=n_rows),
            "feature_c": rng.normal(size=n_rows),
            "target": rng.integers(0, 2, size=n_rows).astype(float),
        }
    )
    if include_pca:
        df["pca_1"] = rng.normal(size=n_rows)
        df["pca_2"] = rng.normal(size=n_rows)
    return df


def _fake_paths(tmp_path: Path, store: SkillStateStore):
    class _FakePaths:
        state_path = store.path
        competition_dir = tmp_path
        data_processed_dir = tmp_path / "data" / "processed"
        data_raw_dir = tmp_path / "data" / "raw"
        reports_dir = tmp_path / "reports"

    return _FakePaths()


def _fake_config(extra: dict | None = None):
    cfg = {
        "target_col": "target",
        "task_type": "classification",
        "metric": "auc",
        "metric_direction": "maximize",
        "shap_leak_threshold": 0.8,
        "n_splits": 3,
        "reproducibility": {"seed": 42},
        "cv_strategy": {"type": "stratifiedkfold", "n_splits": 3},
    }
    if extra:
        cfg.update(extra)
    mock = MagicMock()
    mock.get = lambda key, default=None: cfg.get(key, default)
    mock.slug = "test-slug"
    return mock


def _stub_audit_result(feature_cols=None):
    """Minimal valid _compute_shap_audit return value."""
    cols = feature_cols or ["feature_a", "feature_b"]
    return {
        "oof_probs": np.full(50, 0.5),
        "oof_auc": 0.7,
        "oof_f1": 0.5,
        "oof_rmse": None,
        "threshold": 0.5,
        "fold_scores": [0.7] * 3,
        "ranking": pd.DataFrame(
            {
                "feature": cols,
                "mean_abs_shap": [0.1] * len(cols),
            }
        ),
        "top15_share": 0.5,
        "tail_share": 0.1,
        "leaked_features": [],
        "mi_advisory_feature_names": [],
        "leak_check_method": "pearson",
    }


def _stub_cv_result():
    from types import SimpleNamespace

    return SimpleNamespace(
        oof_probs=np.full(50, 0.5),
        oof_auc=0.7,
        oof_f1=0.5,
        oof_rmse=0.1,
        threshold=0.5,
        fold_scores=[0.7] * 3,
    )


# ---------------------------------------------------------------------------
# Unit tests: PCA filtering happens before _compute_shap_audit is called
# ---------------------------------------------------------------------------


class TestPcaExclusionSingleTarget:

    def _run_with_all_mocked(self, tmp_path, frame, monkeypatch):
        """
        Run skill_10.run() with all downstream calls fully mocked.
        Returns (received_feature_cols, final_state).
        """
        store = _write_state(tmp_path)
        received: list[list[str]] = []

        def _capture(frm, feature_cols, target, **kw):
            received.append(list(feature_cols))
            return _stub_audit_result(feature_cols)

        monkeypatch.setattr(
            s10,
            "resolve_competition_paths",
            lambda require_competition=False, **kw: _fake_paths(tmp_path, store),
        )
        monkeypatch.setattr(
            s10,
            "ChallengeConfig",
            MagicMock(load=MagicMock(return_value=_fake_config())),
        )
        monkeypatch.setattr(s10, "_load_train_frame", lambda *a, **kw: frame)
        monkeypatch.setattr(s10, "_compute_shap_audit", _capture)
        monkeypatch.setattr(
            s10,
            "_build_pruned_feature_set",
            lambda *a, **kw: {
                "correlated_pairs": [],
                "drop_features": [],
                "pruned_features": ["feature_a", "feature_b"],
            },
        )
        monkeypatch.setattr(
            s10, "train_lightgbm_cv", lambda *a, **kw: _stub_cv_result()
        )
        monkeypatch.setattr(s10, "write_oof_record", lambda *a, **kw: {})

        s10.run()
        return received, store.read()

    def test_pca_cols_not_in_compute_shap_audit(self, tmp_path, monkeypatch):
        """_compute_shap_audit must not receive pca_* columns."""
        frame = _make_frame(include_pca=True)
        received, _ = self._run_with_all_mocked(tmp_path, frame, monkeypatch)

        assert received, "No _compute_shap_audit call was made"
        for cols in received:
            pca_in_audit = [c for c in cols if c.startswith("pca_")]
            assert (
                not pca_in_audit
            ), f"PCA columns leaked into SHAP audit: {pca_in_audit}"

    def test_shap_pca_cols_excluded_written_to_state(self, tmp_path, monkeypatch):
        """state['shap_pca_cols_excluded'] must list the excluded columns."""
        frame = _make_frame(include_pca=True)
        _, state = self._run_with_all_mocked(tmp_path, frame, monkeypatch)

        excluded = state.get("shap_pca_cols_excluded")
        assert excluded is not None, "shap_pca_cols_excluded was not written to state"
        assert "pca_1" in excluded
        assert "pca_2" in excluded

    def test_no_pca_cols_does_not_set_excluded_key(self, tmp_path, monkeypatch):
        """When no pca_* columns exist, shap_pca_cols_excluded must not be set."""
        frame = _make_frame(include_pca=False)
        _, state = self._run_with_all_mocked(tmp_path, frame, monkeypatch)

        # Key should either be absent or None
        val = state.get("shap_pca_cols_excluded")
        assert not val, f"Expected no pca exclusions, got: {val}"


# ---------------------------------------------------------------------------
# Gate integration: pca_columns_excluded allows Gate 2 to pass
# ---------------------------------------------------------------------------


class TestPcaExcludedGateIntegration:

    def test_pca_excluded_skip_reason_satisfies_shap_condition(
        self, tmp_path, monkeypatch
    ):
        """
        shap_audit_skipped_reason='pca_columns_excluded' + human approval must
        allow Gate 2 to PASS (not block on shap_gate_failed).
        End-to-end test of the F1 → F6 interaction.
        """
        import zindian.skills.skill_11_gate as s11

        state = skill_state_skeleton()
        state.update(
            {
                "best_variant_this_round": "variant_pca",
                "best_variant_oof_score": 0.91,
                "anchor_oof_score": 0.85,
                "variants_passed": 1,
                "shap_completed_at": "2026-01-01T00:00:00+00:00",
                "pruning_pass": False,
                "shap_audit_skipped_reason": "pca_columns_excluded",
                "leaked_features": [],
                "metric_analysis": {"fold_score_variance": 0.001, "se_oof": 0.001},
                "human_gate_2_variant_pca_approved": True,
            }
        )
        state_path = tmp_path / "SKILL_STATE.json"
        state_path.write_text(json.dumps(state), encoding="utf-8")
        store = SkillStateStore(state_path)

        class _FakePaths:
            state_path = store.path
            competition_dir = tmp_path

        fake_config = _fake_config(
            {
                "metric": "auc",
                "task_type": "classification",
                "metric_direction": "maximize",
                "variance_gate_threshold": 0.01,
                "gate_margin": 0.001,
            }
        )

        monkeypatch.setattr(s11, "resolve_competition_paths", lambda **kw: _FakePaths())
        monkeypatch.setattr(
            s11, "ChallengeConfig", MagicMock(load=MagicMock(return_value=fake_config))
        )
        with patch("subprocess.run"):
            result = s11.run()

        assert (
            result["status"] == "PASS"
        ), f"Expected PASS with pca_columns_excluded escape hatch. Got: {result}"
