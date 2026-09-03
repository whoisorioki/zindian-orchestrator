"""Tests for Audit Remediation & Compliance Verification.

Verifies:
1. DuckDB Ledger schema migration, timestamp format, and unlinked-row upsert
   (match-by-public_score fallback, no row duplication).
2. Board-to-Ledger persistence chain in skill_16_submit, including the
   reports/audits/final_selections.json primary source for Gate 5 selections.
3. Composite metric formula via the production helper
   (zindian.metrics.composite_metric) with recorded OOF provenance values.
4. Preflight AST static-scan enforcement for the 0.5 classification
   hard-label threshold lock.
5. Gate 5 zindi_id derivation from the audited selections file
   (scripts/reconcile_ledger.derive_gate5_ids).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from zindian.ledger import Ledger
from zindian.metrics import composite_metric
from zindian.schemas import skill_state_skeleton
from zindian.skills.skill_11_gate import _metric_key
from zindian.skills.skill_16_submit import show_submission_board
from scripts.preflight_enforce import (
    PreflightError,
    scan_classification_threshold_compliance,
)


def test_ledger_schema_migration_and_upsert(tmp_path: Path):
    """Verify ledger columns (lb_f1, lb_auc, zindi_id, submitted_at) and that an
    unlinked historical row is matched by public_score instead of duplicated."""
    db_path = tmp_path / "experiments.db"

    ledger = Ledger(str(db_path))
    exp_id = ledger.log_experiment(
        branch_name="anchor-v1",
        oof_score=0.82,
        metric="composite",
        gate_result="PASS",
    )

    # 1. Log submission without zindi_id (unlinked row)
    sub_id = ledger.log_submission(
        experiment_id=exp_id,
        branch_name="anchor-v1",
        public_score=0.825558,
        comment="unlinked test submission",
    )
    assert sub_id == 1

    # 2. Upsert with zindi_id matching the unlinked row's public_score
    ledger.upsert_submission_by_zindi_id(
        zindi_id="9oXDE1j3",
        public_score=0.825558,
        lb_f1=0.817,
        lb_auc=0.827,
        selected_for_final=True,
        submitted_at="2026-09-01T19:37:02.754000+00:00",
    )

    subs = ledger.get_submissions()
    assert len(subs) == 1, "Upsert must update unlinked row instead of duplicating"
    row = subs[0]
    assert row["zindi_id"] == "9oXDE1j3"
    assert row["lb_f1"] == pytest.approx(0.817)
    assert row["lb_auc"] == pytest.approx(0.827)
    assert row["selected_for_final"] is True
    assert "2026-09-01" in str(row["submitted_at"])

    # 3. A genuinely new submission must be inserted, not merged
    ledger.upsert_submission_by_zindi_id(
        zindi_id="NEWID123",
        public_score=0.111111,
    )
    assert len(ledger.get_submissions()) == 2
    ledger.close()


def test_board_to_ledger_persistence_chain(tmp_path: Path):
    """Verify show_submission_board persists board entries and that the
    reports/audits/final_selections.json PRIMARY source governs
    selected_for_final over the platform 'chosen' flag."""
    state_path = tmp_path / "SKILL_STATE.json"
    config_path = tmp_path / "challenge_config.json"
    reports_dir = tmp_path / "reports"
    audits_dir = reports_dir / "audits"
    audits_dir.mkdir(parents=True, exist_ok=True)

    state = skill_state_skeleton()
    # NOTE: state fallback intentionally contradicts the file — the file must win.
    state["human_gate_5_selection"] = ["sub_999_not_locked.csv"]
    state_path.write_text(json.dumps(state))

    config_path.write_text(json.dumps({"slug": "test-challenge", "metric": "composite"}))

    # Audited selections file (skill_17 schema) in its real location
    (audits_dir / "final_selections.json").write_text(
        json.dumps(
            {
                "slug": "test-challenge",
                "selections": [
                    {"filename": "sub_005_ensemble.csv", "score": 0.825558}
                ],
            }
        )
    )

    # Manifest metadata
    (reports_dir / "submissions_manifest.json").write_text(
        json.dumps(
            [
                {
                    "zindi_id": "9oXDE1j3",
                    "filename": "sub_005_ensemble.csv",
                    "public_score": 0.825558,
                    "lb_f1": 0.817195,
                    "lb_auc": 0.827310,
                    "created_at": "2026-09-01T19:37:02+00:00",
                }
            ]
        )
    )

    mock_user = MagicMock()
    mock_user.submission_board.return_value = [
        {
            "id": "9oXDE1j3",
            "filename": "sub_005_ensemble.csv",
            "public_score": 0.825558,
            "chosen": False,  # Platform false, but Gate 5 file says locked
            "comment": "sub_005",
            "created_at": "2026-09-01T19:37:02+00:00",
        }
    ]

    with patch("zindian.zindi_client.ZindiClient") as mock_client_cls, patch(
        "zindian.paths.resolve_competition_paths"
    ) as mock_paths1, patch(
        "zindian.skills.skill_16_submit.resolve_competition_paths"
    ) as mock_paths2:
        mock_client_cls.return_value._user = mock_user
        for mp in (mock_paths1, mock_paths2):
            mp.return_value.state_path = state_path
            mp.return_value.config_path = config_path
            mp.return_value.reports_dir = reports_dir

        db_path = reports_dir / "experiments.db"

        with patch("zindian.ledger.resolve_competition_paths") as mock_paths_ledger:
            mock_paths_ledger.return_value.reports_dir = reports_dir
            show_submission_board()

            ledger = Ledger(str(db_path))
            subs = ledger.get_submissions()
            assert len(subs) == 1
            assert subs[0]["zindi_id"] == "9oXDE1j3"
            assert subs[0]["lb_f1"] == pytest.approx(0.817195)
            assert (
                subs[0]["selected_for_final"] is True
            ), "audits/final_selections.json must govern selected_for_final"
            ledger.close()


def test_composite_metric_resolution():
    """Verify metric key mapping and the composite formula via the production
    helper, using the recorded OOF provenance values (experiments.exp6)."""
    assert _metric_key({"metric": "multi"}) == "composite"
    assert _metric_key({"metric": "composite"}) == "composite"
    assert _metric_key({"metric": "zindi"}) == "composite"

    # Recorded OOF anchor values (experiments.exp6.notes) — NOT leaderboard data.
    oof_f1 = 0.8171949630916197
    oof_auc = 0.813342
    expected = 0.6 * oof_f1 + 0.4 * oof_auc

    assert composite_metric(oof_f1, oof_auc) == pytest.approx(expected)
    assert composite_metric(oof_f1, oof_auc) == pytest.approx(0.8156537778549718)
    assert composite_metric(1.0, 1.0) == pytest.approx(1.0)
    assert composite_metric(0.0, 0.0) == pytest.approx(0.0)


def test_composite_metric_lb_provenance_raises():
    """Verify that composite_metric raises ValueError when passed a score with LB provenance."""
    from zindian.metrics import lb_score, oof_score

    # Passing lb_score wrapper must raise ValueError
    with pytest.raises(ValueError, match="prohibits Leaderboard"):
        composite_metric(lb_score(0.817), oof_score(0.827))

    with pytest.raises(ValueError, match="prohibits Leaderboard"):
        composite_metric(oof_score(0.817), lb_score(0.827))

    # Passing explicit f1_origin="lb" or auc_origin="lb" must raise ValueError
    with pytest.raises(ValueError, match="prohibits Leaderboard"):
        composite_metric(0.817, 0.827, f1_origin="lb")

    with pytest.raises(ValueError, match="prohibits Leaderboard"):
        composite_metric(0.817, 0.827, auc_origin="lb")


def test_preflight_classification_threshold_enforcement(tmp_path: Path):
    """Verify the AST scan flags dynamic threshold lookups and non-0.5 fixed
    thresholds, and passes a compliant fixed-0.5 file."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    inf_file = skills_dir / "skill_14_inference.py"

    # Compliant: fixed 0.5 assignment
    inf_file.write_text("threshold = 0.5\nout = probs >= threshold\n")
    scan_classification_threshold_compliance(skills_dir)

    # Non-compliant: dynamic model_config threshold lookup
    inf_file.write_text(
        'thr = model_config.get("threshold")\nout = probs >= thr\n'
    )
    with pytest.raises(PreflightError, match="Rules Compliance Violation"):
        scan_classification_threshold_compliance(skills_dir)

    # Non-compliant: dynamic best_variant_threshold reference
    inf_file.write_text(
        "threshold = state['best_variant_threshold']\nout = probs >= threshold\n"
    )
    with pytest.raises(PreflightError, match="Rules Compliance Violation"):
        scan_classification_threshold_compliance(skills_dir)

    # Non-compliant: fixed but not 0.5
    inf_file.write_text("threshold = 0.4\nout = probs >= threshold\n")
    with pytest.raises(PreflightError, match="Rules Compliance Violation"):
        scan_classification_threshold_compliance(skills_dir)


def test_reconcile_gate5_derivation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Gate 5 zindi_ids must derive from the audited selections file joined
    through the manifest — never from hardcoded IDs."""
    import scripts.reconcile_ledger as rl

    sel_file = tmp_path / "final_selections.json"
    sel_file.write_text(
        json.dumps(
            {
                "selections": [
                    {"filename": "sub_005_ensemble.csv"},
                    {"filename": "sub_008_ensemble.csv"},
                ]
            }
        )
    )
    monkeypatch.setattr(rl, "FINAL_SELECTIONS_PATH", sel_file)

    manifest = [
        {"zindi_id": "9oXDE1j3", "filename": "sub_005_ensemble.csv"},
        {"zindi_id": "PzruUqvQ", "filename": "sub_008_ensemble.csv"},
        {"zindi_id": "EGv7pdTK", "filename": "sub_014_ensemble.csv"},
    ]
    assert rl.derive_gate5_ids(manifest) == {"9oXDE1j3", "PzruUqvQ"}


def test_preflight_oof_lb_collision_guard(tmp_path: Path):
    """The OOF/LB collision guard must fail hard on the exact historical
    contamination (anchor_oof_auc == manifest lb_auc) and pass clean OOF state."""
    from scripts.preflight_enforce import scan_oof_lb_collision

    comp = tmp_path / "comp"
    (comp / "reports").mkdir(parents=True)
    (comp / "reports" / "submissions_manifest.json").write_text(
        json.dumps(
            [
                {
                    "zindi_id": "qouVDWN6",
                    "public_score": 0.821380801,
                    "lb_f1": 0.817427385,
                    "lb_auc": 0.827310924,
                }
            ]
        )
    )

    # Contaminated state — the exact 2026-09-03 failure
    with pytest.raises(PreflightError, match="OOF/LB Collision"):
        scan_oof_lb_collision({"anchor_oof_auc": 0.827310924}, comp)

    # Clean OOF state (the decontaminated values) passes
    scan_oof_lb_collision(
        {
            "anchor_oof_f1": 0.8171949630916197,
            "anchor_oof_auc": 0.813342,
            "anchor_oof_score": 0.8156537778549717,
            "best_variant_oof_auc": 0.8129801203415224,
        },
        comp,
    )

    # No manifest -> check is skipped, not failed
    (comp / "reports" / "submissions_manifest.json").unlink()
    scan_oof_lb_collision({"anchor_oof_auc": 0.5}, comp)
