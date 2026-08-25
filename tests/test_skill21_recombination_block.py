"""[v2.7 / T1] Integration test: strict A12 block policy in skill_21.

Drives _run_multi_target_pseudo_label with a monkeypatched per-target `run`
so that one classification target reports augmentation failure. The strict
block policy must:
  - return retraining_required == False, and
  - NOT promote the _augmented namespace (no update key ends in _oof_augmented).
This is Step 7 of the v2.7 flow; Step 8 re-runs gate tests through this policy.
"""

from unittest.mock import MagicMock

import zindian.skills.skill_21_pseudo_label as s21
from zindian.skills.skill_21_pseudo_label import _run_multi_target_pseudo_label


def _fake_per_target_run(results_by_target):
    """Fake module-level run() that dispatches on target_name_override."""

    def _fake(dry_run=False, target_name_override=None, is_multi_target=False):
        res = results_by_target[target_name_override]
        if res.get("failed"):
            return {"status": "BLOCKED", "reason": "guard_conditions_failed"}
        return {
            "status": "OK",
            "best_iteration": 3,
            "best_oof_f1": 0.8,
            "retraining_required": True,
            "guard_condition_flags": {},
        }

    return _fake


def _make_config(names):
    return {
        "target_config": {
            "targets": [
                {"name": name, "task_type": "classification",
                 "weight": 1.0 / len(names)}
                for name in names
            ],
            "pseudo_label_recombination_policy": (
                "block_composite_until_all_targets_augmented_or_none"
            ),
        }
    }


def _drive(monkeypatch, names, fail_first):
    """Run _run_multi_target_pseudo_label with one classification target failing."""
    results = {}
    for i, name in enumerate(names):
        results[name] = {"failed": fail_first and i == 0}
    monkeypatch.setattr(s21, "run", _fake_per_target_run(results))

    store = MagicMock()
    paths = MagicMock()
    state = {}
    result = _run_multi_target_pseudo_label(
        paths, _make_config(names), store, state, dry_run=False
    )
    return result, store


def test_block_policy_blocks_partial_classification_augmentation(monkeypatch):
    """One classification target failing to augment must block the composite
    even though no regression target exists — and must NOT promote _augmented."""
    result, store = _drive(monkeypatch, ["targetA", "targetB"], fail_first=True)
    assert result["retraining_required"] is False

    all_kwargs = [call.kwargs for call in store.update.call_args_list]

    # The canonical pseudo_label_result must reflect the block.
    pr_block = None
    for kw in all_kwargs:
        pr = kw.get("pseudo_label_result")
        if isinstance(pr, dict) and "retraining_required" in pr:
            pr_block = pr
    assert pr_block is not None, "expected a pseudo_label_result update"
    assert pr_block["retraining_required"] is False

    # The _augmented namespace must NOT be promoted.
    for kw in all_kwargs:
        for key in kw:
            assert not key.endswith("_oof_augmented"), (
                f"augmented key must not be promoted: {key}"
            )


def test_block_policy_allows_when_all_classification_targets_augment(monkeypatch):
    """When every classification target augments successfully, the block policy
    passes the composite through with retraining_required == True."""
    result, _ = _drive(monkeypatch, ["targetA", "targetB"], fail_first=False)
    assert result["retraining_required"] is True


def _extract_pseudo_label_result(store):
    """Pull the canonical pseudo_label_result dict out of the store updates."""
    pr = None
    for call in store.update.call_args_list:
        candidate = call.kwargs.get("pseudo_label_result")
        if isinstance(candidate, dict) and "retraining_required" in candidate:
            pr = candidate
    return pr


def test_end_to_end_real_policy_flag_gates_augmented_baseline(monkeypatch):
    """Step 8: the retraining_required=True reaching the gate must come from the
    REAL skill_21 block-policy output (every target augmented), not a hand-set
    boolean — and the multi-target gate must then consume
    anchor_oof_score_augmented rather than the original anchor."""
    from zindian.skills.skill_11_gate import _run_multi_target_gate

    # -- Stage 1: real block policy, all classification targets augment -------
    result, store = _drive(monkeypatch, ["targetA", "targetB"], fail_first=False)
    assert result["retraining_required"] is True
    pr = _extract_pseudo_label_result(store)
    assert pr is not None and pr["retraining_required"] is True

    # -- Stage 2: feed that real flag into the multi-target gate -------------
    gate_config = {
        "task_type": "classification",
        "metric": "composite",
        "variance_gate_threshold": 0.01,
        "gate_margin": 0.001,
        "use_inverse_variance_weighting": False,
        "target_config": {
            "targets": [
                {"name": "targetA", "task_type": "classification", "weight": 0.5},
                {"name": "targetB", "task_type": "classification", "weight": 0.5},
            ]
        },
    }
    gate_state = {
        "best_variant_this_round": "variant-a",
        "feature_round": 1,
        "anchor_multi_target_metrics": {
            "targetA": {"oof_f1": 0.9},
            "targetB": {"oof_f1": 0.9},
        },
        "shap_multi_target_results": {
            "targetA": {"pruning_pass": True},
            "targetB": {"pruning_pass": True},
        },
        "human_gate_2_variant-a_approved": True,
        "anchor_oof_score": 0.05,           # unfavourable original anchor
        "anchor_oof_score_augmented": 0.20,  # favourable augmented baseline
        "eda": {},
        "metric_analysis": {"composite_fold_score_variance": 0.001},
        "pseudo_label_result": pr,  # <-- real skill_21 output, not hand-set
    }

    mock_store = MagicMock()
    gate_result = _run_multi_target_gate(gate_config, mock_store, gate_state)
    assert gate_result["status"] == "PASS"
    assert gate_result["diagnosis"]["baseline_key"] == "anchor_oof_score_augmented"

    # Control: without the policy flag the gate must fall back to the original
    # (unfavourable) anchor and BLOCK — proving the augmented path really is
    # gated by skill_21's output.
    control_state = dict(gate_state)
    control_state["pseudo_label_result"] = {"ran": True, "retraining_required": False}
    control_result = _run_multi_target_gate(
        gate_config, MagicMock(), control_state
    )
    assert control_result["status"] == "BLOCKED"
    assert control_result["reason"] == "baseline gate failed"
    assert control_result["diagnosis"]["baseline_key"] == "anchor_oof_score"