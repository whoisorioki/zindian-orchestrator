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