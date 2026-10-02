from pathlib import Path

from zindian.state import SkillStateStore, write_oof_record


def sample_oof():
    return {
        "scores": [0.1, 0.9, 0.2],
        "cv_strategy_id": "stratified",
        "seed": 42,
        "branch_name": "anchor",
        "model_config": {"num_leaves": 31},
    }


def test_oof_schema_keys_present():
    oof = sample_oof()
    required = {"scores", "cv_strategy_id", "seed", "branch_name", "model_config"}
    assert required.issubset(set(oof.keys()))


def test_oof_scores_type_and_length():
    oof = sample_oof()
    assert isinstance(oof["scores"], list)
    assert len(oof["scores"]) > 0
    assert all(isinstance(x, (float, int)) for x in oof["scores"])


def test_write_oof_record_uses_canonical_schema(tmp_path: Path):
    store = SkillStateStore(tmp_path / "SKILL_STATE.json")
    record = write_oof_record(
        store,
        branch_name="anchor-baseline",
        scores=[0.1, 0.2, 0.3],
        cv_strategy_id="config:stratified",
        seed=42,
        model_config={"num_leaves": 31},
    )

    assert record == {
        "scores": [0.1, 0.2, 0.3],
        "cv_strategy_id": "config:stratified",
        "seed": 42,
        "branch_name": "anchor-baseline",
        "model_config": {"num_leaves": 31},
        # SoT S-4 row-identity fields. This call site omits id_order, so the
        # record is explicitly marked UNVERIFIED rather than assumed sound --
        # verify_oof_alignment() rejects such a record, and preflight A9 fails
        # on it. Omission must be visible, never silent.
        "id_order_hash": None,
        "id_count": 3,
        "alignment_verified": False,
    }

    state = store.read()
    assert state["branch_anchor-baseline_oof"] == record
