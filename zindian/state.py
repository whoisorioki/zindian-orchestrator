from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Dict

from .joins import RowAlignmentError
from .schemas import skill_state_skeleton, validate_skill_state


def _iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _assert_not_live_competition(path: Path) -> None:
    """Refuse to write into the repo's live ``competitions/`` tree during tests.

    Every ``SkillStateStore`` write funnels through ``_atomic_write_json``,
    which also re-serialises every large payload into ``scores/*.json``.
    Guarding here covers all call sites at once — patching individual
    call sites proved insufficient because several modules resolve paths via
    a *function-local* import, which rebinds the name and silently ignores
    any monkeypatch applied to ``zindian.paths``.

    On 2026-10-02 the test suite rewrote SKILL_STATE.json, 26 files under
    ``scores/``, six phase summaries, six log files and the DuckDB ledger of
    the real competition, destroying evidence for an open investigation.

    KNOWN LIMITATION — this guard is keyed on ``"pytest" in sys.modules``,
    which does **not** reach a child process. A test that spawns a
    subprocess importing :mod:`zindian.state` runs with no guard, because
    pytest is not in that interpreter's ``sys.modules``. As of 2026-10-02 four
    tests use ``subprocess`` and are therefore unprotected:

        tests/test_gate_option_b.py
        tests/test_regression_pipeline_integration.py
        tests/test_shap_pca_exclusion.py
        tests/test_skill11_gate.py

    They currently run in-process (verified by passing the md5 integrity
    check), so the gap is latent rather than active. Closing it properly means
    an env-var or sentinel-file check that survives ``exec``, which is a
    deliberate follow-up: a subprocess-wide guard must not fire during
    genuine competition runs. ``ZINDIAN_TEST_ALLOW_LIVE`` is confirmed absent
    from ``~/.bashrc``, ``~/.profile`` and ``~/.zshrc``, so the guard is armed.
    """
    if os.environ.get("ZINDIAN_TEST_ALLOW_LIVE"):
        return
    if "pytest" not in sys.modules:
        return
    try:
        resolved = Path(path).resolve()
    except Exception:  # pragma: no cover - defensive
        return
    repo_root = Path(__file__).resolve().parent.parent
    live = (repo_root / "competitions").resolve()
    if resolved == live or live in resolved.parents:
        raise RuntimeError(
            f"TEST SAFETY: refusing to write {resolved} into the LIVE "
            f"competition tree. Tests must use tmp_path or pass explicit "
            f"paths. Set ZINDIAN_TEST_ALLOW_LIVE=1 to override deliberately."
        )


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    _assert_not_live_competition(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Externalize large payloads if this is SKILL_STATE.json
    if "SKILL_STATE.json" in str(path):
        scores_dir = path.parent / "scores"
        scores_dir.mkdir(exist_ok=True)

        _EDA_KEYS = frozenset(
            {
                "band_summary_stats",
                "temporal_trends",
                "target_correlation_per_feature",
                "class_separability_index",
            }
        )

        data_copy = {}
        for key, value in data.items():
            # Externalize large OOF score lists (>100 entries)
            if isinstance(value, dict) and "scores" in value:
                scores = value["scores"]
                if isinstance(scores, list) and len(scores) > 100:
                    score_file = scores_dir / f"{key}.json"
                    with open(score_file, "w") as sf:
                        json.dump(scores, sf)

                    new_value = {k: v for k, v in value.items() if k != "scores"}
                    new_value["scores_file"] = f"scores/{key}.json"
                    new_value["count"] = len(scores)
                    data_copy[key] = new_value
                else:
                    data_copy[key] = value
            # Externalize large EDA metric dicts (>10 keys)
            elif key in _EDA_KEYS and isinstance(value, dict) and len(value) > 10:
                eda_file = scores_dir / f"eda_{key}.json"
                with open(eda_file, "w") as sf:
                    json.dump(value, sf)

                new_value = {"eda_file": f"scores/eda_{key}.json", "count": len(value)}
                data_copy[key] = new_value
            # Externalize large cv_split_indices lists (>0 elements)
            elif (
                key == "cv_split_indices" and isinstance(value, list) and len(value) > 0
            ):
                splits_file = scores_dir / "cv_split_indices.json"
                with open(splits_file, "w") as sf:
                    json.dump(value, sf)

                data_copy[key] = {
                    "cv_splits_file": "scores/cv_split_indices.json",
                    "count": len(value),
                }
            # Externalize any other top-level list (>100 elements) to prevent state bloat
            elif isinstance(value, list) and len(value) > 100:
                list_file = scores_dir / f"{key}.json"
                with open(list_file, "w") as sf:
                    json.dump(value, sf)

                data_copy[key] = {
                    "list_file": f"scores/{key}.json",
                    "count": len(value),
                }
            else:
                data_copy[key] = value
        data = data_copy

    serialized = json.dumps(data, indent=2, sort_keys=False) + "\n"

    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=str(path.parent), encoding="utf-8"
    ) as tmp:
        tmp.write(serialized)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


@dataclass
class SkillStateStore:
    path: Path

    def __post_init__(self) -> None:
        if isinstance(self.path, str):
            self.path = Path(self.path)

    # Class-level lock: serializes all read-modify-write operations across
    # ALL instances in the same process. Required because run_deep_research
    # spawns a daemon thread that creates its own SkillStateStore instance;
    # an instance-level lock would not protect against that cross-instance race.
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def read(self) -> Dict[str, Any]:
        if not self.path.exists():
            state = skill_state_skeleton()
            _atomic_write_json(self.path, state)
            return state
        obj = json.loads(self.path.read_text(encoding="utf-8"))

        # Hydrate externalized scores, EDA metrics, CV split indices, and large lists
        for key, value in obj.items():
            if not isinstance(value, dict):
                continue
            if "scores_file" in value:
                score_path = self.path.parent / value["scores_file"]
                if score_path.exists():
                    with open(score_path, "r") as sf:
                        value["scores"] = json.load(sf)
            elif "eda_file" in value:
                eda_path = self.path.parent / value["eda_file"]
                if eda_path.exists():
                    with open(eda_path, "r") as sf:
                        loaded = json.load(sf)
                        value.update(loaded)
            elif "cv_splits_file" in value:
                splits_path = self.path.parent / value["cv_splits_file"]
                if splits_path.exists():
                    try:
                        with open(splits_path, "r") as sf:
                            obj[key] = json.load(sf)
                    except Exception as e:
                        print(
                            f"[WARN] Corrupted cv_splits_file ({splits_path}): {e}. Falling back to empty list []."
                        )
                        obj[key] = []
                else:
                    obj[key] = []
            elif "list_file" in value:
                list_path = self.path.parent / value["list_file"]
                if list_path.exists():
                    try:
                        with open(list_path, "r") as sf:
                            obj[key] = json.load(sf)
                    except Exception as e:
                        print(
                            f"[WARN] Corrupted list_file ({list_path}): {e}. Falling back to empty list []."
                        )
                        obj[key] = []
                else:
                    obj[key] = []

        return validate_skill_state(obj)

    def write(
        self, new_state: Dict[str, Any], *, touch_timestamp: bool = True
    ) -> Dict[str, Any]:
        state = dict(new_state)
        if touch_timestamp:
            state["last_updated"] = _iso_now()
        validate_skill_state(state)
        _atomic_write_json(self.path, state)
        return state

    def update(self, **patch: Any) -> Dict[str, Any]:
        with self._lock:
            state = self.read()
            state.update(patch)
            return self.write(state)

    def increment(self, key: str, delta: int = 1) -> int:
        """Increment a numeric field and return new value."""
        with self._lock:
            state = self.read()
            if key not in state:
                state[key] = 0
            state[key] = state[key] + delta
            self.write(state)
            return state[key]

    def append_selected(self, submission_id: int) -> None:
        """Append submission to selected_submissions list."""
        with self._lock:
            state = self.read()
            sel = state.get("selected_submissions")
            if not isinstance(sel, list):
                sel = []
            if submission_id not in sel:
                sel.append(submission_id)
            state["selected_submissions"] = sel
            self.write(state)


def resolve_active_cv_strategy_id(state_obj: dict, config_obj: dict) -> str:
    """
    Resolve the active CV strategy identifier according to the Source of Truth rules.

    Priority:
      1. If SKILL_STATE contains an active `cv_strategy_override.active` == True,
         return an 'override:<override_strategy>' identifier.
      2. Else, read `challenge_config.json` cv_strategy block and return
         'config:<type>' identifier.
      3. Fallback to 'unknown'.

    This function returns a short string suitable for tagging OOF artifacts
    and SKILL_STATE entries.
    """
    try:
        override = state_obj.get("cv_strategy_override", {}) or {}
        if override.get("active", False):
            return f"override:{override.get('override_strategy') or 'unknown'}"
    except Exception:
        pass

    try:
        cv = (
            (config_obj or {}).get("cv_strategy")
            if isinstance(config_obj, dict)
            else None
        )
        if isinstance(cv, dict):
            return f"config:{cv.get('type', 'unknown')}"
    except Exception:
        pass

    return "unknown"


def compute_secondary_metrics(
    y_true: Any,
    y_pred: Any,
    *,
    temporal_present: bool = False,
    mae_naive_baseline: float | None = None,
) -> dict[str, Any]:
    """Calculate regression diagnostics on concatenated arrays."""
    from sklearn.metrics import mean_absolute_error, r2_score
    import numpy as np

    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)

    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    r2 = float(r2_score(y_true_arr, y_pred_arr))

    # Guard against division-by-zero for MAPE
    non_zero = y_true_arr != 0
    if np.sum(non_zero) > 0:
        mape: float | None = float(
            np.mean(
                np.abs(
                    (y_true_arr[non_zero] - y_pred_arr[non_zero]) / y_true_arr[non_zero]
                )
            )
        )
    else:
        mape = None  # SOT/user correction: mape is None when all targets are zero

    # S2 - implemented 2026-08-03
    zero_fraction = float(np.mean(y_true_arr == 0))
    metrics: dict[str, Any] = {
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "zero_fraction": zero_fraction,
    }
    if temporal_present:
        baseline = float(mae_naive_baseline or 0.0)
        metrics["mase"] = mae / baseline if baseline > 0.0 else None

    return metrics


def oof_id_order_hash(ids: Any) -> str:
    """sha256 of the comma-joined ID order — the row-identity fingerprint.

    A hash of the *ordered* sequence, not the set. A whole-row permutation
    preserves set membership exactly, so any set-based check (or a length
    check) passes while the artifact is misaligned. This is the only cheap
    fingerprint that detects the defect class (SoT S-4).
    """
    parts = [str(v) for v in list(ids)]
    return hashlib.sha256(",".join(parts).encode("utf-8")).hexdigest()


def write_oof_record(
    store: SkillStateStore,
    *,
    branch_name: str,
    scores: Any,
    cv_strategy_id: str,
    seed: int,
    model_config: dict[str, Any],
    secondary_metrics: dict[str, Any] | None = None,
    id_order: Any = None,
    touch_timestamp: bool = True,
) -> dict[str, Any]:
    """Persist a SoT-shaped OOF record under `branch_{branch_name}_oof`.

    ``id_order`` (SoT S-1/S-4) is the ordered ID sequence corresponding to
    ``scores``. When supplied, an ``id_order_hash`` fingerprint and
    ``id_count`` are persisted so consumers can prove the vector still lines
    up with the current training rows.

    When omitted, the record is written with ``id_order_hash = None`` and
    ``alignment_verified = False`` rather than being rejected. That keeps
    existing call sites working during the migration, but the flag makes the
    gap explicit and machine-checkable: ``verify_oof_alignment`` treats an
    unverified record as a hard failure, and preflight check A9 fails on any
    active branch still missing its hash. Silence was what allowed this
    defect class to go unnoticed, so the default state is "unverified",
    never "assumed fine".
    """
    if isinstance(scores, (list, tuple)):
        score_list = [float(value) for value in scores]
    else:
        score_list = [float(scores)]

    record = {
        "scores": score_list,
        "cv_strategy_id": str(cv_strategy_id),
        "seed": int(seed),
        "branch_name": str(branch_name),
        "model_config": dict(model_config),
    }
    if secondary_metrics is not None:
        record["secondary_metrics"] = secondary_metrics

    if id_order is None:
        record["id_order_hash"] = None
        record["id_count"] = len(score_list)
        record["alignment_verified"] = False
    else:
        record["id_order_hash"] = oof_id_order_hash(id_order)
        record["id_count"] = len(list(id_order))
        record["alignment_verified"] = True
        if record["id_count"] != len(score_list):
            raise RowAlignmentError(
                f"write_oof_record('{branch_name}'): id_order has "
                f"{record['id_count']} entries but scores has "
                f"{len(score_list)}. These describe different row sets, so the "
                f"record would be internally inconsistent (SoT S-4)."
            )

    state = store.read()
    retraining_active = bool(
        (state.get("pseudo_label_result") or {}).get("retraining_required", False)
    )
    if retraining_active:
        if not str(branch_name).endswith("_augmented"):
            raise RuntimeError(
                "Retraining active: OOF records during retraining must use the '_augmented' suffix for branch_name"
            )
        base_branch = str(branch_name).removesuffix("_augmented")
        key = f"branch_{base_branch}_oof_augmented"
        original_key = f"branch_{base_branch}_oof"
        if key == original_key:
            raise RuntimeError(
                f"Retraining loop attempted to overwrite original OOF key '{original_key}'. Write to '{key}' instead."
            )
    else:
        key = f"branch_{branch_name}_oof"

    state[key] = record
    store.write(state, touch_timestamp=touch_timestamp)
    return record


def verify_oof_alignment(
    record: dict[str, Any] | None,
    expected_ids: Any,
    *,
    branch_name: str = "<unknown>",
    context: str = "consumer",
) -> bool:
    """Prove an OOF record still aligns with the current training rows.

    SoT S-4. Recomputes the ordered-ID fingerprint from ``expected_ids`` and
    compares it to the record's persisted ``id_order_hash``.

    A **length check is not a substitute**: a whole-row permutation preserves
    length exactly, so length-based guards pass while the artifact is
    misaligned. This is the check that would have caught the merge_asof
    corruption, and it is a hard failure by design.

    Raises :class:`RowAlignmentError` when the record is missing, carries no
    fingerprint, or the fingerprint disagrees. Returns True on success.
    """
    if record is None:
        raise RowAlignmentError(
            f"[{context}] OOF record for branch '{branch_name}' is missing. "
            f"Cannot prove row alignment (SoT S-4)."
        )

    stored_hash = record.get("id_order_hash")
    if stored_hash is None:
        raise RowAlignmentError(
            f"[{context}] OOF record for branch '{branch_name}' has no "
            f"id_order_hash, so its row order cannot be verified. This record "
            f"predates the S-4 schema or was written without id_order. "
            f"Regenerate the branch. Length alone is NOT sufficient: a "
            f"permutation preserves length."
        )

    expected_hash = oof_id_order_hash(expected_ids)
    if stored_hash != expected_hash:
        raise RowAlignmentError(
            f"[{context}] OOF row order for branch '{branch_name}' does not "
            f"match the current training IDs. The score vector is permuted "
            f"relative to the feature rows, so every positional consumer "
            f"(gating, fusion, pruning) would be silently mis-scored. "
            f"Refusing (SoT S-4)."
        )
    return True


def is_anchor_challenge_active(state_obj: dict) -> bool:
    """Safe accessor for anchor_challenge.active in SKILL_STATE.

    Returns True only if `anchor_challenge` is present and has `active`==True.
    This protects automation from KeyError when the block is absent.
    """
    try:
        return bool((state_obj or {}).get("anchor_challenge", {}).get("active", False))
    except Exception:
        return False


def get_anchor_challenge_config(state_obj: dict) -> dict:
    """Return the `anchor_challenge` config block or empty dict if absent."""
    try:
        return dict((state_obj or {}).get("anchor_challenge") or {})
    except Exception:
        return {}


def write_artifact_fingerprint(
    store: SkillStateStore,
    name: str,
    file_path: Path | str,
    df: Any,
) -> None:
    """Computes and registers a derived artifact fingerprint in SKILL_STATE.

    If the file at `file_path` already exists on disk (before being overwritten),
    loads it to compute the maximum absolute difference against `df` for numeric columns.
    """
    import hashlib
    import pandas as pd
    import numpy as np

    file_path = Path(file_path)

    # Try to load existing file for comparison
    max_abs_diff = 0.0
    if file_path.exists():
        try:
            # We assume it is a CSV format for our fingerprinted matrices
            old_df = pd.read_csv(file_path)
            # Compare numeric columns
            num_cols = df.select_dtypes(include=[np.number]).columns
            old_num_cols = old_df.select_dtypes(include=[np.number]).columns
            common_cols = [c for c in num_cols if c in old_num_cols]

            if len(common_cols) > 0 and len(df) == len(old_df):
                diffs = []
                for col in common_cols:
                    v_new = df[col].fillna(0.0).values
                    v_old = old_df[col].fillna(0.0).values
                    abs_diff = np.abs(v_new - v_old)
                    diffs.append(np.max(abs_diff))
                if diffs:
                    max_abs_diff = float(np.max(diffs))
        except Exception:
            # Silence comparison failures (e.g. format mismatch, empty file, etc.)
            pass

    # Compute MD5 hash of CSV content in memory
    try:
        csv_str = df.to_csv(index=False)
        md5_hash = hashlib.md5(csv_str.encode("utf-8")).hexdigest()
    except Exception:
        md5_hash = ""

    # Update state
    state = store.read()
    fingerprints = state.get("derived_artifact_fingerprints", {})
    if not isinstance(fingerprints, dict):
        fingerprints = {}

    fingerprints[name] = {
        "file_path": str(file_path),
        "md5_hash": md5_hash,
        "row_count": len(df),
        "col_count": len(df.columns),
        "max_abs_diff": max_abs_diff,
    }
    state["derived_artifact_fingerprints"] = fingerprints
    store.write(state)
