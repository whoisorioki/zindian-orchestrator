"""
Skill 05 — CV Architect
========================
Builds the cross-validation strategy for the active competition.
Writes the chosen CV strategy to challenge_config.json during Phase 1 only,
and records the selection in SKILL_STATE.json for downstream skills.

Strategy selection is deterministic and based on dataset properties;
no empirical strategy comparison loop is performed.

Usage:
    python -m zindian.skills.skill_05_cv
    python -m zindian.skills.skill_05_cv --strategy=spatial
    python -m zindian.skills.skill_05_cv --strategy=stratified
"""

from __future__ import annotations
import tabula.skill_state_autopatch  # noqa

import json
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold, StratifiedKFold

from zindian.config import ChallengeConfig, get_seed
from zindian.paths import resolve_competition_paths
from zindian.state import SkillStateStore

N_SPLITS = 5
SPATIAL_CLUSTER_MULTIPLIER = 3


# -- CV Strategy Builders -------------------------------------------------------


def _config_data(config: ChallengeConfig) -> dict[str, Any]:
    return getattr(config, "_data", {}) or {}


def _resolve_target_col(config: ChallengeConfig) -> str:
    target_col = config.get("target_col") or config.get("target_column")
    if not target_col:
        raise RuntimeError("target_col not initialized in challenge_config.json")
    return str(target_col)


def _policy_filtered_columns(config: ChallengeConfig) -> set[str]:
    blocked = config.get("policy_filters", []) or []
    if not isinstance(blocked, (list, tuple, set)):
        return set()
    return {str(col) for col in blocked if col is not None}


def _build_sphere_projection(coords: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(coords[:, 0].astype(np.float64))
    lon = np.deg2rad(coords[:, 1].astype(np.float64))
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.column_stack([x, y, z]).astype(np.float64)


def build_stratified_splits(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = N_SPLITS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Strategy: StratifiedKFold for class imbalance stability."""
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=get_seed())
    return list(splitter.split(X, y))


def build_spatial_splits(
    X: np.ndarray,
    y: np.ndarray,
    coords: np.ndarray,
    n_splits: int = N_SPLITS,
    n_clusters: int | None = None,
    task_type: str = "classification",
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """
    Strategy B — Spatial Block CV.
    Clusters observations into geographic blocks using KMeans on Lat/Lon.
    GroupKFold holds out one block at a time — no geographic leakage.

    Args:
        X:          Feature matrix
        y:          Labels
        coords:     (n, 2) array of [Latitude, Longitude]
        n_clusters: Number of geographic blocks (default = 3 × n_splits, capped by sample count)
        task_type:  'classification' or 'regression'

    Returns:
        (splits, geo_groups) — splits for CV, geo_groups for inspection
    """
    cluster_count = n_clusters or max(
        n_splits * SPATIAL_CLUSTER_MULTIPLIER, n_splits + 1
    )
    cluster_count = min(cluster_count, len(coords))
    if cluster_count < n_splits:
        raise RuntimeError("Not enough spatial samples to build stable fold groups")

    projected = _build_sphere_projection(coords)
    kmeans = KMeans(n_clusters=cluster_count, random_state=get_seed(), n_init=10)
    geo_groups = kmeans.fit_predict(projected)

    print("\n  Geographic block distribution:")
    # Check if target is binary/categorical (for classification prevalence logging)
    is_binary = (
        task_type == "classification"
        and np.issubdtype(y.dtype, np.integer)
        and bool(set(np.unique(y)) <= {0, 1})
    )
    for block_id in range(cluster_count):
        block_mask = geo_groups == block_id
        block_total = int(block_mask.sum())
        if block_total:
            if is_binary:
                block_pos = int(y[block_mask].sum())
                prevalence = block_pos / block_total * 100.0
                print(
                    f"    Block {block_id}: {block_total:4d} samples  "
                    f"({block_pos:3d} positive, "
                    f"{prevalence:.1f}% prevalence)"
                )
            else:
                block_mean = float(y[block_mask].mean())
                print(
                    f"    Block {block_id}: {block_total:4d} samples  "
                    f"(target mean: {block_mean:.4f})"
                )
        else:
            print(f"    Block {block_id}:    0 samples")

    # Construct a GroupKFold via the central CV factory and split using geo_groups
    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(X, y, groups=geo_groups))

    return splits, geo_groups


def _resolve_decision(
    config: ChallengeConfig, state: dict[str, Any], ft: pd.DataFrame
) -> dict[str, Any]:
    eda = state.get("eda", {}) or {}
    raw_config = _config_data(config)

    temporal_confirmed = bool(eda.get("temporal_index_confirmed", False))
    group_confirmed = bool(eda.get("group_structure_confirmed", False))
    spatial_signal = bool(
        (raw_config.get("spatial_signal", {}) or {}).get("present", False)
    )
    group_signal = bool(
        (raw_config.get("group_signal", {}) or {}).get("present", False)
    )
    task_type = str(
        raw_config.get("task_type", config.get("task_type", "classification"))
    )
    minority_ratio = raw_config.get("minority_ratio", eda.get("minority_ratio"))

    if temporal_confirmed:
        return {
            "type": "TimeSeriesSplit",
            "shuffle": False,
            "n_splits": int(
                raw_config.get("cv_strategy", {}).get("n_splits", N_SPLITS)
            ),
            "selection_reason": "temporal_index_confirmed",
        }

    if group_confirmed or spatial_signal or group_signal:
        group_col = None
        if group_signal:
            group_col = (raw_config.get("group_signal", {}) or {}).get("col")
        if group_col is None and spatial_signal:
            group_col = (raw_config.get("spatial_signal", {}) or {}).get("group_col")
        if group_col is None:
            group_col = eda.get("group_col")
        # If a group column is available, use GroupKFold. If not, return a
        # GroupKFold decision with no group_col so the caller can attempt a
        # spatial clustering fallback (using latitude/longitude) before
        # falling back to safer strategies.
        if group_col is None:
            return {
                "type": "GroupKFold",
                "shuffle": False,
                "n_splits": int(
                    raw_config.get("cv_strategy", {}).get("n_splits", N_SPLITS)
                ),
                "group_col": None,
                "selection_reason": "group_structure_requested_but_group_col_missing",
            }
        if str(group_col) not in ft.columns:
            raise RuntimeError(
                f"group_col '{group_col}' not present in features_train.csv"
            )
        return {
            "type": "GroupKFold",
            "shuffle": False,
            "n_splits": int(
                raw_config.get("cv_strategy", {}).get("n_splits", N_SPLITS)
            ),
            "group_col": str(group_col),
            "selection_reason": (
                "group_structure_confirmed"
                if group_confirmed
                else (
                    "spatial_signal.present"
                    if spatial_signal
                    else "group_signal.present"
                )
            ),
        }

    if (
        task_type == "classification"
        and minority_ratio is not None
        and float(minority_ratio) < 0.15
    ):
        return {
            "type": "StratifiedKFold",
            "shuffle": True,
            "n_splits": int(
                raw_config.get("cv_strategy", {}).get("n_splits", N_SPLITS)
            ),
            "random_state": int(
                raw_config.get("reproducibility", {}).get("seed", get_seed())
            ),
            "selection_reason": f"minority_ratio={float(minority_ratio):.3f} < 0.15",
        }

    reason = (
        "standard regression strategy chosen for continuous target"
        if task_type == "regression"
        else "default balanced classification fallback"
    )
    return {
        "type": "KFold",
        "shuffle": True,
        "n_splits": int(raw_config.get("cv_strategy", {}).get("n_splits", N_SPLITS)),
        "random_state": int(
            raw_config.get("reproducibility", {}).get("seed", get_seed())
        ),
        "selection_reason": reason,
    }


# -- Main -----------------------------------------------------------------------


def run(strategy: str = "compare") -> dict:
    """
    Skill 05 — CV Architect.

    Args:
        strategy: 'compare' (default), 'spatial', or 'stratified'

    Returns:
        dict with chosen strategy, OOF AUC for each, recommendation.
    """
    print(f"\n{'=' * 60}")
    print("SKILL 05 — CV Architect")
    print(f"{'=' * 60}\n")

    paths = resolve_competition_paths(require_competition=True)
    config = ChallengeConfig.load()
    task_type = str(config.get("task_type", "classification"))
    competition_dir = paths.competition_dir
    if competition_dir is None:
        raise RuntimeError("Competition directory could not be resolved")

    print(f"Competition : {config.slug}")
    print(f"Strategy    : {strategy}")

    # -- Load features ------------------------------------------
    ft_path = competition_dir / "data" / "processed" / "features_train.csv"
    if not ft_path.exists():
        train_file_name = config.get("input_files", {}).get("train") or "Train.csv"
        raw_train_path = paths.data_raw_dir / train_file_name
        if raw_train_path.exists():
            ft_path = raw_train_path
            print(
                "  [WARN]  features_train.csv not found. Falling back to raw Train.csv for CV analysis."
            )
        else:
            raise FileNotFoundError(
                f"features_train.csv not found at {ft_path}. "
                "Run Skill 07 feature extraction first."
            )

    ft = pd.read_csv(ft_path)
    print(f"\nFeatures loaded: {ft.shape}")

    target_col = _resolve_target_col(config)
    state_store = SkillStateStore(paths.state_path)
    state = state_store.read()

    # Respect config's cv_strategy when already set (e.g. by skill_02 intake),
    # only fall back to data-driven detection when config is empty.
    config_cv_type = (config.get("cv_strategy") or {}).get("type")
    if config_cv_type and config_cv_type not in ("auto", "compare", None):
        decision = {
            "type": config_cv_type,
            "n_splits": config.get("cv_strategy", {}).get("n_splits", N_SPLITS),
            "shuffle": config.get("cv_strategy", {}).get("shuffle", True),
            "random_state": config.get("cv_strategy", {}).get("random_state", get_seed()),
            "selection_reason": "configured_in_challenge_config",
        }
    else:
        decision = _resolve_decision(config, state, ft)

    forced_strategy = (
        strategy
        if strategy in ("spatial", "stratified", "timeseries", "kfold")
        else None
    )
    selected_type = forced_strategy or decision["type"]
    selection_reason = (
        decision["selection_reason"]
        if forced_strategy is None
        else f"forced_{forced_strategy}"
    )

    policy_blocked = _policy_filtered_columns(config)
    coord_names = {
        str(config.get("latitude_column", "Latitude")),
        str(config.get("longitude_column", "Longitude")),
    }
    group_col = decision.get("group_col")
    id_col = config.get("id_col") or config.get("id_column") or "ID"
    excluded_cols = {id_col, target_col, *coord_names, *policy_blocked}
    if group_col is not None:
        excluded_cols.add(str(group_col))

    feature_cols = [c for c in ft.columns if c not in excluded_cols]
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(ft[c])]
    coord_cols = [c for c in coord_names if c in ft.columns]

    # Use explicit np.asarray to provide concrete ndarray types for static checkers
    X = np.asarray(ft[feature_cols].values, dtype=np.float32)
    if target_col in ft.columns:
        target_series = ft[target_col]
    else:
        # Load from raw train
        train_file_name = config.get("input_files", {}).get("train") or "Train.csv"
        raw_train_path = paths.data_raw_dir / train_file_name
        if raw_train_path.exists():
            raw_train = pd.read_csv(raw_train_path)
            if target_col in raw_train.columns:
                target_series = raw_train[target_col]
            else:
                raise KeyError(
                    f"Target column '{target_col}' not found in raw train or features."
                )
        else:
            raise FileNotFoundError(f"Raw train file not found at {raw_train_path}")

    # Factorize if categorical (string/object)
    if not pd.api.types.is_numeric_dtype(target_series):
        target_series = pd.Series(
            pd.factorize(target_series)[0], index=target_series.index
        )

    y_dtype = np.float32 if task_type == "regression" else np.int32
    y = np.asarray(target_series.values, dtype=y_dtype)
    coords = (
        np.asarray(ft[coord_cols].values, dtype=np.float64)
        if len(coord_cols) == 2
        else None
    )

    print(f"Features     : {len(feature_cols)}")
    print(f"Samples      : {len(y)}")
    if len(y) > 0:
        if task_type == "regression":
            print(
                f"Target mean  : {float(np.asarray(y, dtype=np.float64).mean()):.4f} "
                f"(range: {float(y.min()):.4f} to {float(y.max()):.4f})"
            )
        else:
            print(
                f"Positive rate: {float(np.asarray(y, dtype=np.float64).mean()) * 100:.1f}%"
            )

    # If GroupKFold was selected but no explicit group_col is available,
    # attempt a spatial clustering fallback when coordinates exist. If that
    # fails (too few points or clustering error), gracefully fall back to
    # StratifiedKFold (classification with imbalance) or KFold otherwise.
    minority_ratio = config.get("minority_ratio") or (state.get("eda") or {}).get(
        "minority_ratio"
    )

    if selected_type == "GroupKFold" and decision.get("group_col") is None:
        print(
            "\nGroupKFold requested but no group_col supplied — attempting spatial clustering fallback"
        )
        if coords is not None and len(coords) >= N_SPLITS:
            try:
                _splits, geo_groups = build_spatial_splits(
                    X,
                    y,
                    coords,
                    n_splits=int(decision.get("n_splits", N_SPLITS)),
                    task_type=task_type,
                )
                # If clustering succeeded, mark group_col as generated and persist small artifact
                selected_type = "GroupKFold"
                selection_reason = selection_reason + "; spatial_clusters_generated"
                # signal generated cluster group in state; concrete group_col name is an implementation detail
                state_store.update(
                    spatial_cluster_generated=True,
                    spatial_cluster_count=int(max(1, len(set(geo_groups)))),
                )
                print(
                    "  [OK] spatial clusters generated; using GroupKFold on cluster groups"
                )
            except Exception as exc:
                print(
                    f"  [WARN]  Spatial clustering failed or insufficient samples: {exc} — falling back to safer CV"
                )
                if (
                    config.get("task_type") == "classification"
                    and minority_ratio is not None
                    and float(minority_ratio) < 0.15
                ):
                    selected_type = "StratifiedKFold"
                    selection_reason = (
                        selection_reason
                        + "; fallback_to_stratified_due_to_sparse_spatial"
                    )
                else:
                    selected_type = "KFold"
                    selection_reason = (
                        selection_reason + "; fallback_to_kfold_due_to_sparse_spatial"
                    )
        else:
            print(
                "  [WARN]  No coordinate columns available or too few rows for spatial clustering — falling back to safer CV"
            )
            if (
                config.get("task_type") == "classification"
                and minority_ratio is not None
                and float(minority_ratio) < 0.15
            ):
                selected_type = "StratifiedKFold"
                selection_reason = (
                    selection_reason + "; fallback_to_stratified_no_coords"
                )
            else:
                selected_type = "KFold"
                selection_reason = selection_reason + "; fallback_to_kfold_no_coords"

    print(f"\nSelected CV strategy: {selected_type}")
    print(f"Selection reason     : {selection_reason}")

    state_update = {
        "cv_strategy": {
            "type": selected_type,
            "n_splits": int(decision.get("n_splits", N_SPLITS)),
            "shuffle": bool(decision.get("shuffle", False)),
            "random_state": decision.get("random_state"),
            "group_col": decision.get("group_col"),
            "selection_reason": selection_reason,
        },
        "cv_strategy_type": selected_type,
        "cv_strategy_selection_reason": selection_reason,
        "cv_group_col": decision.get("group_col"),
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }
    current_phase = state.get("dag_phase")
    if current_phase in (
        None,
        "uninitialized",
        "phase_0_foundation",
        "phase_1_complete",
        "phase_2_legality_checked",
    ):
        state_update["dag_phase"] = "phase_3_features"
    state_store.update(**state_update)
    print(f"\n[OK] SKILL_STATE.json updated: cv_strategy={selected_type}")

    # -- Per SoT: persist chosen cv_strategy into challenge_config.json during Phase 1 only
    try:
        allowed_write_phases = (
            None,
            "uninitialized",
            "phase_0_foundation",
            "phase_1",
            "phase_1_integrity",
        )
        if current_phase in allowed_write_phases:
            cfg_path = paths.config_path
            if cfg_path.exists():
                cfg_data = json.loads(cfg_path.read_text(encoding="utf-8"))
            else:
                cfg_data = {}

            cv_block = {
                "type": selected_type,
                "n_splits": int(decision.get("n_splits", N_SPLITS)),
                "shuffle": bool(decision.get("shuffle", False)),
                "random_state": decision.get("random_state"),
                "group_col": decision.get("group_col"),
                "selection_reason": selection_reason,
            }

            if cfg_data.get("cv_strategy") != cv_block:
                cfg_data["cv_strategy"] = cv_block
                cfg_path.write_text(
                    json.dumps(cfg_data, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                print(
                    f"[OK] challenge_config.json updated with cv_strategy: {cv_block}"
                )
        else:
            print(
                f"[INFO]  Skipping challenge_config.json write — current phase '{current_phase}' prohibits config mutation."
            )
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[WARN]  Failed to write cv_strategy to challenge_config.json: {exc}")

    return {
        "status": "OK",
        "strategy_chosen": selected_type,
        "selection_reason": selection_reason,
        "cv_strategy": state_update["cv_strategy"],
    }


if __name__ == "__main__":
    strategy = "compare"
    for arg in sys.argv[1:]:
        if arg.startswith("--strategy="):
            strategy = arg.split("=", 1)[1]
        elif arg == "--strategy" and len(sys.argv) > sys.argv.index(arg) + 1:
            strategy = sys.argv[sys.argv.index(arg) + 1]

    if strategy not in ("compare", "spatial", "stratified", "timeseries", "kfold"):
        print(
            f"[FAIL] Unknown strategy '{strategy}'. "
            f"Use: compare, spatial, stratified, timeseries, kfold"
        )
        sys.exit(1)

    result = run(strategy=strategy)
    printable = {k: v for k, v in result.items() if k != "results"}
    print(json.dumps(printable, indent=2))
