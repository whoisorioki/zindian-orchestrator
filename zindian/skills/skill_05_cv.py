"""
Skill 05 — CV Architect
========================
Builds and validates the cross-validation strategy for the active competition.
Writes the chosen CV split indices to SKILL_STATE.json for use by all downstream skills.

Two strategies implemented:
  A) StratifiedKFold    — standard, ignores geography (current baseline)
  B) Spatial Block CV   — GroupKFold on KMeans geographic clusters (recommended)

Usage:
  python -m zindian.skills.skill_05_cv                    # compare both, write best
  python -m zindian.skills.skill_05_cv --strategy=spatial # force spatial
  python -m zindian.skills.skill_05_cv --strategy=stratified # force stratified
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

from zindian.config import ChallengeConfig
from zindian.paths import resolve_competition_paths
from zindian.state import SkillStateStore

SEED     = 42
N_SPLITS = 5


# ── CV Strategy Builders ───────────────────────────────────────────────────────

def build_stratified_splits(
    X: np.ndarray,
    y: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Strategy A — StratifiedKFold.
    Preserves class balance per fold.
    Ignores geographic structure — nearby points can appear in both train/val.
    """
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    return list(skf.split(X, y))


def build_spatial_splits(
    X:      np.ndarray,
    y:      np.ndarray,
    coords: np.ndarray,
    n_clusters: int = N_SPLITS,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """
    Strategy B — Spatial Block CV.
    Clusters observations into geographic blocks using KMeans on Lat/Lon.
    GroupKFold holds out one block at a time — no geographic leakage.

    Args:
        X:          Feature matrix
        y:          Labels
        coords:     (n, 2) array of [Latitude, Longitude]
        n_clusters: Number of geographic blocks (default = N_SPLITS)

    Returns:
        (splits, geo_groups) — splits for CV, geo_groups for inspection
    """
    kmeans     = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
    geo_groups = kmeans.fit_predict(coords)

    print(f"\n  Geographic block distribution:")
    for block_id in range(n_clusters):
        block_mask  = geo_groups == block_id
        block_pos   = y[block_mask].sum()
        block_total = block_mask.sum()
        print(f"    Block {block_id}: {block_total:4d} samples  "
              f"({block_pos:3d} positive, "
              f"{block_pos/block_total*100:.1f}% prevalence)")

    gkf    = GroupKFold(n_splits=N_SPLITS)
    splits = list(gkf.split(X, y, groups=geo_groups))

    return splits, geo_groups


# ── CV Evaluator ───────────────────────────────────────────────────────────────

def evaluate_cv_strategy(
    X:        np.ndarray,
    y:        np.ndarray,
    splits:   list[tuple[np.ndarray, np.ndarray]],
    strategy: str,
) -> dict:
    """
    Train a lightweight LightGBM on the given splits.
    Returns OOF AUC — used to compare strategies.
    LGB params are intentionally light (fast eval only, not anchor quality).
    """
    oof_probs = np.zeros(len(y), dtype=np.float64)

    params = {
        "objective":     "binary",
        "metric":        "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves":    31,
        "verbose":       -1,
        "seed":          SEED,
    }

    fold_aucs = []
    for fold_idx, (tr_idx, val_idx) in enumerate(splits):
        train_set = lgb.Dataset(X[tr_idx], label=y[tr_idx])
        val_set   = lgb.Dataset(X[val_idx], label=y[val_idx], reference=train_set)

        model = lgb.train(
            params,
            train_set,
            num_boost_round=300,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(period=-1)],
        )

        oof_probs[val_idx] = model.predict(X[val_idx])
        fold_auc = roc_auc_score(y[val_idx], oof_probs[val_idx])
        fold_aucs.append(fold_auc)
        print(f"    [{strategy}] Fold {fold_idx+1}: AUC={fold_auc:.5f}")

    oof_auc = float(roc_auc_score(y, oof_probs))
    print(f"  [{strategy}] OOF AUC: {oof_auc:.5f}  "
          f"(std={np.std(fold_aucs):.5f})")

    return {
        "strategy":  strategy,
        "oof_auc":   oof_auc,
        "fold_aucs": [float(a) for a in fold_aucs],
        "fold_std":  float(np.std(fold_aucs)),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def run(strategy: str = "compare") -> dict:
    """
    Skill 05 — CV Architect.

    Args:
        strategy: 'compare' (default), 'spatial', or 'stratified'

    Returns:
        dict with chosen strategy, OOF AUC for each, recommendation.
    """
    print(f"\n{'='*60}")
    print(f"SKILL 05 — CV Architect")
    print(f"{'='*60}\n")

    paths  = resolve_competition_paths()
    config = ChallengeConfig.load()

    print(f"Competition : {config.slug}")
    print(f"Strategy    : {strategy}")

    # ── Load features ──────────────────────────────────────────
    ft_path = paths.competition_dir / "data" / "processed" / "features_train.csv"
    if not ft_path.exists():
        raise FileNotFoundError(
            f"features_train.csv not found at {ft_path}. "
            "Run Skill 07 feature extraction first."
        )

    ft = pd.read_csv(ft_path)
    print(f"\nFeatures loaded: {ft.shape}")

    target_col   = "Occurrence Status"
    feature_cols = [c for c in ft.columns
                    if c not in ("ID", target_col)]
    coord_cols   = ["Latitude", "Longitude"]

    X      = ft[feature_cols].values.astype(np.float32)
    y      = ft[target_col].values.astype(np.int32)
    coords = ft[coord_cols].values.astype(np.float64)

    print(f"Features     : {len(feature_cols)}")
    print(f"Samples      : {len(y)}")
    print(f"Positive rate: {y.mean()*100:.1f}%")

    results = {}

    # ── Strategy A — Stratified ────────────────────────────────
    if strategy in ("compare", "stratified"):
        print(f"\n── Strategy A: StratifiedKFold ──")
        strat_splits = build_stratified_splits(X, y)
        results["stratified"] = evaluate_cv_strategy(
            X, y, strat_splits, "stratified"
        )

    # ── Strategy B — Spatial Block ─────────────────────────────
    if strategy in ("compare", "spatial"):
        print(f"\n── Strategy B: Spatial Block CV ──")
        spatial_splits, geo_groups = build_spatial_splits(X, y, coords)
        results["spatial"] = evaluate_cv_strategy(
            X, y, spatial_splits, "spatial"
        )

    # ── Compare and recommend ──────────────────────────────────
    print(f"\n{'='*60}")
    print(f"CV COMPARISON RESULTS")
    print(f"{'='*60}")

    for name, res in results.items():
        print(f"  {name:12s}: OOF AUC={res['oof_auc']:.5f}  "
              f"std={res['fold_std']:.5f}")

    if strategy == "compare" and len(results) == 2:
        strat_auc   = results["stratified"]["oof_auc"]
        spatial_auc = results["spatial"]["oof_auc"]
        gap         = strat_auc - spatial_auc

        print(f"\n  Gap (stratified - spatial): {gap:+.5f}")

        if gap > 0.01:
            recommendation = "spatial"
            print(f"\n  ⚠️  Gap > 0.01 — stratified CV is OPTIMISTIC.")
            print(f"  Recommendation: switch to SPATIAL block CV.")
            print(f"  The gate has been too lenient — raise MIN_DELTA to 0.008.")
        elif gap > 0.003:
            recommendation = "spatial"
            print(f"\n  ⚠️  Moderate gap — spatial CV is safer.")
            print(f"  Recommendation: switch to SPATIAL block CV.")
        else:
            recommendation = "stratified"
            print(f"\n  ✅ Gap < 0.003 — stratified CV is acceptable.")
            print(f"  Recommendation: keep STRATIFIED (current).")
    else:
        recommendation = strategy

    # ── Write to SKILL_STATE.json ──────────────────────────────
    state_store = SkillStateStore(paths.state_path)
    state_store.update(
        cv_strategy=recommendation,
        cv_stratified_oof_auc=results.get("stratified", {}).get("oof_auc"),
        cv_spatial_oof_auc=results.get("spatial", {}).get("oof_auc"),
        cv_gap=round(
            (results.get("stratified", {}).get("oof_auc", 0) or 0) -
            (results.get("spatial", {}).get("oof_auc", 0) or 0), 6
        ) if strategy == "compare" else None,
        dag_phase="phase_3_features",
        last_updated=datetime.now(timezone.utc).isoformat(),
    )
    print(f"\n✅ SKILL_STATE.json updated: cv_strategy={recommendation}")

    return {
        "status":         "OK",
        "strategy_chosen": recommendation,
        "results":         results,
        "recommendation":  recommendation,
    }


if __name__ == "__main__":
    strategy = "compare"
    for arg in sys.argv[1:]:
        if arg.startswith("--strategy="):
            strategy = arg.split("=", 1)[1]
        elif arg == "--strategy" and len(sys.argv) > sys.argv.index(arg) + 1:
            strategy = sys.argv[sys.argv.index(arg) + 1]

    if strategy not in ("compare", "spatial", "stratified"):
        print(f"❌ Unknown strategy '{strategy}'. "
              f"Use: compare, spatial, stratified")
        sys.exit(1)

    result = run(strategy=strategy)
    printable = {k: v for k, v in result.items() if k != "results"}
    print(json.dumps(printable, indent=2))