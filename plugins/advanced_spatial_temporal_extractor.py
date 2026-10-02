"""Advanced Spatial-Temporal Extractor Plugin.

Implements the "Boost-then-Convolve" Spatial-Temporal Architecture:
1. Treed Distributed Lag Non-Linear Modeling (DLNM) Matrices (1-to-8 week rolling lags).
   - Sum for precipitation, Mean for thermal/temperature variables.
2. Out-of-Fold (OOF) Spatial Risk Smoothing via BallTree (Haversine).
   - Built strictly on TRAIN location centroids to prevent test location graph leakage.
   - Strict fold-restricted historical target risk calculation (Approach A).
   - Distance threshold (>150km) fallback to fold global mean.
3. Zero-leakage merge using an order-safe backward as-of join
   (`zindian.joins.sorted_asof_join`), not a bare `pd.merge_asof` -- the
   `.loc[df.index]` restore that followed a plain merge_asof is positional,
   not identity-based, and silently scrambles rows whenever the join key
   contains ties.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree

from plugins.base_extractor import FeatureExtractor
from zindian.joins import sorted_asof_join


class Extractor(FeatureExtractor):
    """A5-compliant Feature Extractor for Advanced Spatial-Temporal DLNM & KNN Smoothing."""

    def fetch(self, paths, config, allow_network: bool = True) -> Path:
        """Fetch or locate base external climate and health dataset."""
        comp_dir = getattr(paths, "competition_dir", paths.data_raw_dir.parent)
        aligned_path = comp_dir / "data" / "external" / "aligned_macro_proxies.parquet"
        macro_path = comp_dir / "data" / "external" / "macro_stress_proxies.parquet"

        if aligned_path.exists():
            return aligned_path
        if macro_path.exists():
            return macro_path

        try:
            from scripts.build_macro_stress_proxies import build_macro_stress_proxies
            return build_macro_stress_proxies(paths.data_raw_dir, aligned_path, config)
        except Exception:
            return paths.data_raw_dir / config.get("input_files", {}).get("train", "Train.csv")

    def _compute_dlnm_lags(
        self,
        df: pd.DataFrame,
        date_col: str,
        group_cols: List[str],
        var_cols: List[str],
        max_weeks: int = 8,
    ) -> pd.DataFrame:
        """
        Compute discrete 1-to-8 week lagged matrices for climate variables.
        Captures vector-borne biological incubation periods.
        - Sum for precipitation.
        - Mean for temperature and heat stress metrics.
        """
        df = df.sort_values(by=group_cols + [date_col]).copy()

        for var in var_cols:
            if var not in df.columns:
                continue
            is_precip = "precip" in var.lower()
            for w in range(1, max_weeks + 1):
                lag_col = f"{var}_lag_{w}w"
                shifted = df.groupby(group_cols)[var].shift((w - 1) * 7)
                if is_precip:
                    df[lag_col] = shifted.rolling(window=7, min_periods=1).sum()
                else:
                    df[lag_col] = shifted.rolling(window=7, min_periods=1).mean()
                df[lag_col] = df[lag_col].fillna(0.0)
        return df

    def _compute_oof_spatial_smooth(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        lat_col: str,
        lon_col: str,
        target_col: str,
        paths: Any,
        n_neighbors: int = 3,
        max_distance_km: float = 150.0,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Hierarchical Out-of-Fold Spatial Risk Smoothing across location centroids using BallTree.
        
        Strict zero-leakage contract:
        - BallTree is constructed STRICTLY on training set location centroids.
        - Training locations find their nearest neighboring training locations (excluding self).
        - Test locations query the training BallTree to find nearest training neighbors.
        """
        train_locs = (
            train_df[[lat_col, lon_col]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        test_locs = (
            test_df[[lat_col, lon_col]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        earth_radius_km = 6371.0
        train_coords_rad = np.radians(train_locs[[lat_col, lon_col]].values)
        test_coords_rad = np.radians(test_locs[[lat_col, lon_col]].values)

        # Build BallTree strictly on training location centroids
        tree = BallTree(train_coords_rad, metric="haversine")

        # 1. Build neighbor mapping for training locations (querying training tree, k+1 to exclude self)
        train_knn_map: Dict[Tuple[float, float], List[Tuple[Tuple[float, float], float]]] = {}
        k_query_tr = min(n_neighbors + 1, len(train_locs))
        for idx, row in train_locs.iterrows():
            loc_key = (float(row[lat_col]), float(row[lon_col]))
            dists_rad, indices = tree.query(train_coords_rad[idx : idx + 1], k=k_query_tr)
            dists_km = dists_rad[0] * earth_radius_km
            idx_list = indices[0]

            neighbors_with_dists = []
            for d_km, i in zip(dists_km, idx_list):
                if i != idx and len(neighbors_with_dists) < n_neighbors:
                    n_key = (float(train_locs.iloc[i][lat_col]), float(train_locs.iloc[i][lon_col]))
                    neighbors_with_dists.append((n_key, float(d_km)))
            train_knn_map[loc_key] = neighbors_with_dists

        # 2. Build neighbor mapping for test locations (querying training tree, k neighbors)
        test_knn_map: Dict[Tuple[float, float], List[Tuple[Tuple[float, float], float]]] = {}
        k_query_te = min(n_neighbors, len(train_locs))
        for idx, row in test_locs.iterrows():
            loc_key = (float(row[lat_col]), float(row[lon_col]))
            dists_rad, indices = tree.query(test_coords_rad[idx : idx + 1], k=k_query_te)
            dists_km = dists_rad[0] * earth_radius_km
            idx_list = indices[0]

            neighbors_with_dists = []
            for d_km, i in zip(dists_km, idx_list):
                n_key = (float(train_locs.iloc[i][lat_col]), float(train_locs.iloc[i][lon_col]))
                neighbors_with_dists.append((n_key, float(d_km)))
            test_knn_map[loc_key] = neighbors_with_dists

        # Load explicit CV splits from data_processed_dir or state if available
        cv_splits = None
        cv_split_path = paths.data_processed_dir / "cv_split_indices.json"
        if cv_split_path.exists():
            try:
                cv_splits = json.loads(cv_split_path.read_text(encoding="utf-8"))
            except Exception:
                cv_splits = None

        if cv_splits is None:
            from zindian.cv import make_cv_splitter
            kf = make_cv_splitter({"type": "kfold", "n_splits": 5}, random_seed=42)
            cv_splits = [(tr.tolist(), val.tolist()) for tr, val in kf.split(train_df)]

        train_spatial_risk = pd.Series(0.0, index=train_df.index)

        # OOF Fold iteration on training set
        if target_col in train_df.columns:
            for tr_idx, val_idx in cv_splits:
                tr_fold = train_df.iloc[tr_idx]

                # Compute location mean target risk using train fold rows ONLY
                loc_risk = tr_fold.groupby([lat_col, lon_col])[target_col].mean().to_dict()
                global_risk = float(tr_fold[target_col].mean()) if len(tr_fold) > 0 else 0.0

                val_risk_list = []
                for _, row in train_df.iloc[val_idx].iterrows():
                    loc_key = (float(row[lat_col]), float(row[lon_col]))
                    neighbors = train_knn_map.get(loc_key, [])
                    valid_neighbor_risks = []
                    for n_key, d_km in neighbors:
                        if d_km <= max_distance_km and n_key in loc_risk:
                            valid_neighbor_risks.append(loc_risk[n_key])

                    if valid_neighbor_risks:
                        val_risk_list.append(float(np.mean(valid_neighbor_risks)))
                    else:
                        val_risk_list.append(global_risk)

                train_spatial_risk.iloc[val_idx] = val_risk_list

        # Test set risk computation using FULL training set target means
        full_loc_risk = train_df.groupby([lat_col, lon_col])[target_col].mean().to_dict()
        full_global_risk = float(train_df[target_col].mean()) if len(train_df) > 0 else 0.0

        test_risk_list = []
        for _, row in test_df.iterrows():
            loc_key = (float(row[lat_col]), float(row[lon_col]))
            neighbors = test_knn_map.get(loc_key, [])
            valid_neighbor_risks = []
            for n_key, d_km in neighbors:
                if d_km <= max_distance_km and n_key in full_loc_risk:
                    valid_neighbor_risks.append(full_loc_risk[n_key])

            if valid_neighbor_risks:
                test_risk_list.append(float(np.mean(valid_neighbor_risks)))
            else:
                test_risk_list.append(full_global_risk)

        test_spatial_risk = pd.Series(test_risk_list, index=test_df.index)

        return train_spatial_risk, test_spatial_risk

    def extract(
        self, paths, data_path: Path, config, branch_name: str | None = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract features, build DLNM lags, compute OOF spatial risk, and perform the order-safe backward as-of join."""
        input_files = config.get("input_files", {}) or {}
        train_file = input_files.get("train", "Train.csv")
        test_file = input_files.get("test", "Test.csv")

        cols_cfg = config.get("columns", {}) or {}
        id_col = config.get("id_col") or "ID"
        lat_col = cols_cfg.get("latitude") or config.get("spatial_signal", {}).get("lat_col") or "latitude"
        lon_col = cols_cfg.get("longitude") or config.get("spatial_signal", {}).get("lon_col") or "longitude"
        target_col = config.get("target_col") or "is_climate_sensitive"
        date_col = config.get("temporal_col") or "deathdate"
        banned = list(config.get("banned_features") or [])

        train = pd.read_csv(paths.data_raw_dir / train_file)
        test = pd.read_csv(paths.data_raw_dir / test_file)

        if date_col in train.columns:
            train[date_col] = pd.to_datetime(train[date_col])
        if date_col in test.columns:
            test[date_col] = pd.to_datetime(test[date_col])

        climate_vars = []
        for candidate in ["precipitation", "wbgt_max", "wbgt_approx", "wbgt_max_7d", "max_temperature", "avg_temperature"]:
            if candidate in train.columns:
                climate_vars.append(candidate)

        if not climate_vars:
            climate_vars = [c for c in train.columns if c not in (id_col, lat_col, lon_col, date_col, target_col)][:2]

        # 1. Compute Treed DLNM Lag Matrices (1 to 8 weeks)
        train_dlnm = self._compute_dlnm_lags(
            train, date_col=date_col, group_cols=[lat_col, lon_col], var_cols=climate_vars, max_weeks=8
        )
        test_dlnm = self._compute_dlnm_lags(
            test, date_col=date_col, group_cols=[lat_col, lon_col], var_cols=climate_vars, max_weeks=8
        )

        # 2. Compute Out-of-Fold Spatial KNN Risk Smoothing via BallTree (Train-only tree)
        train_spatial_risk, test_spatial_risk = self._compute_oof_spatial_smooth(
            train_dlnm, test_dlnm, lat_col=lat_col, lon_col=lon_col, target_col=target_col, paths=paths, n_neighbors=3
        )
        train_dlnm["spatial_smooth_risk"] = train_spatial_risk.values
        test_dlnm["spatial_smooth_risk"] = test_spatial_risk.values

        # 3. Point-in-time backward join, order-safe.
        #
        # The previous form sorted, merged, then restored with
        # `train_sorted.loc[train.index]`. `merge_asof` returns a FRESH
        # RangeIndex, so that restore is positional, not by identity -- under a
        # date tie it attached the wrong proxy values to the wrong rows while
        # every row count and every file-level ID check still passed.
        # `sorted_asof_join` carries `id_col` through and restores by ID.
        #
        # `allow_unmatched=True` preserves the prior behaviour: rows whose
        # location has no earlier history arrive as NaN and are imputed by the
        # median pass below. With `by=[lat, lon]` this is the common case.
        if data_path.suffix == ".parquet" and data_path.exists():
            ext_df = pd.read_parquet(data_path)
            if date_col in ext_df.columns:
                ext_df[date_col] = pd.to_datetime(ext_df[date_col])

                # Merge external proxy columns backward
                ext_cols = [c for c in ext_df.columns if c not in (lat_col, lon_col, date_col)]
                if ext_cols:
                    train_dlnm = sorted_asof_join(
                        train_dlnm,
                        ext_df[[date_col, lat_col, lon_col] + ext_cols],
                        id_col=id_col,
                        date_col=date_col,
                        by_cols=[lat_col, lon_col],
                        direction="backward",
                        allow_unmatched=True,
                    )
                    test_dlnm = sorted_asof_join(
                        test_dlnm,
                        ext_df[[date_col, lat_col, lon_col] + ext_cols],
                        id_col=id_col,
                        date_col=date_col,
                        by_cols=[lat_col, lon_col],
                        direction="backward",
                        allow_unmatched=True,
                    )

        # Impute missing feature values
        feature_cols = [
            c for c in train_dlnm.columns
            if c not in (id_col, target_col, date_col) + tuple(banned)
        ]
        for col in feature_cols:
            if train_dlnm[col].dtype.kind in "bifc":
                med_val = train_dlnm[col].median()
                if pd.isna(med_val):
                    med_val = 0.0
                train_dlnm[col] = train_dlnm[col].fillna(med_val)
                test_dlnm[col] = test_dlnm[col].fillna(med_val)

        # Drop banned columns. `id_col` is RETAINED so the output can be verified
        # against raw `Train.csv` by ID. A5: id_col comes from config.
        drop_cols_tr = list(set(c for c in banned if c in train_dlnm.columns))
        drop_cols_te = list(set(c for c in banned if c in test_dlnm.columns))

        train_feat = train_dlnm.drop(columns=drop_cols_tr, errors="ignore")
        test_feat = test_dlnm.drop(columns=drop_cols_te, errors="ignore")

        # Save to processed directory for DAG reproducibility contract
        branch = branch_name or "residual-dlnm-specialist"
        paths.data_processed_dir.mkdir(parents=True, exist_ok=True)
        train_feat.to_csv(
            paths.data_processed_dir / f"features_train_{branch}.csv", index=False
        )
        test_feat.to_csv(
            paths.data_processed_dir / f"features_test_{branch}.csv", index=False
        )

        return train_feat, test_feat


def extract(
    paths, data_path: Path, config, branch_name: str = "residual-dlnm-specialist"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Top-level extract function for plugin dispatcher interface."""
    extractor = Extractor(config)
    return extractor.extract(paths, data_path, config, branch_name)


def fetch(paths, config, allow_network: bool = True) -> Path:
    """Top-level fetch function for plugin dispatcher interface."""
    extractor = Extractor(config)
    return extractor.fetch(paths, config, allow_network)
