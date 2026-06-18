from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    Tuple,
    Protocol,
    runtime_checkable,
    cast,
)

import numpy as np
import pandas as pd
import os
import sys

try:
    orig_path = sys.path.copy()
    sys.path = [
        p
        for p in sys.path
        if p not in ("", ".", os.getcwd(), os.path.abspath(os.getcwd()))
    ]
    if "lightgbm" in sys.modules:
        del sys.modules["lightgbm"]
    import lightgbm as lgb
finally:
    sys.path = orig_path

from sklearn.metrics import f1_score, roc_auc_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from zindian.cv import get_cv_splits


@runtime_checkable
class Splitter(Protocol):
    def split(
        self, X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]: ...


@dataclass(frozen=True)
class LightGBMRunResult:
    oof_probs: np.ndarray
    test_probs: np.ndarray
    oof_auc: float  # retained for compatibility (classification)
    oof_f1: float  # retained for compatibility (classification)
    threshold: float
    fold_scores: list[float]
    oof_rmse: float = 0.0  # regression metric


def train_lightgbm_cv(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    *,
    n_splits: int = 5,
    random_seed: int | None = None,
    cv: Splitter | Iterable[Tuple[np.ndarray, np.ndarray]] | None = None,
    params: dict[str, Any] | None = None,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 50,
    scale: bool = True,
    threshold_grid: np.ndarray | None = None,
    per_fold_feature_fn: (
        Callable[
            [pd.DataFrame, pd.DataFrame, list, np.ndarray, np.ndarray | None],
            tuple[np.ndarray, np.ndarray],
        ]
        | None
    ) = None,
    regression_metric: str | None = None,
    variant_name: str | None = None,
) -> LightGBMRunResult:
    """Train a LightGBM CV model and return metrics.
    Supports both classification and regression based on challenge_config.task_type.
    When task_type == "regression", uses regression_metric to apply the correct
    target transformation and prediction inverse-mapping per the SoT v2.2
    Regression Target Transformation Lifecycle:
        "rmsle"                -> log1p(y) train, expm1(clip(raw, 0)) preds
        "root_mean_squared_error" / "mean_absolute_error" -> identity scale,
                                domain clipping via target_domain_bounds
    """
    from zindian.config import ChallengeConfig

    # Safely obtain task_type from provided config or fallback to default.
    try:
        from zindian.config import ChallengeConfig

        cfg = ChallengeConfig.load()
        task_type = str(cfg.get("task_type", "classification")).lower()
    except Exception:
        task_type = "classification"
    
    # Override task_type if regression_metric is explicitly provided
    if regression_metric is not None:
        task_type = "regression"
    # Resolve canonical seed if not provided
    if random_seed is None:
        from zindian.config import get_seed

        random_seed = get_seed()

    """Train a LightGBM CV model and return OOF/test probabilities plus metrics."""
    # Resolve canonical seed if not provided
    if random_seed is None:
        from zindian.config import get_seed

        random_seed = get_seed()

    np.random.seed(int(random_seed))

    lgb_params: dict[str, Any] = {
        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbose": -1,
        "seed": int(random_seed),
    }

    # If per_fold_feature_fn is provided, X and X_test will be computed inside the fold loop

    # Read variant model_override from sidecar if present
    _variant_objective = None
    _variant_max_delta = None
    if variant_name is not None:
        import pathlib as _pathlib
        import json as _json

        try:
            from zindian.config import ChallengeConfig

            _cfg = ChallengeConfig.load()._data
            _comp_slug = _cfg.get("slug") or _cfg.get("competition_slug") or ""
            if _comp_slug:
                _variant_sidecar = (
                    _pathlib.Path(__file__).parent.parent.parent
                    / "competitions"
                    / _comp_slug
                    / "variants"
                    / f"{variant_name}.json"
                )
                if _variant_sidecar.exists():
                    _sidecar_data = _json.loads(_variant_sidecar.read_text())
                    _model_override = _sidecar_data.get("model_override", {})
                    _variant_objective = _model_override.get("objective")
                    _variant_max_delta = _model_override.get("max_delta_step")
        except Exception:
            pass

    if task_type == "regression":
        y = np.asarray(train[target_col].values, dtype=np.float64)
        # Resolve target transformation based on regression metric (SoT v2.2)
        metric = str(regression_metric or "").lower()
        use_log1p = metric == "rmsle"
    else:
        y_raw = train[target_col].values
        if y_raw.dtype.kind in ("U", "S", "O"):  # text or object targets
            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()
            y = le.fit_transform(y_raw.astype(str))
        else:
            y = y_raw
        y = np.asarray(y, dtype=np.int32)
        use_log1p = False
    if per_fold_feature_fn is None:
        X = np.asarray(train[feature_cols].values, dtype=np.float64)
        X_test = np.asarray(test[feature_cols].values, dtype=np.float64)

        if scale:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            X_test = scaler.transform(X_test)
    else:
        X = np.zeros((len(train), len(feature_cols)), dtype=np.float64)
        X_test = np.zeros((len(test), len(feature_cols)), dtype=np.float64)
    if task_type == "regression":
        if _variant_objective == "poisson":
            lgb_params.update(
                {
                    "objective": "poisson",
                    "metric": "rmse",
                    "max_delta_step": _variant_max_delta or 0.7,
                }
            )
        else:
            lgb_params.update({"objective": "regression", "metric": "rmse"})
    else:
        # For non-binary/multiclass target spaces in special metrics, use multiclass objective
        n_classes = len(np.unique(y))
        if n_classes > 2:
            lgb_params.update(
                {
                    "objective": "multiclass",
                    "num_class": n_classes,
                    "metric": "multi_logloss",
                }
            )
        else:
            lgb_params.update({"objective": "binary", "metric": "binary_logloss"})
    if params:
        lgb_params.update(params)

    # Determine output shape for OOF/test arrays
    if task_type == "regression":
        oof_probs = np.zeros(len(train), dtype=np.float64)
        test_probs = np.zeros(len(test), dtype=np.float64)
    else:
        n_classes = len(np.unique(y))
        if n_classes > 2:
            oof_probs = np.zeros((len(train), n_classes), dtype=np.float64)
            test_probs = np.zeros((len(test), n_classes), dtype=np.float64)
        else:
            oof_probs = np.zeros(len(train), dtype=np.float64)
            test_probs = np.zeros(len(test), dtype=np.float64)
    fold_scores: list[float] = []

    # Obtain CV splits. If `cv` is provided it may be either:
    # - an sklearn splitter object (with .split)
    # - an iterable of (train_idx, val_idx) tuples
    # Otherwise fall back to the canonical CV splitter from `zindian.cv`.
    if cv is None:
        # Obtain an iterator of (train_idx, val_idx) from the central CV helpers
        split_iter = get_cv_splits(X, y, random_seed=random_seed)
    else:
        # If `cv` implements `split`, call it; otherwise assume it's an iterable of index pairs.
        if hasattr(cv, "split"):
            split_iter = cast(Splitter, cv).split(X, y)
        else:
            split_iter = iter(cv)

    # Resolve target_domain_bounds for RMSE/MAE domain clipping
    domain_bounds = None
    if task_type == "regression" and not use_log1p:
        try:
            _cfg = ChallengeConfig.load()
            _bounds = _cfg.get("target_domain_bounds") or {}
            min_b = _bounds.get("min")
            max_b = _bounds.get("max")
            if min_b is not None and max_b is not None:
                domain_bounds = (float(min_b), float(max_b))
        except Exception:
            domain_bounds = None

    for fold_idx, (tr_idx, val_idx) in enumerate(split_iter):
        # If per_fold_feature_fn is provided, recompute X and X_test for this fold
        if per_fold_feature_fn is not None:
            # Provide train, test DataFrames and indices to the callback. The callback
            # must return (X_full, X_test) arrays aligned to `train` and `test` rows.
            X_full, X_test = per_fold_feature_fn(
                train, test, feature_cols, tr_idx, np.asarray(train[target_col].values)
            )
            if scale:
                scaler = StandardScaler()
                X_full = scaler.fit_transform(X_full)
                X_test = scaler.transform(X_test)
            X = X_full

        # Target transformation per SoT v2.2 Regression Target Transformation Lifecycle:
        #   poisson -> raw counts (log-link applied internally by LightGBM)
        #   rmsle  -> log1p(y)  ;  RMSE/MAE -> identity
        if _variant_objective == "poisson":
            y_train_fold = y[tr_idx].astype(np.float64)
            y_val_fold = y[val_idx].astype(np.float64)
        elif task_type == "regression" and use_log1p:
            y_train_fold = np.log1p(y[tr_idx])
            y_val_fold = np.log1p(y[val_idx])
        else:
            y_train_fold = y[tr_idx]
            y_val_fold = y[val_idx]

        train_set = lgb.Dataset(X[tr_idx], label=y_train_fold)
        val_set = lgb.Dataset(X[val_idx], label=y_val_fold, reference=train_set)

        model = lgb.train(
            lgb_params,
            train_set,
            num_boost_round=num_boost_round,
            valid_sets=[val_set],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds),
                lgb.log_evaluation(period=-1),
            ],
        )

        val_pred_raw = np.asarray(model.predict(X[val_idx]), dtype=np.float64)
        test_pred_raw = np.asarray(model.predict(X_test), dtype=np.float64)

        # Store predictions based on task type and number of classes
        if task_type == "regression":
            val_pred_flat = val_pred_raw
            test_pred_flat = test_pred_raw
        elif len(val_pred_raw.shape) > 1 and val_pred_raw.shape[1] > 2:
            # Multiclass: keep full probability matrix
            oof_probs[val_idx] = val_pred_raw
            test_probs += test_pred_raw / n_splits
            val_pred_flat = np.argmax(val_pred_raw, axis=1).astype(np.float64)
            test_pred_flat = None  # Not used for multiclass
        elif len(val_pred_raw.shape) > 1 and val_pred_raw.shape[1] == 2:
            # Binary classification with 2-column output
            val_pred_flat = val_pred_raw[:, 1]
            test_pred_flat = test_pred_raw[:, 1]
        else:
            val_pred_flat = val_pred_raw
            test_pred_flat = test_pred_raw

        # Prediction inverse-mapping per SoT v2.2:
        #   poisson -> clip(raw, 0) only (already in count space)
        #   rmsle  -> clip(raw, 0) then expm1
        #   RMSE/MAE -> clip(raw, domain_bounds)
        if task_type == "regression":
            if _variant_objective == "poisson":
                val_pred_final = np.clip(val_pred_flat, 0, None)
                test_pred_final = np.clip(test_pred_flat, 0, None)
            elif use_log1p:
                val_pred_final = np.expm1(np.clip(val_pred_flat, 0, None))
                test_pred_final = np.expm1(np.clip(test_pred_flat, 0, None))
            else:
                if domain_bounds is not None:
                    val_pred_final = np.clip(
                        val_pred_flat, domain_bounds[0], domain_bounds[1]
                    )
                    test_pred_final = np.clip(
                        test_pred_flat, domain_bounds[0], domain_bounds[1]
                    )
                else:
                    val_pred_final = val_pred_flat
                    test_pred_final = test_pred_flat
            oof_probs[val_idx] = val_pred_final
            test_probs += test_pred_final / n_splits
        elif len(val_pred_raw.shape) == 1 or val_pred_raw.shape[1] <= 2:
            # Binary classification: store scalar predictions
            val_pred_final = val_pred_flat
            test_pred_final = test_pred_flat
            oof_probs[val_idx] = val_pred_final
            test_probs += test_pred_final / n_splits
        # else: multiclass already stored above
        if task_type == "regression":
            if use_log1p:
                # Compute RMSLE on back-transformed (original-space) predictions
                fold_rmsle = float(
                    np.sqrt(
                        np.mean((np.log1p(y[val_idx]) - np.log1p(oof_probs[val_idx])) ** 2)
                    )
                )
                fold_scores.append(fold_rmsle)
                print(f"  Fold {fold_idx + 1}/{n_splits}: rmsle={fold_rmsle:.6f}")
            else:
                fold_rmse = float(root_mean_squared_error(y[val_idx], oof_probs[val_idx]))
                fold_scores.append(fold_rmse)
                print(f"  Fold {fold_idx + 1}/{n_splits}: rmse={fold_rmse:.6f}")
        else:
            try:
                # Standard binary classification validation
                fold_auc = float(roc_auc_score(y[val_idx], val_pred_flat))
            except Exception:
                # Secondary validation block for unconventional metrics/multiclass
                from sklearn.metrics import accuracy_score

                try:
                    fold_auc = float(
                        accuracy_score(y[val_idx], np.round(val_pred_flat).astype(int))
                    )
                except Exception:
                    fold_auc = 0.0
            fold_scores.append(fold_auc)
            print(f"  Fold {fold_idx + 1}/{n_splits}: score={fold_auc:.6f}")

    if task_type == "regression":
        # Score computation per SoT v2.2 Regression Target Transformation Lifecycle:
        #   rmsle -> RMSLE in original space: sqrt(mean((log(y+1) - log(yhat+1))^2))
        #   RMSE/MAE -> standard RMSE in original space (computed on back-transformed predictions)
        if use_log1p:
            rmsle_val = np.sqrt(np.mean((np.log1p(y) - np.log1p(oof_probs)) ** 2))
            oof_rmse = float(rmsle_val)
        else:
            oof_rmse = float(root_mean_squared_error(y, oof_probs))
        return LightGBMRunResult(
            oof_probs=oof_probs,
            test_probs=test_probs,
            oof_auc=0.0,
            oof_f1=0.0,
            oof_rmse=oof_rmse,
            threshold=0.0,
            fold_scores=fold_scores,
        )
    else:
        n_classes = len(np.unique(y))
        is_multiclass = n_classes > 2
        
        try:
            if is_multiclass:
                from sklearn.preprocessing import label_binarize
                y_bin = label_binarize(y, classes=np.unique(y))
                oof_auc = float(roc_auc_score(y_bin, oof_probs, average='macro', multi_class='ovr'))
            else:
                oof_auc = float(roc_auc_score(y, oof_probs))
        except Exception:
            oof_auc = 0.0

        if is_multiclass:
            y_pred = np.argmax(oof_probs, axis=1)
            oof_f1 = float(f1_score(y, y_pred, average='macro'))
            best_t = 0.0
        else:
            try:
                if threshold_grid is None:
                    threshold_grid = np.arange(0.3, 0.7, 0.01)
                best_t = float(
                    max(
                        threshold_grid,
                        key=lambda t: f1_score(y, (oof_probs >= t).astype(int)),
                    )
                )
                oof_f1 = float(f1_score(y, (oof_probs >= best_t).astype(int)))
            except Exception:
                best_t = 0.0
                oof_f1 = 0.0

        return LightGBMRunResult(
            oof_probs=oof_probs,
            test_probs=test_probs,
            oof_auc=oof_auc,
            oof_f1=oof_f1,
            oof_rmse=0.0,
            threshold=best_t,
            fold_scores=fold_scores,
        )
