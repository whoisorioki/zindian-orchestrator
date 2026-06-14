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
import lightgbm as lgb
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
    fold_aucs: list[float]
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
) -> LightGBMRunResult:
    """Train a LightGBM CV model and return metrics.
    Supports both classification and regression based on challenge_config.task_type.
    """
    from zindian.config import ChallengeConfig

    # Safely obtain task_type from provided config or fallback to default.
    try:
        from zindian.config import ChallengeConfig
        cfg = ChallengeConfig.load()
        task_type = str(cfg.get("task_type", "classification")).lower()
    except Exception:
        task_type = "classification"
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
    if task_type == "regression":
        y = np.asarray(train[target_col].values, dtype=np.float64)
    else:
        y_raw = train[target_col].values
        if y_raw.dtype.kind in ('U', 'S', 'O'):  # text or object targets
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y_raw.astype(str))
        else:
            y = y_raw
        y = np.asarray(y, dtype=np.int32)
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
        lgb_params.update({"objective": "regression", "metric": "rmse"})
    else:
        # For non-binary/multiclass target spaces in special metrics, use multiclass objective
        n_classes = len(np.unique(y))
        if n_classes > 2:
            lgb_params.update({"objective": "multiclass", "num_class": n_classes, "metric": "multi_logloss"})
        else:
            lgb_params.update({"objective": "binary", "metric": "binary_logloss"})
    if params:
        lgb_params.update(params)

    oof_probs = np.zeros(len(train), dtype=np.float64)
    test_probs = np.zeros(len(test), dtype=np.float64)
    fold_aucs: list[float] = []

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

        train_set = lgb.Dataset(X[tr_idx], label=y[tr_idx])
        val_set = lgb.Dataset(X[val_idx], label=y[val_idx], reference=train_set)

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

        val_pred = np.asarray(model.predict(X[val_idx]), dtype=np.float64)
        test_pred = np.asarray(model.predict(X_test), dtype=np.float64)
        
        # If multiclass output, val_pred/test_pred will have shape (N, n_classes)
        # For simplicity, extract the probability of class 1 or prediction index
        if len(val_pred.shape) > 1 and val_pred.shape[1] > 1:
            val_pred_flat = val_pred[:, 1] if val_pred.shape[1] == 2 else np.argmax(val_pred, axis=1).astype(np.float64)
            test_pred_flat = test_pred[:, 1] if test_pred.shape[1] == 2 else np.argmax(test_pred, axis=1).astype(np.float64)
        else:
            val_pred_flat = val_pred
            test_pred_flat = test_pred

        oof_probs[val_idx] = val_pred_flat
        test_probs += test_pred_flat / n_splits
        if task_type == "regression":
            fold_rmse = root_mean_squared_error(y[val_idx], val_pred_flat)
            fold_aucs.append(fold_rmse)
            print(f"  Fold {fold_idx + 1}/{n_splits}: rmse={fold_rmse:.6f}")
        else:
            try:
                # Standard binary classification validation
                fold_auc = float(roc_auc_score(y[val_idx], val_pred_flat))
            except Exception:
                # Secondary validation block for unconventional metrics/multiclass
                from sklearn.metrics import accuracy_score
                try:
                    fold_auc = float(accuracy_score(y[val_idx], np.round(val_pred_flat).astype(int)))
                except Exception:
                    fold_auc = 0.0
            fold_aucs.append(fold_auc)
            print(f"  Fold {fold_idx + 1}/{n_splits}: score={fold_auc:.6f}")

    if task_type == "regression":
        oof_rmse = root_mean_squared_error(y, oof_probs)
        return LightGBMRunResult(
            oof_probs=oof_probs,
            test_probs=test_probs,
            oof_auc=0.0,
            oof_f1=0.0,
            oof_rmse=oof_rmse,
            threshold=0.0,
            fold_aucs=fold_aucs,
        )
    else:
        try:
            oof_auc = float(roc_auc_score(y, oof_probs))
        except Exception:
            oof_auc = 0.0
        
        try:
            if threshold_grid is None:
                threshold_grid = np.arange(0.3, 0.7, 0.01)
            best_t = float(
                max(threshold_grid, key=lambda t: f1_score(y, (oof_probs >= t).astype(int)))
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
            fold_aucs=fold_aucs,
        )
