"""Skill 06 — Cleaning / Data Imputation.

Phase 2A. Reads cleaned EDA state, applies MNAR indicators, MCAR imputation,
and dynamic constant-column dropping.

Phase contract (SoT §Phase 2A):
    policy_gate() → skill_06_cleaning

Reads:
    state["eda"]["mnar_columns"]       — columns with non-random missingness
    state["eda"]["mcar_columns"]       — columns with random missingness
    state["config"]                    — competition config (optional fallback)

Writes:
    state["cleaning"] — {
        "mnar_indicators_created": [...],
        "mcar_imputed_medians": {...},
        "constant_columns_dropped": [...],
        "feature_matrix_train_shape": [...],
        "feature_matrix_test_shape": [...],
        "n_constant_train": int,
        "n_constant_test": int,
    }
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def _build_mnar_indicators(
    df: pd.DataFrame,
    mnar_columns: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """First pass — create _is_missing binary indicators for all MNAR columns.

    This MUST complete across ALL MNAR columns before any imputation happens.
    The SoT (§Phase 2A) mandates: "ORDER IS MANDATORY — indicator before fill."
    """
    indicators_created: List[str] = []
    for col in mnar_columns:
        if col not in df.columns:
            continue
        indicator_col = f"{col}_is_missing"
        df[indicator_col] = df[col].isnull().astype(np.int8)
        indicators_created.append(indicator_col)
    return df, indicators_created


def _impute_mcar(
    df: pd.DataFrame,
    mcar_columns: List[str],
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Second pass — impute MCAR columns with fold-derived median (numeric) or mode (categorical).

    Returns the imputed DataFrame and a dict mapping column → imputation value.
    """
    impute_values: Dict[str, Any] = {}
    for col in mcar_columns:
        if col not in df.columns:
            continue
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        value: Any
        if is_numeric:
            # Use fold-restricted median from training data
            # (here we compute from full df as proxy; the orchestrator
            #  should restrict to training fold before calling this)
            value = float(df[col].median())
            df[col] = df[col].fillna(value)
        else:
            value = df[col].mode().iloc[0] if not df[col].mode().empty else None
            df[col] = df[col].fillna(value)
        impute_values[col] = value
    return df, impute_values


def _drop_constants(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str], List[str]]:
    """Third pass — dynamic variance scan. Drop columns constant in BOTH splits.

    Returns cleaned train, cleaned test, intersection of dropped cols,
    dropped in train only, dropped in test only.
    """
    train_nunique = train.nunique(dropna=False)
    test_nunique = test.nunique(dropna=False)

    const_in_train = [str(c) for c, n in train_nunique.items() if n <= 1]
    const_in_test = [str(c) for c, n in test_nunique.items() if n <= 1]

    # Only drop columns constant in BOTH (intersection) to avoid split mismatch
    const_both = list(set(const_in_train) & set(const_in_test))

    train_clean = train.drop(columns=const_both, errors="ignore")
    test_clean = test.drop(columns=const_both, errors="ignore")

    return train_clean, test_clean, const_both, const_in_train, const_in_test


def run(config: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Run the Phase 2A cleaning pipeline: MNAR indicators → MCAR impute → drop constants.

    Args:
        config: challenge_config.json as a dict.
        state: SKILL_STATE.json as a dict (must contain state["eda"]).

    Returns:
        Updated state dict with cleaning metadata written.
    """
    # ── Read EDA state ──────────────────────────────────────────────
    eda = state.get("eda", {})
    mnar_columns: List[str] = eda.get("mnar_columns", [])
    mcar_columns: List[str] = eda.get("mcar_columns", [])

    # Feature matrices must already be materialised upstream.
    # The orchestrator provides X_train, X_test via state or side-channel.
    _x_train_raw = state.get("X_train")
    _x_test_raw = state.get("X_test")

    if _x_train_raw is None or _x_test_raw is None:
        raise ValueError(
            "skill_06_cleaning requires 'X_train' and 'X_test' in state. "
            "The orchestrator must load and pass feature matrices."
        )
    train: pd.DataFrame = _x_train_raw
    test: pd.DataFrame = _x_test_raw

    # ── Step 1: MNAR indicators (all columns, before any fill) ─────
    train, indicators = _build_mnar_indicators(train, mnar_columns)
    test, _ = _build_mnar_indicators(test, mnar_columns)

    # ── Step 2: MCAR imputation (fold-restricted median/mode) ──────
    train, impute_values = _impute_mcar(train, mcar_columns)
    # Use training-derived impute values on test to avoid data leakage
    for col, value in impute_values.items():
        if col in test.columns:
            test[col] = test[col].fillna(value)

    # ── Step 3: Dynamic constant column dropping ────────────────────
    train, test, const_both, const_train_only, const_test_only = _drop_constants(
        train, test
    )

    # ── Write cleaning metadata to state ────────────────────────────
    state["cleaning"] = {
        "mnar_indicators_created": indicators,
        "mcar_imputed_medians": impute_values,
        "constant_columns_dropped": const_both,
        "constant_columns_train_only": list(const_train_only),
        "constant_columns_test_only": list(const_test_only),
        "feature_matrix_train_shape": list(train.shape),
        "feature_matrix_test_shape": list(test.shape),
        "n_mnar_indicators": len(indicators),
        "n_mcar_imputed": len(impute_values),
        "n_constant_dropped": len(const_both),
    }

    # Surface cleaned matrices back to state for downstream skills
    state["X_train_clean"] = train
    state["X_test_clean"] = test

    return state
