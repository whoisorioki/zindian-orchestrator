"""
Skill 04 — EDA / Data Quality Audit

Generic, competition-agnostic EDA for pipeline stages 2 and 3.
Writes reports/eda_report.json and reports/eda_summary.md and updates SKILL_STATE.json.

Usage: python3 -m zindian.skills.skill_04_eda
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from typing import Any, Mapping

from zindian.paths import resolve_competition_paths, CompetitionPaths


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return data


def detect_target(paths: CompetitionPaths) -> str:
    # Try challenge_config.json -> SKILL_STATE.json only; never guess.
    cfg = _load_json_object(paths.config_path)
    for key in ("target_col", "target", "label", "target_column", "output_column"):
        value = cfg.get(key)
        if value:
            return str(value)

    state = _load_json_object(paths.state_path)
    for key in ("target_col", "target", "target_column", "primary_key", "label"):
        value = state.get(key)
        if value:
            return str(value)

    raise ValueError(
        "Unable to resolve target column from challenge_config.json or SKILL_STATE.json; "
        "EDA will not guess a target column."
    )


def _load_eda_config(paths: CompetitionPaths) -> dict[str, Any]:
    return _load_json_object(paths.config_path)


def _extract_explicit_encoding_rules(config_data: Mapping[str, Any]) -> dict[str, str]:
    rules: dict[str, str] = {}

    def add_rules(value: Any, default_encoding: str) -> None:
        if isinstance(value, Mapping):
            for name, encoding in value.items():
                if name:
                    rules[str(name)] = str(encoding or default_encoding)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                if isinstance(item, str):
                    rules[item] = default_encoding
                elif isinstance(item, Mapping):
                    name = item.get("name") or item.get("column") or item.get("feature")
                    if name:
                        rules[str(name)] = str(
                            item.get("encoding") or item.get("type") or default_encoding
                        )

    add_rules(config_data.get("feature_encoding", {}), "categorical")
    add_rules(config_data.get("categorical_columns", []), "categorical")
    add_rules(config_data.get("ordinal_columns", []), "ordinal")
    add_rules(config_data.get("nominal_columns", []), "nominal")
    return rules


def _high_correlation_pairs(
    corr: pd.DataFrame, thresh: float = 0.95
) -> list[tuple[str, str, float]]:
    pairs: list[tuple[str, str, float]] = []
    labels = list(corr.index)
    positions = {label: idx for idx, label in enumerate(labels)}

    for row_name, row_values in corr.iterrows():
        row_pos = positions[row_name]
        for col_name, value in row_values.items():
            if positions[col_name] <= row_pos:
                continue
            if pd.notna(value) and float(value) > thresh:
                pairs.append((str(row_name), str(col_name), float(value)))
    return pairs


def _outlier_summary(series: pd.Series, total_rows: int) -> dict[str, Any]:
    numeric_values = (
        pd.to_numeric(series, errors="coerce").dropna().astype(float).to_numpy()
    )
    if numeric_values.size == 0:
        return {"outlier_pct": 0.0, "flag": False, "method": "empty", "skewness": 0.0}

    skew_value: Any = (
        pd.Series(numeric_values).skew() if numeric_values.size >= 3 else 0.0
    )
    skewness = float(skew_value) if not pd.isna(skew_value) else 0.0
    if pd.isna(skewness):
        skewness = 0.0

    abs_skewness = abs(skewness)
    if abs_skewness >= 1.0:
        median = float(np.median(numeric_values))
        mad = float(np.median(np.abs(numeric_values - median)))
        if mad > 0:
            modified_z = 0.6745 * np.abs(numeric_values - median) / mad
            outlier_mask = modified_z > 3.5
            method = "mad"
        else:
            outlier_mask = np.abs(numeric_values - median) > 0
            method = "median_deviation"
    else:
        q1 = float(np.quantile(numeric_values, 0.25))
        q3 = float(np.quantile(numeric_values, 0.75))
        iqr = q3 - q1
        if iqr > 0:
            low = q1 - 1.5 * iqr
            high = q3 + 1.5 * iqr
            outlier_mask = np.logical_or(numeric_values < low, numeric_values > high)
            method = "iqr"
        else:
            low, high = np.quantile(numeric_values, [0.01, 0.99])
            outlier_mask = np.logical_or(numeric_values < low, numeric_values > high)
            method = "quantile_fence"

    outlier_pct = float(outlier_mask.sum() / max(1, total_rows))
    return {
        "outlier_pct": outlier_pct,
        "flag": outlier_pct > 0.05,
        "method": method,
        "skewness": skewness,
    }


def _build_categorical_columns(
    df: pd.DataFrame, feature_cols: list[str], explicit_rules: Mapping[str, str]
) -> list[dict[str, Any]]:
    categorical_columns: list[dict[str, Any]] = []
    for column in feature_cols:
        column_rules = explicit_rules.get(column)
        if (
            pd.api.types.is_object_dtype(df[column])
            or pd.api.types.is_string_dtype(df[column])
            or str(df[column].dtype) == "category"
        ):
            categorical_columns.append(
                {
                    "name": column,
                    "cardinality": int(df[column].nunique(dropna=False)),
                    "encoding": column_rules or "one-hot or ordinal",
                }
            )
        elif column_rules is not None:
            categorical_columns.append(
                {
                    "name": column,
                    "cardinality": int(df[column].nunique(dropna=False)),
                    "encoding": column_rules,
                }
            )
    return categorical_columns


def mcar_mnar_assessment(df: pd.DataFrame, col: str, target: str) -> str:
    # If no nulls, return 'none'
    series = df[col]
    null_rate = series.isnull().mean()
    if null_rate == 0:
        return "none"
    # If correlation between null indicator and target > small threshold -> MNAR
    if target in df.columns:
        null_ind = series.isnull().astype(float)
        target_series = pd.to_numeric(df[target], errors="coerce").astype(float)
        corr = null_ind.corr(target_series)
        if pd.notna(corr) and abs(corr) >= 0.05:
            return "MNAR"
    return "MCAR"


def run():
    paths = resolve_competition_paths(require_competition=True)
    competition_dir = paths.competition_dir
    if competition_dir is None:
        raise FileNotFoundError(
            "resolve_competition_paths did not return a competition_dir"
        )
    reports_dir = paths.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Prefer processed features file, fall back to raw training file
    proc_train = competition_dir / "data" / "processed" / "features_train.csv"
    cfg = _load_json_object(paths.config_path)
    train_file = (cfg.get("input_files") or {}).get("train", "Training_Data.csv")
    raw_train = paths.data_raw_dir / train_file

    if proc_train.exists():
        df = pd.read_csv(proc_train)
    else:
        if not raw_train.exists():
            raise FileNotFoundError(
                f"Training data not found at {proc_train} or {raw_train}"
            )
        df = pd.read_csv(raw_train)

    config_data = _load_eda_config(paths)
    explicit_rules = _extract_explicit_encoding_rules(config_data)

    # Detect target column
    target = detect_target(paths)

    if target not in df.columns:
        raise ValueError(
            f"Resolved target column '{target}' is not present in the training data"
        )

    # Compute standard deviation of target column
    target_std = float(np.std(df[target].values, ddof=1))

    # Exclude ID, coords, and target
    exclude_lower = {"id", "id_number", "latitude", "longitude", target.lower()}
    feature_cols = [c for c in df.columns if c.lower() not in exclude_lower]

    # DATA QUALITY
    n_rows, n_cols = df.shape
    feature_count = len(feature_cols)
    target_balance = df[target].value_counts(dropna=False).to_dict()
    imbalance_ratio = None
    try:
        vals = list(target_balance.values())
        if len(vals) >= 2:
            imbalance_ratio = max(vals) / sum(vals)
    except Exception:
        pass

    total_nulls = int(df[feature_cols].isnull().sum().sum())
    null_cols = df[feature_cols].columns[df[feature_cols].isnull().any()].tolist()
    null_pct = {c: float(df[c].isnull().mean()) for c in null_cols}

    var = df[feature_cols].var(numeric_only=True)
    zero_variance = var[var == 0].index.tolist()
    near_zero_variance = var[var < 0.01].index.tolist()

    constants = [c for c in feature_cols if df[c].nunique(dropna=False) == 1]

    # Correlations
    numeric_feats = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    corr = df[numeric_feats].corr().abs()
    high_corr_pairs = _high_correlation_pairs(corr, thresh=0.95)

    pii_keywords = {"email", "phone", "name", "id_number", "ssn"}
    pii_risk = [c for c in df.columns if any(k in c.lower() for k in pii_keywords)]

    # Data sufficiency: rows/features ratio heuristic
    ratio = n_rows / max(1, feature_count)
    if ratio >= 50:
        suff = "sufficient"
    elif ratio >= 10:
        suff = "borderline"
    else:
        suff = "insufficient"

    # PREPROCESSING AUDIT
    missingness_pattern = {c: mcar_mnar_assessment(df, c, target) for c in null_cols}

    categorical_columns = _build_categorical_columns(df, feature_cols, explicit_rules)

    scaling_needed = []
    for c in numeric_feats:
        snum = pd.to_numeric(df[c], errors="coerce").dropna()
        if snum.empty:
            continue
        range_span = float(snum.quantile(0.95) - snum.quantile(0.05))
        scale_ratio = (
            float(snum.std(ddof=1) / max(snum.abs().median(), 1e-12))
            if len(snum) > 1
            else 0.0
        )
        if range_span > 1000 or scale_ratio > 10:
            scaling_needed.append(c)

    outlier_flags = {}
    for c in numeric_feats:
        outlier_flags[c] = _outlier_summary(df[c], len(df))

    # Standardisation verdict
    std_verdict = {
        "trees_ok": True,
        "linear_nn_need_scaling": len(scaling_needed) > 0,
        "recommendation": "Use tree ensembles with minimal scaling; apply scaling if training linear/NN models.",
    }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_quality": {
            "shape": {"rows": n_rows, "cols": n_cols, "feature_count": feature_count},
            "target": target,
            "target_balance": target_balance,
            "imbalance_ratio": imbalance_ratio,
            "total_nulls": total_nulls,
            "null_columns": null_cols,
            "null_pct": null_pct,
            "zero_variance": zero_variance,
            "near_zero_variance": near_zero_variance,
            "constant_features": constants,
            "high_correlation_pairs_count": len(high_corr_pairs),
            "high_correlation_pairs": high_corr_pairs,
            "pii_risk": pii_risk,
            "data_sufficiency": {"ratio": ratio, "verdict": suff},
        },
        "preprocessing_audit": {
            "missingness_pattern": missingness_pattern,
            "categorical_columns": categorical_columns,
            "scaling_needed": scaling_needed,
            "outlier_assessment": outlier_flags,
            "standardisation_verdict": std_verdict,
        },
    }

    # Write JSON report
    rep_path = reports_dir / "eda_report.json"
    rep_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Write human-readable summary
    md_lines = []
    md_lines.append(f"# EDA Summary — {datetime.now(timezone.utc).isoformat()}\n")
    md_lines.append(
        f"**Data shape**: {n_rows} rows, {n_cols} cols ({feature_count} features)"
    )
    md_lines.append(f"**Target**: {target} — distribution: {target_balance}")
    md_lines.append(f"**Total nulls**: {total_nulls}; null columns: {len(null_cols)}")
    md_lines.append(f"**Zero variance**: {zero_variance}")
    md_lines.append(f"**Near-zero variance**: {near_zero_variance}")
    md_lines.append(f"**Constant features**: {constants}")
    md_lines.append(f"**High-correlation pairs (>0.95)**: {len(high_corr_pairs)}")
    md_lines.append(f"**PII risk columns**: {pii_risk}")
    md_lines.append("\n## Preprocessing notes")
    md_lines.append(
        f"Missingness pattern examples (first 10): {dict(list(missingness_pattern.items())[:10])}"
    )
    md_lines.append(f"Categorical candidates: {categorical_columns}")
    md_lines.append(f"Scaling needed (range>1000): {scaling_needed}")
    md_lines.append(
        f"Outlier flags (features >5% outliers): {[k for k, v in outlier_flags.items() if v['flag']]} "
    )
    md_lines.append(f"Standardisation verdict: {std_verdict['recommendation']}")

    summary_path = reports_dir / "eda_summary.md"
    summary_path.write_text("\n".join(md_lines), encoding="utf-8")

    # Update SKILL_STATE.json using SkillStateStore (safe, validated)
    from zindian.state import SkillStateStore

    state_path = paths.state_path
    store = SkillStateStore(state_path)
    state = store.read()
    current_phase = state.get("dag_phase")
    allowed = current_phase in (None, "uninitialized", "phase_0_foundation") or (
        isinstance(current_phase, str) and current_phase.startswith("phase_1_")
    )

    mnar_cols = [c for c, pattern in missingness_pattern.items() if pattern == "MNAR"]
    mcar_cols = [c for c, pattern in missingness_pattern.items() if pattern == "MCAR"]

    eda_updates = {
        "eda_completed_at": datetime.now(timezone.utc).isoformat(),
        "dead_features": zero_variance,
        "high_corr_pairs_count": len(high_corr_pairs),
        "target_std": target_std,
        "mnar_columns": mnar_cols,
        "mcar_columns": mcar_cols,
    }

    updates: dict[str, Any] = {
        "eda": eda_updates,
    }
    if allowed:
        updates["dag_phase"] = "phase_1_eda_complete"
    try:
        store.update(**updates)

    except Exception:
        print("ERROR: failed to update SKILL_STATE.json after EDA")
        traceback.print_exc()
        raise

    # Print clean summary
    print("EDA complete — report written to:", rep_path)
    print("Summary written to:", summary_path)
    print("Zero-variance features:", zero_variance)
    print("Near-zero variance:", near_zero_variance)
    print("High-correlation pairs:", len(high_corr_pairs))


if __name__ == "__main__":
    run()
