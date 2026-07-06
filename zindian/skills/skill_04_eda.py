"""
Skill 04 — EDA / Data Quality Audit

Generic, competition-agnostic EDA for pipeline stages 2 and 3.
Writes reports/eda_report.json and reports/eda_summary.md and updates SKILL_STATE.json.

Usage: python3 -m zindian.skills.skill_04_eda
"""

from __future__ import annotations
import tabula.skill_state_autopatch  # noqa

import json
import traceback
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from typing import Any, Mapping, cast

from zindian.paths import resolve_competition_paths, CompetitionPaths


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return data


def detect_target(paths: CompetitionPaths) -> str | list[str]:
    """Detect target column(s). Returns list for multi-target, string for single-target."""
    cfg = _load_json_object(paths.config_path)

    # A11: Check for multi-target config first
    target_config = cfg.get("target_config")
    if target_config and isinstance(target_config, dict):
        targets = target_config.get("targets", [])
        if targets:
            return [t["name"] for t in targets]

    # Fallback to legacy single-target keys
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
        "Unable to resolve target column from challenge_config.json or SKILL_STATE.json"
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
    numeric_raw = pd.to_numeric(series, errors="coerce")
    numeric_series = cast(pd.Series, numeric_raw).dropna()
    numeric_values = numeric_series.astype(float).to_numpy()
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
        column_series = cast(pd.Series, df[column])
        column_rules = explicit_rules.get(column)
        if (
            pd.api.types.is_object_dtype(column_series)
            or pd.api.types.is_string_dtype(column_series)
            or str(column_series.dtype) == "category"
        ):
            categorical_columns.append(
                {
                    "name": column,
                    "cardinality": int(column_series.nunique(dropna=False)),
                    "encoding": column_rules or "one-hot or ordinal",
                }
            )
        elif column_rules is not None:
            categorical_columns.append(
                {
                    "name": column,
                    "cardinality": int(column_series.nunique(dropna=False)),
                    "encoding": column_rules,
                }
            )
    return categorical_columns


def mcar_mnar_assessment(df: pd.DataFrame, col: str, targets: list[str]) -> str:
    """Multi-target MNAR assessment. If missingness correlates with ANY target, flag as MNAR."""
    series = cast(pd.Series, df[col])
    null_rate = series.isnull().mean()
    if null_rate == 0:
        return "none"

    # Check correlation with each target
    null_ind = series.isnull().astype(float)
    for target in targets:
        if target in df.columns:
            target_raw = pd.to_numeric(cast(pd.Series, df[target]), errors="coerce")
            target_series = cast(pd.Series, target_raw).astype(float)
            corr = null_ind.corr(target_series)
            if pd.notna(corr) and abs(corr) >= 0.05:
                return "MNAR"  # Correlated with at least one target
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
        # features_train.csv intentionally excludes targets (dropped by the
        # extractor plugin).  Re-attach target column(s) from the raw file so
        # EDA can compute balance, std, and correlation metrics.
        if raw_train.exists():
            raw_df = pd.read_csv(raw_train)
            cfg_tmp = _load_json_object(paths.config_path)
            _target_cfg = cfg_tmp.get("target_config", {}) or {}
            _targets_to_attach = [t["name"] for t in _target_cfg.get("targets", [])]
            if not _targets_to_attach:
                # Fallback: single-target from top-level key
                _single = cfg_tmp.get("target_col")
                if _single:
                    _targets_to_attach = [_single]
            _cols_to_attach = [
                _t
                for _t in _targets_to_attach
                if _t in raw_df.columns and _t not in df.columns
            ]
            if _cols_to_attach and len(raw_df) == len(df):
                df = pd.concat(
                    [df, raw_df[_cols_to_attach].reset_index(drop=True)],
                    axis=1,
                )
            elif _cols_to_attach:
                # Row counts differ — cannot align; fall back to raw
                df = raw_df
    else:
        if not raw_train.exists():
            raise FileNotFoundError(
                f"Training data not found at {proc_train} or {raw_train}"
            )
        df = pd.read_csv(raw_train)

    config_data = _load_eda_config(paths)
    explicit_rules = _extract_explicit_encoding_rules(config_data)

    # Detect target column(s)
    target = detect_target(paths)
    is_multi_target = isinstance(target, list)
    targets: list[str] = list(target) if is_multi_target else [str(target)]

    # Validate all targets exist
    for t in targets:
        if t not in df.columns:
            raise ValueError(f"Target column '{t}' not present in training data")

    # Per-target standard deviation for regression targets
    target_std_dict = {}
    cfg = _load_json_object(paths.config_path)
    target_config = cfg.get("target_config", {})

    if is_multi_target and target_config.get("targets"):
        for target_spec in target_config["targets"]:
            t_name = target_spec["name"]
            if target_spec["task_type"] == "regression":
                target_std_dict[f"{t_name}_std"] = float(
                    np.std(np.asarray(df[t_name].values, dtype=float), ddof=1)
                )
    else:
        # Single-target: use legacy target_std
        target_vals = np.asarray(df[targets[0]].values, dtype=float)
        target_std_dict["target_std"] = float(np.std(target_vals, ddof=1))

    # Exclude ID, coords, and all targets
    exclude_lower = {"id", "id_number", "latitude", "longitude"}
    exclude_lower.update(t.lower() for t in targets)
    feature_cols = [c for c in df.columns if c.lower() not in exclude_lower]

    # DATA QUALITY (use first target for balance metrics)
    primary_target = targets[0]
    n_rows, n_cols = df.shape
    feature_count = len(feature_cols)
    target_balance = df[primary_target].value_counts(dropna=False).to_dict()
    imbalance_ratio = None
    try:
        vals = list(target_balance.values())
        if len(vals) >= 2:
            imbalance_ratio = max(vals) / sum(vals)
    except Exception:
        pass

    feature_frame = cast(pd.DataFrame, df[feature_cols])
    total_nulls = int(feature_frame.isnull().sum().sum())
    null_cols = feature_frame.columns[feature_frame.isnull().any()].tolist()
    null_pct = {c: float(cast(pd.Series, df[c]).isnull().mean()) for c in null_cols}

    var_series = cast(pd.Series, feature_frame.var(numeric_only=True))
    zero_variance = [name for name, value in var_series.items() if value == 0]
    near_zero_variance = [name for name, value in var_series.items() if value < 0.01]

    constants = [
        c for c in feature_cols if cast(pd.Series, df[c]).nunique(dropna=False) == 1
    ]

    # Correlations
    numeric_feats = feature_frame.select_dtypes(include=[np.number]).columns.tolist()
    numeric_frame = cast(pd.DataFrame, feature_frame[numeric_feats])
    corr = numeric_frame.corr().abs()
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

    # PREPROCESSING AUDIT (multi-target MNAR assessment)
    missingness_pattern = {c: mcar_mnar_assessment(df, c, targets) for c in null_cols}

    categorical_columns = _build_categorical_columns(df, feature_cols, explicit_rules)

    scaling_needed = []
    for c in numeric_feats:
        snum_raw = pd.to_numeric(df[c], errors="coerce")
        snum = cast(pd.Series, snum_raw).dropna()
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
        outlier_flags[c] = _outlier_summary(cast(pd.Series, df[c]), len(df))

    # Standardisation verdict
    std_verdict = {
        "trees_ok": True,
        "linear_nn_need_scaling": len(scaling_needed) > 0,
        "recommendation": "Use tree ensembles with minimal scaling; apply scaling if training linear/NN models.",
    }

    # ── EDA Enhancements: band-aware diagnostics ─────────────────────
    # Discover spectral/temporal bands from column naming pattern.
    # Pattern: BAND_MM where MM is a 2-digit month (01–12).
    # Example: VH_01, VV_03, blue_12 → bands: VH, VV, blue
    detected_bands: list[str] = []
    _seen: set[str] = set()
    for c in feature_cols:
        if "_" in c:
            parts = c.split("_")
            candidate = parts[0]
            month = parts[-1]
            if month.isdigit() and candidate not in _seen:
                detected_bands.append(candidate)
                _seen.add(candidate)

    # ── 2a: Per-band summary statistics ──────────────────────────────
    band_summary_stats: dict[str, dict[str, float]] = {}
    for band in detected_bands:
        band_cols = [c for c in feature_cols if c.startswith(band + "_")]
        if not band_cols:
            continue
        vals = cast(pd.DataFrame, df[band_cols]).to_numpy(dtype=float)
        band_summary_stats[band] = {
            "mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals, ddof=1)),
            "min": float(np.nanmin(vals)),
            "max": float(np.nanmax(vals)),
        }

    # ── 2b: Seasonal amplitude per band ──────────────────────────────
    seasonal_amplitude: dict[str, float] = {}
    for band in detected_bands:
        band_cols = sorted(
            [c for c in feature_cols if c.startswith(band + "_")],
            key=lambda c: int(c.split("_")[-1]),
        )
        if len(band_cols) < 2:
            continue
        monthly_vals: np.ndarray = np.asarray(
            cast(pd.Series, df[band_cols].mean(axis=0)).to_numpy(dtype=float),
            dtype=float,
        )
        seasonal_amplitude[band] = float(np.max(monthly_vals) - np.min(monthly_vals))

    # ── 2c: Temporal trend analysis ──────────────────────────────────
    # Structural feature — computed from feature columns, not targets.
    # Per the two-mode contract, structural features may be computed on
    # the full dataset at any time (no fold-restriction needed).
    temporal_trends: dict[str, dict[str, list[float]]] = {}
    for band in detected_bands:
        band_cols = sorted(
            [c for c in feature_cols if c.startswith(band + "_")],
            key=lambda c: int(c.split("_")[-1]),
        )
        if len(band_cols) < 2:
            continue
        trend_vals: np.ndarray = np.asarray(
            cast(pd.Series, df[band_cols].mean(axis=0)).to_numpy(dtype=float),
            dtype=float,
        )
        monthly_list: list[float] = trend_vals.tolist()
        mom_delta: list[float] = (trend_vals[1:] - trend_vals[:-1]).tolist()
        temporal_trends[band] = {
            "monthly_means": monthly_list,
            "month_over_month_delta": mom_delta,
        }

    # ── 2d: Target correlation per feature ───────────────────────────
    target_correlation_per_feature: dict[str, float] = {}
    if primary_target in df.columns:
        y_raw = pd.to_numeric(cast(pd.Series, df[primary_target]), errors="coerce")
        y_vals = cast(pd.Series, y_raw).to_numpy(dtype=float)
        for c in numeric_feats:
            try:
                x_raw = pd.to_numeric(cast(pd.Series, df[c]), errors="coerce")
                x_vals = cast(pd.Series, x_raw).to_numpy(dtype=float)
                mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
                if mask.sum() > 2:
                    xy = np.corrcoef(x_vals[mask], y_vals[mask])
                    corr_val = float(xy[0, 1])
                    if not np.isnan(corr_val):
                        target_correlation_per_feature[c] = corr_val
            except Exception:
                pass

    # ── 2e: Class-separability index ─────────────────────────────────
    # Diagnostic only — single-feature decision stumps used as a ranking
    # heuristic during EDA. This is NOT model selection (no feature is
    # chosen or tuned), NOT AutoML (no hyperparameter search), and NOT
    # used by any downstream skill to prune features. It is a pure
    # characterisation step analogous to computing correlation or
    # variance — just a more discriminative lens.
    class_separability_index: dict[str, float] = {}
    seed = int(cfg.get("reproducibility", {}).get("seed", 42))
    if primary_target in df.columns and len(numeric_feats) > 0:
        try:
            from sklearn.tree import DecisionTreeClassifier as _DTC
            from sklearn.metrics import f1_score as _f1

            y_sep = df[primary_target]
            for c in numeric_feats:
                try:
                    X_sep = df[[c]].fillna(df[c].median())
                    stump = _DTC(max_depth=1, random_state=seed)
                    stump.fit(X_sep, y_sep)
                    preds = stump.predict(X_sep)
                    class_separability_index[c] = float(_f1(y_sep, preds))
                except Exception:
                    pass
        except ImportError:
            pass  # scikit-learn not available — skip; not a pipeline halt

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_quality": {
            "shape": {"rows": n_rows, "cols": n_cols, "feature_count": feature_count},
            "target": targets if is_multi_target else targets[0],
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
    md_lines.append(
        f"**Target**: {targets if is_multi_target else targets[0]} — distribution: {target_balance}"
    )
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
        **target_std_dict,  # Per-target std for regression
        "mnar_columns": mnar_cols,
        "mcar_columns": mcar_cols,
        # ── Phase 1 improvement plan: 5 band-aware diagnostics ──
        "band_summary_stats": band_summary_stats,
        "seasonal_amplitude": seasonal_amplitude,
        "temporal_trends": temporal_trends,
        "target_correlation_per_feature": target_correlation_per_feature,
        "class_separability_index": class_separability_index,
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
