"""
Skill 04 — EDA / Data Quality Audit

Generic, competition-agnostic EDA for pipeline stages 2 and 3.
Writes reports/eda_report.json and reports/eda_summary.md and updates SKILL_STATE.json.

Usage: python3 -m zindian.skills.skill_04_eda
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd

from zindian.paths import resolve_competition_paths, CompetitionPaths


def detect_target(paths: CompetitionPaths) -> str | None:
    # Try challenge_config.json -> SKILL_STATE.json -> heuristics
    cfg_path = paths.config_path
    state_path = paths.state_path
    target = None
    try:
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            # common key names
            for k in ("target", "label", "target_column", "output_column"):
                if k in cfg and cfg[k]:
                    return cfg[k]
    except Exception:
        pass
    try:
        if state_path.exists():
            st = json.loads(state_path.read_text(encoding="utf-8"))
            for k in ("target", "target_column", "primary_key", "label"):
                if k in st and st[k]:
                    return st[k]
    except Exception:
        pass
    return None


def mcar_mnar_assessment(df: pd.DataFrame, col: str, target: str) -> str:
    # If no nulls, return 'none'
    series = df[col]
    null_rate = series.isnull().mean()
    if null_rate == 0:
        return "none"
    # If correlation between null indicator and target > small threshold -> MNAR
    if target in df.columns:
        try:
            null_ind = series.isnull().astype(float)
            target_series = pd.to_numeric(df[target], errors="coerce").astype(float)
            corr = null_ind.corr(target_series)
            if pd.notna(corr) and abs(corr) >= 0.05:
                return "MNAR"
        except Exception:
            pass
    return "MCAR"


def run():
    paths = resolve_competition_paths(require_competition=True)
    competition_dir = paths.competition_dir
    if competition_dir is None:
        raise FileNotFoundError("resolve_competition_paths did not return a competition_dir")
    reports_dir = paths.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Prefer processed features file, fall back to raw training file
    proc_train = (paths.competition_dir / "data" / "processed" / "features_train.csv")
    raw_train = paths.data_raw_dir / "Training_Data.csv"

    if proc_train.exists():
        df = pd.read_csv(proc_train)
    else:
        if not raw_train.exists():
            raise FileNotFoundError(f"Training data not found at {proc_train} or {raw_train}")
        df = pd.read_csv(raw_train)

    # Detect target column
    target = detect_target(paths)
    if target is None or target not in df.columns:
        # Heuristic: look for binary-looking columns
        for c in df.columns:
            if df[c].dropna().isin([0, 1]).all() and df[c].nunique() <= 2:
                target = c
                break
    if target is None:
        # fallback to second column if can't detect
        target = df.columns[1] if len(df.columns) > 1 else df.columns[0]

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
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_corr_pairs = []
    thresh = 0.95
    for i in upper.columns:
        for j in upper.index:
            v = upper.loc[j, i]
            if pd.notna(v) and v > thresh:
                high_corr_pairs.append((j, i, float(v)))

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

    categorical_columns = []
    for c in feature_cols:
        if df[c].dtype == object:
            categorical_columns.append({"name": c, "cardinality": int(df[c].nunique()) , "encoding": "one-hot or ordinal"})
        elif pd.api.types.is_integer_dtype(df[c]) and df[c].nunique() < 20:
            categorical_columns.append({"name": c, "cardinality": int(df[c].nunique()), "encoding": "ordinal"})

    scaling_needed = []
    for c in numeric_feats:
        try:
            snum = pd.to_numeric(df[c], errors="coerce").dropna()
            if snum.empty:
                continue
            val = float(snum.max() - snum.min())
            if val > 1000:
                scaling_needed.append(c)
        except Exception:
            continue

    # Outlier via IQR
    outlier_flags = {}
    for c in numeric_feats:
        series = pd.to_numeric(df[c], errors="coerce").dropna()
        if series.empty:
            outlier_flags[c] = {"outlier_pct": 0.0, "flag": False}
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        outliers = ((series < low) | (series > high)).sum()
        pct = float(outliers / len(df))
        outlier_flags[c] = {"outlier_pct": pct, "flag": pct > 0.05}

    # Standardisation verdict
    std_verdict = {
        "trees_ok": True,
        "linear_nn_need_scaling": len(scaling_needed) > 0,
        "recommendation": "Use tree ensembles with minimal scaling; apply scaling if training linear/NN models."
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
        }
    }

    # Write JSON report
    rep_path = reports_dir / "eda_report.json"
    rep_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Write human-readable summary
    md_lines = []
    md_lines.append(f"# EDA Summary — {datetime.now(timezone.utc).isoformat()}\n")
    md_lines.append(f"**Data shape**: {n_rows} rows, {n_cols} cols ({feature_count} features)")
    md_lines.append(f"**Target**: {target} — distribution: {target_balance}")
    md_lines.append(f"**Total nulls**: {total_nulls}; null columns: {len(null_cols)}")
    md_lines.append(f"**Zero variance**: {zero_variance}")
    md_lines.append(f"**Near-zero variance**: {near_zero_variance}")
    md_lines.append(f"**Constant features**: {constants}")
    md_lines.append(f"**High-correlation pairs (>0.95)**: {len(high_corr_pairs)}")
    md_lines.append(f"**PII risk columns**: {pii_risk}")
    md_lines.append("\n## Preprocessing notes")
    md_lines.append(f"Missingness pattern examples (first 10): {dict(list(missingness_pattern.items())[:10])}")
    md_lines.append(f"Categorical candidates: {categorical_columns}")
    md_lines.append(f"Scaling needed (range>1000): {scaling_needed}")
    md_lines.append(f"Outlier flags (features >5% outliers): {[k for k,v in outlier_flags.items() if v['flag']]} ")
    md_lines.append(f"Standardisation verdict: {std_verdict['recommendation']}")

    summary_path = reports_dir / "eda_summary.md"
    summary_path.write_text("\n".join(md_lines), encoding="utf-8")

    # Update SKILL_STATE.json safely
    try:
        state_path = paths.state_path
        state = {}
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
        current_phase = state.get("dag_phase")
        allowed = current_phase in (None, "phase_1_integrity", "phase_1_eda")
        if allowed:
            state["dag_phase"] = "phase_1_eda_complete"
        state["eda_completed_at"] = datetime.now(timezone.utc).isoformat()
        state["dead_features"] = zero_variance
        state["high_corr_pairs_count"] = len(high_corr_pairs)
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception:
        pass

    # Print clean summary
    print("EDA complete — report written to:", rep_path)
    print("Summary written to:", summary_path)
    print("Zero-variance features:", zero_variance)
    print("Near-zero variance:", near_zero_variance)
    print("High-correlation pairs:", len(high_corr_pairs))


if __name__ == "__main__":
    run()
