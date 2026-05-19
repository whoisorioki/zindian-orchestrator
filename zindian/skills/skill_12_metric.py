"""Skill 12 — Metric Trade-off Analysis.

Safely scans existing out-of-fold artifacts, computes the best thresholded
F1 for each OOF file, and summarizes the ranking. This skill does not train
models and does not submit anything.

Outputs:
  - competitions/<slug>/reports/metric_scan.json
  - competitions/<slug>/reports/metric_scan.md
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.metrics import f1_score

from zindian.config import ChallengeConfig
from zindian.paths import resolve_competition_paths


@dataclass(frozen=True)
class MetricScanResult:
    file_name: str
    best_f1: float
    best_threshold: float
    generalization_gap: float
    rows: int
    columns: list[str]


def _detect_target(frame: pd.DataFrame, config: ChallengeConfig) -> str:
    for key in ("target_column", "target", "label", "output_column"):
        value = config.get(key)
        if isinstance(value, str) and value and value in frame.columns:
            return value
    for candidate in ("Occurrence Status", "target", "label"):
        if candidate in frame.columns:
            return candidate
    for column in frame.columns:
        series = frame[column].dropna()
        if not series.empty and series.isin([0, 1]).all() and series.nunique() <= 2:
            return column
    raise ValueError("Could not infer target column")


def _scan_thresholds(y_true: pd.Series, probs: pd.Series) -> tuple[float, float, float]:
    best_f1 = -1.0
    best_threshold = 0.5
    threshold_scores: list[tuple[float, float]] = []
    for threshold in [i / 100 for i in range(1, 100)]:
        preds = (probs >= threshold).astype(int)
        score = float(f1_score(y_true, preds))
        threshold_scores.append((threshold, score))
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    # Generalization proxy: score stability around the winning threshold.
    # A smaller gap means the OOF score is less sensitive to a small threshold shift.
    local_band = [score for threshold, score in threshold_scores if abs(threshold - best_threshold) <= 0.05]
    if not local_band:
        local_band = [score for _, score in threshold_scores]
    generalization_gap = float(best_f1 - (sum(local_band) / len(local_band)))
    return best_f1, best_threshold, generalization_gap


def _summarize_results(results: Iterable[MetricScanResult]) -> list[str]:
    lines = [
        "# Metric Scan Summary",
        "",
        f"**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "| File | Best F1 | Best Threshold | Gen Gap | Rows | Columns |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for result in results:
        lines.append(
            f"| {result.file_name} | {result.best_f1:.6f} | {result.best_threshold:.2f} | {result.generalization_gap:+.6f} | {result.rows} | {', '.join(result.columns)} |"
        )
    return lines


def run() -> dict:
    paths = resolve_competition_paths(require_competition=True)
    if paths.competition_dir is None:
        raise FileNotFoundError("Competition directory could not be resolved")

    config = ChallengeConfig.load()
    train_path = paths.data_raw_dir / "Training_Data.csv"
    processed_dir = paths.data_processed_dir

    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")

    train = pd.read_csv(train_path)
    target = _detect_target(train, config)

    results: list[MetricScanResult] = []
    for oof_path in sorted(processed_dir.glob("oof_variant-*.csv")):
        oof = pd.read_csv(oof_path)
        merged = oof.merge(train[["ID", target]], on="ID", how="left")
        if "oof_prob" not in merged.columns:
            raise ValueError(f"{oof_path.name} is missing required 'oof_prob' column")
        best_f1, best_threshold, generalization_gap = _scan_thresholds(merged[target], merged["oof_prob"])
        results.append(
            MetricScanResult(
                file_name=oof_path.name,
                best_f1=best_f1,
                best_threshold=best_threshold,
                generalization_gap=generalization_gap,
                rows=len(merged),
                columns=list(oof.columns),
            )
        )

    results = sorted(results, key=lambda item: item.best_f1, reverse=True)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "competition": config.slug,
        "target_column": target,
        "metric": "f1_score",
        "use_probabilities": bool(config.use_probabilities),
        "results": [
            {
                "file_name": result.file_name,
                "best_f1": result.best_f1,
                "best_threshold": result.best_threshold,
                "generalization_gap": result.generalization_gap,
                "rows": result.rows,
                "columns": result.columns,
            }
            for result in results
        ],
        "best_result": None if not results else {
            "file_name": results[0].file_name,
            "best_f1": results[0].best_f1,
            "best_threshold": results[0].best_threshold,
            "generalization_gap": results[0].generalization_gap,
            "rows": results[0].rows,
            "columns": results[0].columns,
        },
    }

    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = paths.reports_dir / "metric_scan.json"
    md_path = paths.reports_dir / "metric_scan.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(_summarize_results(results)), encoding="utf-8")

    print(f"✅ Metric scan written → {json_path}")
    print(f"✅ Metric summary written → {md_path}")
    if results:
        top = results[0]
        print(f"Best OOF file : {top.file_name}")
        print(f"Best F1       : {top.best_f1:.6f}")
        print(f"Best threshold: {top.best_threshold:.2f}")
        print(f"Gen gap       : {top.generalization_gap:+.6f}")

    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
