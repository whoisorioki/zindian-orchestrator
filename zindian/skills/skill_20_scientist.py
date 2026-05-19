"""
Skill 20 — The Scientist
Synthesizes domain hypotheses + prior art, then runs two-stage validation.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import google.genai as genai
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from zindian.paths import resolve_competition_paths
from zindian.state import SkillStateStore

try:
    from zindian.skills.skill_07_features import TC_VARIABLES
except Exception:
    TC_VARIABLES = [
        "aet", "def", "pdsi", "pet", "ppt",
        "q", "soil", "srad", "swe",
        "tmax", "tmin", "vap", "vpd",
    ]

CLIENT = genai.Client()
MODEL_NAME = "gemini-2.5-flash"
TARGET_COL_CANDIDATES = ["Occurrence Status", "target", "label", "y"]


def get_available_columns() -> list[str]:
    cols = []
    for var in TC_VARIABLES:
        for stat in ["mean", "std", "min", "max"]:
            cols.append(f"{var}_{stat}")
        cols.append(f"{var}_last3mo_mean")
        cols.append(f"{var}_range")
        cols.append(f"{var}_cv")
    return cols


SCIENTIST_SYSTEM = """
You are a feature engineering scientist for species distribution modelling.
You will receive ecological hypotheses and ML prior art, plus a list of AVAILABLE feature columns.

Your job: identify which features or transformations are most likely to predict frog occurrence in southeastern Australia,
using ONLY the available columns provided.

Output a JSON array. Each element must follow this exact schema:
{
    "hypothesis_id":     "hyp_001",
    "source_paper_id":   "paper_id_or_domain_knowledge",
    "rationale":         "one sentence explaining the ecological signal",
    "feature_columns":   ["exact_column_name_from_available_columns"],
    "transformation":    "raw | interaction_product | ratio | threshold_binary | polynomial_2",
    "expected_signal":   "positive | negative | nonlinear",
    "confidence":        0.85
}

Hard rules:
- feature_columns values must be from the available_columns list ONLY.
- No spatial coordinates (lat, lon, latitude, longitude).
- No external data sources.
- Output only valid JSON — no prose, no markdown fences.
""".strip()


def load_json(path: Path) -> dict | list:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _normalize_json_block(text: str) -> str:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.replace("```json", "").replace("```", "").strip()
    return raw


def _hypothesis_signature(hypothesis: dict[str, Any]) -> str:
    columns = sorted(hypothesis.get("feature_columns", []))
    return "|".join(columns) + "::" + str(hypothesis.get("transformation", ""))


def load_failed_ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        return data.get("entries", [])
    if isinstance(data, list):
        return data
    return []


def save_failed_ledger(path: Path, entries: list[dict]) -> None:
    path.write_text(json.dumps({"entries": entries}, indent=2), encoding="utf-8")


def resolve_target_column(frame: pd.DataFrame) -> str:
    for column in TARGET_COL_CANDIDATES:
        if column in frame.columns:
            return column
    raise ValueError(f"No target column found. Tried: {TARGET_COL_CANDIDATES}")


def static_validate_hypothesis(hypothesis: dict[str, Any], available_columns: set[str]) -> tuple[bool, str]:
    feature_columns = hypothesis.get("feature_columns", [])
    if not feature_columns:
        return False, "missing feature_columns"
    if not isinstance(feature_columns, list):
        return False, "feature_columns must be a list"
    if any(col not in available_columns for col in feature_columns):
        bad = [col for col in feature_columns if col not in available_columns]
        return False, f"unknown columns: {bad}"
    if len(set(feature_columns)) != len(feature_columns):
        return False, "duplicate feature columns"
    return True, "pass"


def empirical_validate_hypothesis(
    hypothesis: dict[str, Any],
    train_frame: pd.DataFrame,
) -> tuple[bool, str]:
    feature_columns = hypothesis.get("feature_columns", [])
    target_col = resolve_target_column(train_frame)

    subset = train_frame[feature_columns].replace([np.inf, -np.inf], np.nan)
    if subset.isna().all().all():
        return False, "all values are missing"

    # Stage 2a: mutual information must be positive.
    filled = subset.fillna(subset.median(numeric_only=True)).fillna(0.0)
    mi_scores = mutual_info_classif(filled, train_frame[target_col].astype(int), random_state=42)
    if float(np.nanmean(mi_scores)) <= 0.0:
        return False, "mutual information <= 0"

    # Stage 2b: a fast LightGBM fit should produce non-zero gain.
    model = lgb.LGBMClassifier(
        n_estimators=25,
        learning_rate=0.1,
        num_leaves=8,
        max_depth=3,
        random_state=42,
        verbosity=-1,
    )
    model.fit(filled, train_frame[target_col].astype(int))
    gain = model.booster_.feature_importance(importance_type="gain")
    if float(np.sum(gain)) <= 0.0:
        return False, "lightgbm gain <= 0"

    return True, "pass"


def validate_hypotheses(
    hypotheses: list[dict[str, Any]],
    feature_frame: pd.DataFrame,
    failed_ledger: list[dict],
) -> tuple[list[dict], list[dict]]:
    available_columns = set(feature_frame.columns) - set(TARGET_COL_CANDIDATES)
    failed_signatures = {
        entry.get("signature") for entry in failed_ledger if entry.get("do_not_retry")
    }

    kept: list[dict] = []
    failed: list[dict] = []
    for hypothesis in hypotheses:
        signature = _hypothesis_signature(hypothesis)
        if signature in failed_signatures:
            failed.append({
                **hypothesis,
                "validation_status": "blocked",
                "do_not_retry": True,
                "failure_stage": "ledger",
                "failure_reason": "signature previously blocked",
                "signature": signature,
            })
            continue

        ok, reason = static_validate_hypothesis(hypothesis, available_columns)
        if not ok:
            failed.append({
                **hypothesis,
                "validation_status": "failed",
                "do_not_retry": True,
                "failure_stage": "static",
                "failure_reason": reason,
                "signature": signature,
            })
            continue

        ok, reason = empirical_validate_hypothesis(hypothesis, feature_frame)
        if not ok:
            failed.append({
                **hypothesis,
                "validation_status": "failed",
                "do_not_retry": True,
                "failure_stage": "empirical",
                "failure_reason": reason,
                "signature": signature,
            })
            continue

        kept.append({
            **hypothesis,
            "validation_status": "passed",
            "do_not_retry": False,
            "signature": signature,
        })

    return kept, failed


def run_scientist(
    hypotheses_path: str,
    priorart_path: str,
    hypothesis_path: str,
    failed_hypotheses_path: str | None = None,
) -> list[dict]:
    paths = resolve_competition_paths()
    state_store = SkillStateStore(paths.state_path)

    domain_hypotheses = load_json(Path(hypotheses_path))
    prior_art = load_json(Path(priorart_path))

    available_columns = get_available_columns()

    user_prompt = f"""
System Guardrails and Constraints:
{SCIENTIST_SYSTEM}

Available feature columns ({len(available_columns)} total):
{json.dumps(available_columns, indent=2)}

Domain hypotheses:
{json.dumps(domain_hypotheses, indent=2)}

ML prior art:
{json.dumps(prior_art, indent=2)}

Generate feature engineering hypotheses as a raw JSON array now.
""".strip()

    print(f"[Scientist] Submitting text blocks to local/cloud {MODEL_NAME} engine...")
    response = CLIENT.models.generate_content(
        model=MODEL_NAME,
        contents=user_prompt,
        config={"response_mime_type": "application/json"},
    )

    raw = (response.text or "").strip()
    if not raw:
        raise ValueError("No text content returned by Gemini response")

    hypotheses = json.loads(_normalize_json_block(raw))
    if not isinstance(hypotheses, list):
        raise ValueError("Scientist model did not return a JSON array")

    reports_dir = paths.competition_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    feature_frame_path = paths.competition_dir / "data/processed/features_train.csv"
    if not feature_frame_path.exists():
        raise FileNotFoundError(f"Missing feature matrix: {feature_frame_path}")
    feature_frame = pd.read_csv(feature_frame_path)

    failed_path = Path(failed_hypotheses_path) if failed_hypotheses_path else (reports_dir / "failed_hypotheses.json")
    failed_ledger = load_failed_ledger(failed_path)

    validated, failed = validate_hypotheses(hypotheses, feature_frame, failed_ledger)

    Path(hypothesis_path).write_text(json.dumps(validated, indent=2), encoding="utf-8")

    combined_failed = failed_ledger + [
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **entry,
        }
        for entry in failed
    ]
    save_failed_ledger(failed_path, combined_failed)

    state_store.update(
        scientist_last_run=json.dumps({
            "status": "OK",
            "validated": len(validated),
            "failed": len(failed),
        }),
        last_updated=datetime.now(timezone.utc).isoformat(),
    )

    print(f"[Scientist] {len(validated)}/{len(hypotheses)} hypotheses passed validation → {hypothesis_path}")
    print(f"[Scientist] Failed hypotheses ledger → {failed_path}")
    return validated


if __name__ == "__main__":
    run_scientist(
        hypotheses_path="competitions/ey-frogs/reports/domain_hypotheses.json",
        priorart_path="competitions/ey-frogs/reports/ml_priorart.json",
        hypothesis_path="competitions/ey-frogs/reports/validated_hypotheses.json",
        failed_hypotheses_path="competitions/ey-frogs/reports/failed_hypotheses.json",
    )
