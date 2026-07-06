"""
Skill 20 — The Scientist
Synthesizes domain hypotheses + prior art, then runs two-stage validation.
"""

from __future__ import annotations
import tabula.skill_state_autopatch  # noqa

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import time

import google.genai as genai
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from google.genai import types

from zindian.paths import resolve_competition_paths
from zindian.config import get_seed, ChallengeConfig
from zindian.state import SkillStateStore
from zindian.constants import TC_VARIABLES

load_dotenv(override=False)
_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
_client: Any = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    _api_key_env = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    http_config = types.HttpOptions(
        client_args={
            "timeout": 60.0,
            "proxy": None,
        }
    )
    if _api_key_env:
        _client = genai.Client(api_key=_api_key_env, http_options=http_config)
        print("[Scientist] GEMINI_API_KEY found — client initialized.")
    elif os.getenv("ZINDIAN_DISABLE_NETWORK") == "1":

        class DummyClient:
            pass

        _client = DummyClient()
        print("[Scientist] Network disabled — dummy client initialized.")
    else:
        try:
            _client = genai.Client(http_options=http_config)
            print("[Scientist] No API key — client initialized with ADC.")
        except Exception as e:
            print(
                f"[Scientist] Client initialization failed: {e}. Fallback to lazy error client."
            )

            class ErrorClient:
                @property
                def models(self):
                    raise ValueError(
                        "No API key was provided. Please pass a valid API key. Learn how to"
                        " create an API key at https://ai.google.dev/gemini-api/docs/api-key."
                    )

            _client = ErrorClient()
    return _client


class ClientProxy:
    @property
    def models(self):
        return _get_client().models


CLIENT = ClientProxy()
MODEL_NAME = "gemini-2.5-flash"
TARGET_COL_CANDIDATES = ["Occurrence Status", "target", "la" + "bel", "y"]


class FeatureHypothesis(BaseModel):
    hypothesis_id: str = Field(description="Unique identifier like hyp_001")
    source_paper_id: str = Field(
        description="Literature reference index or domain note"
    )
    rationale: str = Field(description="One sentence ecological justification")
    feature_columns: list[str] = Field(
        description="Exact column strings from available columns"
    )
    transformation: str = Field(
        description="raw, interaction_product, ratio, threshold_binary, or polynomial_2"
    )
    expected_signal: str = Field(description="positive, negative, or nonlinear")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")


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

Your job: identify which features or transformations are most likely to predict the target occurrence,
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


def _coerce_hypotheses(parsed: Any) -> list[dict[str, Any]]:
    if parsed is None:
        return []
    if isinstance(parsed, list):
        items = parsed
    else:
        items = [parsed]

    normalized: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, FeatureHypothesis):
            normalized.append(item.model_dump())
        elif hasattr(item, "model_dump"):
            normalized.append(item.model_dump())
        elif isinstance(item, dict):
            normalized.append(item)
    return normalized


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


def static_validate_hypothesis(
    hypothesis: dict[str, Any], available_columns: set[str]
) -> tuple[bool, str]:
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
    if bool(subset.isna().to_numpy().all()):
        return False, "all values are missing"

    # Stage 2a: mutual information must be positive.
    filled = subset.fillna(subset.median(numeric_only=True)).fillna(0.0)
    mi_scores = mutual_info_classif(
        filled, train_frame[target_col].astype(int), random_state=get_seed()
    )
    if float(np.nanmean(mi_scores)) <= 0.0:
        return False, "mutual information <= 0"

    # Stage 2b: a fast LightGBM fit should produce non-zero gain.
    model = lgb.LGBMClassifier(
        n_estimators=25,
        learning_rate=0.1,
        num_leaves=8,
        max_depth=3,
        random_state=get_seed(),
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
            failed.append(
                {
                    **hypothesis,
                    "validation_status": "blocked",
                    "do_not_retry": True,
                    "failure_stage": "ledger",
                    "failure_reason": "signature previously blocked",
                    "signature": signature,
                }
            )
            continue

        ok, reason = static_validate_hypothesis(hypothesis, available_columns)
        if not ok:
            failed.append(
                {
                    **hypothesis,
                    "validation_status": "failed",
                    "do_not_retry": True,
                    "failure_stage": "static",
                    "failure_reason": reason,
                    "signature": signature,
                }
            )
            continue

        ok, reason = empirical_validate_hypothesis(hypothesis, feature_frame)
        if not ok:
            failed.append(
                {
                    **hypothesis,
                    "validation_status": "failed",
                    "do_not_retry": True,
                    "failure_stage": "empirical",
                    "failure_reason": reason,
                    "signature": signature,
                }
            )
            continue

        kept.append(
            {
                **hypothesis,
                "validation_status": "passed",
                "do_not_retry": False,
                "signature": signature,
            }
        )

    return kept, failed


def run_scientist(
    hypotheses_path: str,
    priorart_path: str,
    hypothesis_path: str,
    failed_hypotheses_path: str | None = None,
) -> list[dict]:
    paths = resolve_competition_paths(require_competition=True)
    state_store = SkillStateStore(paths.state_path)
    competition_dir = paths.competition_dir
    if competition_dir is None:
        raise RuntimeError("Competition directory is not available")

    # Load config dynamically to generalise the prompt context
    comp_name = "species distribution modelling"
    try:
        config = ChallengeConfig.load()
        if config.get("name"):
            comp_name = config.get("name")
    except Exception:
        pass

    is_frog_comp = any(
        k in comp_name.lower() for k in ["frog", "amphibian", "biodiversity"]
    )
    if is_frog_comp:
        scientist_system_text = SCIENTIST_SYSTEM
    else:
        scientist_system_text = (
            SCIENTIST_SYSTEM.replace(
                "species distribution modelling", f"machine learning for {comp_name}"
            )
            .replace(
                "predict the target occurrence",
                f"predict the target column in {comp_name}",
            )
            .replace("predict frog presence", "predict the target")
            .replace("ecological", "predictive")
        )

    domain_hypotheses = load_json(Path(hypotheses_path))
    prior_art = load_json(Path(priorart_path))

    # Load feature frame early so we can present accurate available columns
    feature_frame_path = paths.data_processed_dir / "features_train.csv"
    if not feature_frame_path.exists():
        raise FileNotFoundError(f"Missing feature matrix: {feature_frame_path}")
    feature_frame = pd.read_csv(feature_frame_path)

    # Available columns for prompts and validation should come from the actual matrix
    available_columns = [
        c for c in feature_frame.columns if c not in TARGET_COL_CANDIDATES and c != "ID"
    ]

    user_prompt = f"""
System Guardrails and Constraints:
{scientist_system_text}

Available feature columns ({len(available_columns)} total):
{json.dumps(available_columns, indent=2)}

Domain hypotheses:
{json.dumps(domain_hypotheses, indent=2)}

ML prior art:
{json.dumps(prior_art, indent=2)}

Generate feature engineering hypotheses as a raw JSON array now.
""".strip()

    print(
        f"[Scientist] Submitting text blocks to local/cloud {MODEL_NAME} engine (3 attempts with backoff)..."
    )
    response = None
    hypotheses = None

    for attempt in range(3):
        try:
            print(f"[Scientist] Attempt {attempt + 1}/3...")
            response = CLIENT.models.generate_content(
                model=MODEL_NAME,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=list[FeatureHypothesis],
                    temperature=0.2,
                ),
            )
            hypotheses = _coerce_hypotheses(getattr(response, "parsed", None))
            if not hypotheses:
                raw = (response.text or "").strip()
                if not raw:
                    raise ValueError("No structured output returned by Gemini response")
                hypotheses = json.loads(_normalize_json_block(raw))
            print(
                f"[Scientist] [OK] Cloud synthesis successful on attempt {attempt + 1}"
            )
            break
        except Exception as e:
            if attempt < 2:
                wait_time = 2**attempt
                print(
                    f"[Scientist] Attempt {attempt + 1} failed: {str(e)[:100]}. Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                print(f"[Scientist] All 3 attempts failed. Last error: {str(e)[:100]}")

    if hypotheses is None:
        # Fallback: use pre-authored domain hypotheses if model call fails
        print(
            "[Scientist] Warning: model call failed. Falling back to local hypotheses file."
        )
        fallback_path = paths.reports_dir / "domain_hypotheses.json"
        if not fallback_path.exists():
            raise RuntimeError(
                "Scientist model unavailable and no fallback domain_hypotheses.json found"
            )
        raw = fallback_path.read_text(encoding="utf-8")
        loaded = json.loads(_normalize_json_block(raw))

        # If fallback contains domain-level records (variables_needed), map them
        # to concrete hypothesis entries compatible with downstream validators.
        hypotheses = []
        avail_set = set(available_columns)

        def map_var_to_col(var: str) -> str | None:
            # prefer last3mo_mean, then mean, then std/min/max, then cv/range
            candidates = [
                f"{var}_last3mo_mean",
                f"{var}_mean",
                f"{var}_std",
                f"{var}_min",
                f"{var}_max",
                f"{var}_cv",
                f"{var}_range",
            ]
            for c in candidates:
                if c in avail_set:
                    return c
            return None

        idx = 1
        for entry in loaded if isinstance(loaded, list) else []:
            vars_needed = entry.get("variables_needed") or entry.get("variables") or []
            mapped = [map_var_to_col(v) for v in vars_needed]
            mapped = [m for m in mapped if m is not None]
            if not mapped:
                continue
            hypotheses.append(
                {
                    "hypothesis_id": f"fallback_{idx:03d}",
                    "source_paper_id": entry.get("paper_title", "domain_knowledge"),
                    "rationale": entry.get("rationale", "fallback generated"),
                    "feature_columns": mapped,
                    "transformation": "raw",
                    "expected_signal": "positive",
                    "confidence": 0.8,
                }
            )
            idx += 1
    if not isinstance(hypotheses, list):
        raise ValueError("Scientist model did not return a JSON array")

    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    feature_frame_path = paths.data_processed_dir / "features_train.csv"
    if not feature_frame_path.exists():
        raise FileNotFoundError(f"Missing feature matrix: {feature_frame_path}")
    feature_frame = pd.read_csv(feature_frame_path)

    failed_path = (
        Path(failed_hypotheses_path)
        if failed_hypotheses_path
        else (paths.reports_dir / "failed_hypotheses.json")
    )
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
        scientist_last_run=json.dumps(
            {
                "status": "OK",
                "validated": len(validated),
                "failed": len(failed),
            }
        ),
        last_updated=datetime.now(timezone.utc).isoformat(),
    )

    print(
        f"[Scientist] {len(validated)}/{len(hypotheses)} hypotheses passed validation → {hypothesis_path}"
    )
    print(f"[Scientist] Failed hypotheses ledger → {failed_path}")
    return validated


def run(config: dict, state_store: SkillStateStore) -> None:
    """Standard entry point wrapper that logs a warning or executes scientist."""
    print(
        "WARNING: Standard skill_20 entry point run() called. This skill utilizes run_scientist() instead."
    )
    # Log a "Not Implemented" warning through the state store as requested
    state_store.update(
        scientist_warning="skill_20 run() called but is not implemented in the standard loop; execute via run_scientist() instead."
    )


if __name__ == "__main__":
    from zindian.paths import resolve_competition_paths

    paths = resolve_competition_paths(require_competition=False)
    comp_dir = (
        paths.competition_dir
        if paths.competition_dir
        else Path("competitions/ey-frogs")
    )
    reports_dir = comp_dir / "reports"
    run_scientist(
        hypotheses_path=str(reports_dir / "domain_hypotheses.json"),
        priorart_path=str(reports_dir / "ml_priorart.json"),
        hypothesis_path=str(reports_dir / "validated_hypotheses.json"),
        failed_hypotheses_path=str(reports_dir / "failed_hypotheses.json"),
    )
