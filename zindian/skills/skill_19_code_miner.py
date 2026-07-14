"""
Skill 19 — The Code Miner
=========================
Searches for winning ML pipeline patterns from public competition writeups
and GitHub repositories using Gemini Flash's built-in web search capability.

Reads competition domain and name from challenge_config.json at runtime
to generate relevant search queries. No hardcoded competition-specific
strings — fully competition-agnostic.

Output: reports/ml_priorart.json
        reports/code_miner_report.md

Usage:
  python -m zindian.skills.skill_19_code_miner
  python -m zindian.skills.skill_19_code_miner --dry-run

Requirements:
    pip install google-genai
"""

from __future__ import annotations
import tabula.skill_state_autopatch  # noqa

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from typing import Any

from zindian.config import ChallengeConfig
from zindian.paths import resolve_competition_paths
from zindian.state import SkillStateStore

genai: Any = None
GEMINI_AVAILABLE = False
try:
    import google.genai as _genai_module

    genai = _genai_module
    GEMINI_AVAILABLE = True
except ImportError:
    pass

MODEL_NAME = "gemini-2.5-flash"


def build_queries() -> list[tuple[str, str]]:
    """Generate search queries dynamically from competition config.

    Reads competition name, domain, target column, and task type from
    challenge_config.json to build relevant Gemini web search queries.
    Falls back to generic ML competition queries when config unavailable.
    """
    config = ChallengeConfig.load()
    comp_name = config.get("name") or "machine learning competition"
    domain_str = config.get("domain") or ""
    task_type = config.get("task_type") or "tabular"

    queries: list[tuple[str, str]] = []

    # Competition-specific queries
    if comp_name:
        queries.append(
            (f"{comp_name} Kaggle winning solution approach", f"Solution: {comp_name}")
        )
        queries.append(
            (
                f"{comp_name} feature engineering techniques used",
                f"Features: {comp_name}",
            )
        )
        queries.append((f"{comp_name} cross validation strategy", f"CV: {comp_name}"))

    # Domain-specific queries
    domain_lower = (domain_str or "").lower()
    is_remote = any(
        kw in domain_lower
        for kw in [
            "sar",
            "radar",
            "satellite",
            "optical",
            "remote sensing",
            "geospatial",
            "aquaculture",
            "agriculture",
        ]
    )
    is_biodiversity = any(
        kw in domain_lower for kw in ["biodiversity", "species", "climate", "habitat"]
    )
    is_tabular = (
        task_type == "tabular"
        or task_type == "classification"
        or task_type == "regression"
    )

    if is_remote:
        queries.append(
            (
                "satellite imagery machine learning classification feature engineering Kaggle",
                "Satellite ML",
            )
        )
        queries.append(
            (
                "remote sensing competition winning solution ensemble stacking",
                "Remote sensing ensemble",
            )
        )
        queries.append(
            (
                f"SAR optical feature engineering {task_type} competition",
                "SAR/optical features",
            )
        )
    elif is_biodiversity:
        queries.append(
            (
                "environmental variables species distribution machine learning feature engineering",
                "Eco ML features",
            )
        )
        queries.append(
            (
                "imbalanced presence absence classification ensemble strategies",
                "Imbalanced classification",
            )
        )
    elif is_tabular:
        queries.append(
            (
                "Kaggle tabular competition winning solution feature engineering LightGBM CatBoost",
                "Tabular ML",
            )
        )
        queries.append(
            (
                "binary classification threshold optimization F1 score imbalanced dataset",
                "F1 optimization",
            )
        )

    # Generic queries that apply to all competition types
    queries.extend(
        [
            (
                "cross validation strategy leakage prevention out-of-fold predictions ensemble",
                "CV/OOF patterns",
            ),
            (
                "feature engineering techniques winning Kaggle solution stacking blending",
                "Feature engineering",
            ),
            (
                "LightGBM XGBoost hyperparameter optimization tabular competition",
                "GBM tuning",
            ),
        ]
    )

    # Deduplicate by label
    seen = set()
    deduped = []
    for q, label in queries:
        if label not in seen:
            seen.add(label)
            deduped.append((q, label))
    return deduped


def empty_priorart_entry(query: str, query_label: str, domain: str) -> dict:
    return {
        "id": None,
        "domain": domain,
        "query": query,
        "query_label": query_label,
        "source": "gemini_web_search",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tricks": [],
        "validation_strategies": [],
        "feature_ideas": [],
        "ensemble_patterns": [],
        "warnings": [],
        "raw_summary": None,
        "status": "pending",
    }


EXTRACTION_PROMPT = """
You are a competitive data science research assistant.
Search the web for: "{query}"

Return ONLY a valid JSON object with these exact fields. No markdown, no fences, no preamble.
Start directly with {{ and end with }}.

{{
  "tricks": ["Trick 1", "Trick 2", "Trick 3"],
  "validation_strategies": ["Strategy 1", "Strategy 2"],
  "feature_ideas": ["Feature 1", "Feature 2", "Feature 3"],
  "ensemble_patterns": ["Pattern 1", "Pattern 2"],
  "warnings": ["Warning 1", "Warning 2"],
  "sources": ["source 1", "source 2"],
  "confidence": "high"
}}

Rules:
- Return ONLY valid JSON starting with {{ and ending with }}
- No markdown fences (```), no backticks, no preamble or explanation
- Empty arrays if nothing relevant found
- Each array element is a string, max 1 sentence
- confidence: "high" | "medium" | "low"
"""


def _extract_json_from_response(raw_text: str) -> dict | None:
    """Extract valid JSON from Gemini response, handling markdown fences."""
    if not raw_text:
        return None
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        parts = raw_text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                raw_text = part
                break
    if not raw_text.startswith("{"):
        match = re.search(r"\{[\s\S]*\}", raw_text)
        if match:
            raw_text = match.group(0)
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as e:
        print(f"    Failed to parse JSON: {e}")
        print(f"    Raw text (first 200 chars): {raw_text[:200]}")
        return None


def query_gemini(
    gemini_client,
    query: str,
    query_label: str,
    domain: str,
    dry_run: bool = False,
) -> dict:
    entry = empty_priorart_entry(query, query_label, domain)
    if dry_run:
        print(f"  [DRY RUN] Would search: {query_label}")
        entry["status"] = "dry_run"
        entry["tricks"] = ["[dry run]"]
        return entry

    try:
        prompt = EXTRACTION_PROMPT.format(query=query)
        response = gemini_client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
        parsed = _extract_json_from_response(response.text.strip())
        if parsed is None:
            entry["status"] = "parse_error"
            entry["raw_summary"] = response.text[:500]
            entry["warnings"] = ["Failed to extract JSON from response"]
            return entry
        entry["tricks"] = parsed.get("tricks", [])
        entry["validation_strategies"] = parsed.get("validation_strategies", [])
        entry["feature_ideas"] = parsed.get("feature_ideas", [])
        entry["ensemble_patterns"] = parsed.get("ensemble_patterns", [])
        entry["warnings"] = parsed.get("warnings", [])
        entry["raw_summary"] = parsed.get("sources", [])
        entry["status"] = "success"
        entry["confidence"] = parsed.get("confidence", "unknown")
        print(f"  [OK] {query_label}: {len(entry['tricks'])} tricks found")
    except Exception as e:
        entry["status"] = "error"
        entry["warnings"] = [f"API error: {e}"]
        print(f"  [FAIL] API error on '{query_label}': {e}")
    return entry


def _build_synthesis_prompt(entries: list[dict]) -> str:
    """Build a synthesis prompt with competition context from config."""
    try:
        config = ChallengeConfig.load()
        comp_name = config.get("name") or "unknown"
        target_col = config.get("target_col") or "target"
        task_type = config.get("task_type") or "tabular"
        metric = config.get("metric") or "accuracy"
        metric_dir = config.get("metric_direction") or "maximize"
        shape = config.get("data_shape", {}) or {}
        extra_ctx = f"- Dataset: {shape.get('n_train', 'N/A')} rows, {shape.get('n_cols', 'N/A')} features"
    except Exception:
        comp_name = "machine learning competition"
        target_col = "target"
        task_type = "tabular"
        metric = "accuracy"
        metric_dir = "maximize"
        extra_ctx = ""

    trimmed = []
    for e in entries:
        if e["status"] == "success":
            trimmed.append(
                {
                    "query_label": e["query_label"],
                    "tricks": e["tricks"][:3],
                    "validation": e["validation_strategies"][:2],
                    "features": e["feature_ideas"][:3],
                    "ensemble": e["ensemble_patterns"][:2],
                }
            )

    return f"""You are a senior ML competition strategist.

Below are search results from multiple queries about winning ML competition approaches
for the {comp_name} competition.

Competition context:
- Target: {target_col} ({task_type})
- Metric: {metric} ({metric_dir})
{extra_ctx}

Search results:
{json.dumps(trimmed, indent=2)}

Synthesize into a JSON object with exactly these fields:

{{
  "top_3_actionable_tricks": [
    {{
      "trick": "Concrete thing to implement",
      "expected_impact": "high|medium|low",
      "implementation_complexity": "high|medium|low"
    }}
  ],
  "feature_hypotheses": [
    {{
      "hypothesis": "Specific new feature to create",
      "rationale": "Why this should predict the target",
      "variables_needed": ["list of features needed"],
      "complexity": "simple|moderate|complex"
    }}
  ],
  "validation_recommendation": "One concrete CV strategy recommendation",
  "ensemble_recommendation": "One concrete ensembling recommendation",
  "do_not_attempt": [
    "Things that repeatedly failed in similar competitions"
  ]
}}

Rules:
- Only synthesize from the provided search results
- Return ONLY valid JSON, no preamble, no markdown fences
"""


def synthesize_results(
    gemini_client, entries: list[dict], dry_run: bool = False
) -> dict:
    if dry_run:
        return {
            "top_3_actionable_tricks": [],
            "feature_hypotheses": [],
            "validation_recommendation": "[dry run]",
            "ensemble_recommendation": "[dry run]",
        }
    successful = [e for e in entries if e["status"] == "success"]
    if not successful:
        return {"error": "No successful search results to synthesize"}
    try:
        prompt = _build_synthesis_prompt(entries)
        response = gemini_client.models.generate_content(
            model=MODEL_NAME, contents=prompt
        )
        raw_text = response.text.strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
        return json.loads(raw_text)
    except Exception as e:
        return {"error": f"Synthesis failed: {e}"}


def _confidence_from_impact(impact: str) -> float:
    return {"high": 0.9, "medium": 0.75, "low": 0.6}.get(
        str(impact).strip().lower(), 0.5
    )


def _build_code_miner_cache(entries, queries, domain, synthesis):
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "OK" if sum(1 for e in entries if e.get("status") == "success") else "EMPTY"
        ),
        "model": MODEL_NAME,
        "domain": domain,
        "query_count": len(queries),
        "successful_count": sum(1 for e in entries if e.get("status") == "success"),
        "synthesis_summary": {
            "top_tricks": len(synthesis.get("top_3_actionable_tricks", [])),
            "feature_hypotheses": len(synthesis.get("feature_hypotheses", [])),
        },
    }


def _build_code_miner_patterns(entries, synthesis, domain):
    patterns = []
    for index, trick in enumerate(synthesis.get("top_3_actionable_tricks", []), 1):
        impact = trick.get("expected_impact", "unknown")
        patterns.append(
            {
                "pattern_id": f"cm_{index:03d}",
                "source_type": "synthesized",
                "technique_name": trick.get("trick", "unknown"),
                "problem_shape": f"{domain.title()} competition",
                "expected_gain": impact,
                "confidence": _confidence_from_impact(impact),
            }
        )
    for offset, hypothesis in enumerate(synthesis.get("feature_hypotheses", []), 1):
        patterns.append(
            {
                "pattern_id": f"cm_{len(patterns) + offset:03d}",
                "source_type": "synthesized",
                "technique_name": hypothesis.get("hypothesis", "unknown"),
                "problem_shape": f"{domain.title()} competition",
                "expected_gain": hypothesis.get("complexity", "unknown"),
                "confidence": 0.7,
            }
        )
    return {
        "status": "OK" if patterns else "EMPTY",
        "patterns_count": len(patterns),
        "patterns": patterns,
    }


def write_markdown_report(entries, synthesis, report_path, domain):
    lines = [
        f"# Code Miner Report — {domain.title()} Domain",
        f"**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Queries run**: {len(entries)}",
        f"**Successful**: {sum(1 for e in entries if e['status'] == 'success')}",
        "",
        "---",
        "",
        "## Top Actionable Tricks",
        "",
    ]
    for i, t in enumerate(synthesis.get("top_3_actionable_tricks", []), 1):
        lines.append(f"### {i}. {t.get('trick', 'unknown')}")
        lines.append(f"- **Impact**: {t.get('expected_impact', '?')}")
        lines.append(f"- **Complexity**: {t.get('implementation_complexity', '?')}")
        lines.append("")
    lines += ["## Feature Hypotheses", ""]
    for h in synthesis.get("feature_hypotheses", []):
        lines.append(f"### {h.get('hypothesis', 'unknown')}")
        lines.append(f"- **Rationale**: {h.get('rationale', '?')}")
        lines.append(f"- **Variables**: {h.get('variables_needed', [])}")
        lines.append("")
    lines += [
        "## Validation Recommendation",
        "",
        synthesis.get("validation_recommendation", "*None*"),
        "",
        "## Ensemble Recommendation",
        "",
        synthesis.get("ensemble_recommendation", "*None*"),
        "",
        "## Do Not Attempt",
        "",
    ]
    for item in synthesis.get("do_not_attempt", []):
        lines.append(f"- {item}")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run(domain: str = "geospatial", dry_run: bool = False) -> dict:
    print(f"\n{'=' * 60}\nSKILL 19 — The Code Miner\n{'=' * 60}\n")
    print(f"Domain  : {domain}")
    print(f"Dry run : {dry_run}")

    paths = resolve_competition_paths()
    config = ChallengeConfig.load()

    if not GEMINI_AVAILABLE and not dry_run:
        print("[FAIL] google-genai not installed.")
        return {"status": "ERROR", "message": "google-genai not installed"}
    if not dry_run:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("[FAIL] GEMINI_API_KEY not found in environment.")
            return {"status": "ERROR", "message": "GEMINI_API_KEY not set"}
        gemini_model = genai.Client(api_key=api_key)
        print("[OK] Gemini Flash initialized")
    else:
        gemini_model = None

    # Build queries dynamically from config
    queries = build_queries()
    print(f"\nRunning {len(queries)} queries…\n")

    entries = []
    for i, (query, label) in enumerate(queries):
        print(f"  [{i + 1}/{len(queries)}] {label}")
        entry = query_gemini(gemini_model, query, label, domain, dry_run)
        entry["id"] = f"CM_{i + 1:03d}"
        entries.append(entry)
        if not dry_run and i < len(queries) - 1:
            time.sleep(4.0)

    print(f"\nSynthesizing {len(entries)} results…")
    if not dry_run:
        time.sleep(3.0)
    synthesis = synthesize_results(gemini_model, entries, dry_run)

    if paths.competition_dir is None:
        return {"status": "ERROR", "message": "Competition directory not configured"}

    reports_dir = paths.competition_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "skill": "skill_19_code_miner",
        "competition": config.slug,
        "domain": domain,
        "generated": datetime.now(timezone.utc).isoformat(),
        "query_count": len(queries),
        "entries": entries,
        "synthesis": synthesis,
    }
    json_path = reports_dir / "ml_priorart.json"
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"[OK] JSON saved → {json_path}")

    legacy_cache = _build_code_miner_cache(entries, queries, domain, synthesis)
    legacy_patterns = _build_code_miner_patterns(entries, synthesis, domain)
    (reports_dir / "code_miner_cache.json").write_text(
        json.dumps(legacy_cache, indent=2), encoding="utf-8"
    )
    (reports_dir / "code_miner_patterns.json").write_text(
        json.dumps(legacy_patterns, indent=2), encoding="utf-8"
    )

    report_path = reports_dir / "code_miner_report.md"
    write_markdown_report(entries, synthesis, report_path, domain)
    print(f"[OK] Report written → {report_path}")

    state_store = SkillStateStore(paths.state_path)
    state_store.update(
        code_miner_last_run=datetime.now(timezone.utc).isoformat(),
        code_miner_domain=domain,
    )

    print(f"\n{'=' * 60}\nSYNTHESIS — TOP ACTIONABLE TRICKS\n{'=' * 60}")
    for i, trick in enumerate(synthesis.get("top_3_actionable_tricks", []), 1):
        print(
            f"  {i}. {trick.get('trick', '?')} (Impact: {trick.get('expected_impact', '?')})"
        )

    return {"status": "OK", "queries_run": len(queries), "json_path": str(json_path)}


def run_code_miner(domain: str = "geospatial", dry_run: bool = False) -> dict:
    return run(domain=domain, dry_run=dry_run)


if __name__ == "__main__":
    domain = "geospatial"
    dry_run = False
    for arg in sys.argv[1:]:
        if arg.startswith("--domain="):
            domain = arg.split("=", 1)[1]
        elif arg == "--dry-run":
            dry_run = True
    result = run_code_miner(domain=domain, dry_run=dry_run)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("entries", "synthesis")},
            indent=2,
        )
    )
