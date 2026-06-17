"""
Skill 19 — The Code Miner
=========================
Searches for winning ML pipeline patterns from public competition writeups
and GitHub repositories using Gemini Flash's built-in web search capability.

Two problems served:
  Problem 1 (Generic Agent): Extract reusable ML prior art for any competition
  Problem 2 (Geospatial):    Find geospatial species distribution tricks

Output: reports/ml_priorart.json
        reports/code_miner_report.md

Usage:
  python -m zindian.skills.skill_19_code_miner
  python -m zindian.skills.skill_19_code_miner --domain=geospatial
  python -m zindian.skills.skill_19_code_miner --domain=tabular
  python -m zindian.skills.skill_19_code_miner --dry-run

Requirements:
    pip install google-genai

NOTE: Uses Gemini Flash free tier (google.genai).
            Rate limit: 15 requests/minute on free tier.
            Never scrapes Kaggle directly — searches public writeup summaries only.
"""

from __future__ import annotations
import tabula.skill_state_autopatch  # noqa

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from zindian.config import ChallengeConfig
from zindian.paths import resolve_competition_paths
from zindian.state import SkillStateStore

genai: Any = None
try:
    import google.genai as genai

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

MODEL_NAME = "gemini-2.5-flash"


SEARCH_TEMPLATES = {
    "geospatial": [
        (
            "species distribution model spatial autocorrelation Kaggle solution "
            "geospatial tabular features winning approach",
            "Geospatial SDM — validation and feature tricks",
        ),
        (
            "Kaggle geospatial competition spatial cross-validation block CV "
            "geographic leakage winning solution writeup",
            "Spatial CV strategies from competition winners",
        ),
        (
            "tabular binary classification imbalanced geospatial presence absence "
            "LightGBM Random Forest ensemble top solution",
            "Ensemble strategies for presence/absence classification",
        ),
        (
            "environmental feature engineering climate variables species occurrence "
            "temporal lag window features predictive model",
            "Temporal feature engineering for climate-species models",
        ),
        (
            "TerraClimate WorldClim bioclimatic variables species distribution "
            "machine learning feature importance SHAP",
            "Climate variable feature selection for SDMs",
        ),
    ],
    "tabular": [
        (
            "Kaggle tabular competition winning solution feature engineering "
            "LightGBM CatBoost ensemble stacking approach",
            "General tabular ML winning strategies",
        ),
        (
            "tabular binary classification threshold optimization F1 score "
            "imbalanced dataset handling top solution",
            "F1 optimization and threshold calibration",
        ),
        (
            "Kaggle solution cross-validation strategy leakage prevention "
            "out-of-fold predictions ensemble",
            "CV strategy and OOF ensemble patterns",
        ),
    ],
    "biodiversity": [
        (
            "species occurrence prediction environmental predictors machine learning",
            "Biodiversity modeling ML",
        ),
        (
            "species habitat suitability model climate variables "
            "precipitation temperature breeding trigger",
            "Habitat suitability modeling signals",
        ),
        (
            "citizen science species distribution model " "environmental predictors",
            "Species distribution modeling approaches",
        ),
    ],
}


def empty_priorart_entry(
    query: str,
    query_label: str,
    domain: str,
) -> dict:
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
  "confidence": "high",
  "relevance_to_geospatial_species": "high"
}}

Rules:
- Return ONLY valid JSON starting with {{ and ending with }}
- No markdown fences (```), no backticks, no preamble or explanation
- Empty arrays if nothing relevant found
- Each array element is a string, max 1 sentence
- confidence: "high" | "medium" | "low"
- relevance_to_geospatial_species: "high" | "medium" | "low" | "not_applicable"
"""


def _extract_json_from_response(raw_text: str) -> dict | None:
    """Extract valid JSON from Gemini response, handling markdown fences."""
    if not raw_text:
        return None

    raw_text = raw_text.strip()

    # Try 1: If wrapped in markdown fences, strip them
    if raw_text.startswith("```"):
        # Split by backticks and try to find JSON
        parts = raw_text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                raw_text = part
                break

    # Try 2: Find JSON object via regex (from { to matching })
    if not raw_text.startswith("{"):
        match = re.search(r"\{[\s\S]*\}", raw_text)
        if match:
            raw_text = match.group(0)

    # Try 3: Parse the JSON
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
    raw_text = None

    if dry_run:
        print(f"  [DRY RUN] Would search: {query_label}")
        entry["status"] = "dry_run"
        entry["tricks"] = ["[dry run — no actual search performed]"]
        return entry

    try:
        prompt = EXTRACTION_PROMPT.format(query=query)
        response = gemini_client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
        raw_text = response.text.strip()

        # Use robust JSON extraction
        parsed = _extract_json_from_response(raw_text)

        if parsed is None:
            entry["status"] = "parse_error"
            entry["raw_summary"] = raw_text[:500] if raw_text else None
            entry["warnings"] = ["Failed to extract JSON from response"]
            entry["tricks"] = []
            entry["validation_strategies"] = []
            entry["feature_ideas"] = []
            entry["ensemble_patterns"] = []
            print(
                f"  ⚠️  JSON extraction failed on '{query_label}' — using empty result"
            )
            return entry

        entry["tricks"] = parsed.get("tricks", [])
        entry["validation_strategies"] = parsed.get("validation_strategies", [])
        entry["feature_ideas"] = parsed.get("feature_ideas", [])
        entry["ensemble_patterns"] = parsed.get("ensemble_patterns", [])
        entry["warnings"] = parsed.get("warnings", [])
        entry["raw_summary"] = parsed.get("sources", [])
        entry["status"] = "success"
        entry["confidence"] = parsed.get("confidence", "unknown")
        entry["relevance"] = parsed.get("relevance_to_geospatial_species", "unknown")
        print(f"  ✓ {query_label}: {len(entry['tricks'])} tricks found")

    except Exception as e:
        entry["status"] = "error"
        entry["warnings"] = [f"API error: {e}"]
        print(f"  ❌ API error on '{query_label}': {e}")

    return entry


SYNTHESIS_PROMPT = """
You are a senior ML competition strategist.

Below are search results from multiple queries about winning ML competition approaches
for a species distribution problem.

Competition context:
- Target: binary classification (target present = 1, absent = 0)
- Metric: F1 score (maximize)
- Features: 52 TerraClimate climate variables (no lat/lon allowed)
- Dataset: 6312 training rows, Nov 2017 - Nov 2019
- Current best LB: 0.8846
- Gap to top 10: ~0.08

Search results:
{results_json}

Synthesize into a JSON object with exactly these fields:

{\
  "top_3_actionable_tricks": [
    {\
      "trick": "Concrete thing to implement",
      "expected_impact": "high|medium|low",
      "implementation_complexity": "high|medium|low",
      "relevant_to_tc_only_features": true
    }\
  ],
  "feature_hypotheses": [
    {\
      "hypothesis": "Specific new feature to create from TerraClimate variables",
      "ecological_basis": "Why this should predict target presence",
      "variables_needed": ["list of TC variable names"],
      "complexity": "simple|moderate|complex"
    }
  ],
  "validation_recommendation": "One concrete CV strategy recommendation",
  "ensemble_recommendation": "One concrete ensembling recommendation",
  "do_not_attempt": [
    "Things that repeatedly failed in similar competitions"
  ]
}\

Rules:
- Only synthesize from the provided search results
- Mark relevance to TC-only feature constraint explicitly
- Return ONLY valid JSON, no preamble, no markdown fences
"""


def synthesize_results(
    gemini_client,
    entries: list[dict],
    dry_run: bool = False,
) -> dict:
    if dry_run:
        return {
            "top_3_actionable_tricks": [
                {
                    "trick": "[dry run]",
                    "expected_impact": "unknown",
                    "implementation_complexity": "unknown",
                    "relevant_to_tc_only_features": True,
                }
            ],
            "feature_hypotheses": [],
            "validation_recommendation": "[dry run]",
            "ensemble_recommendation": "[dry run]",
            "do_not_attempt": [],
        }

    successful = [e for e in entries if e["status"] == "success"]
    if not successful:
        return {"error": "No successful search results to synthesize"}

    trimmed = []
    for e in successful:
        trimmed.append(
            {
                "label": e["query_label"],
                "tricks": e["tricks"][:3],
                "validation": e["validation_strategies"][:2],
                "features": e["feature_ideas"][:3],
                "ensemble": e["ensemble_patterns"][:2],
                "warnings": e["warnings"][:2],
            }
        )

    try:
        comp_name = "Species distribution model"
        domain_name = "geospatial"
        target_col = "target"
        task_type = "binary classification"
        metric = "F1"
        metric_dir = "maximize"
        extra_ctx = ""
        try:
            config = ChallengeConfig.load()
            if config.get("name"):
                comp_name = config.get("name")
            if config.domain:
                domain_name = config.domain
            if config.get("target_col"):
                target_col = config.get("target_col")
            if config.get("task_type"):
                task_type = config.get("task_type")
            if config.metric:
                metric = config.metric
            if config.metric_direction:
                metric_dir = config.metric_direction

            if (
                "frog" in comp_name.lower()
                or "frog" in config.slug.lower()
                or "biodiversity" in comp_name.lower()
            ):
                extra_ctx = (
                    "- Features: 52 TerraClimate climate variables (no lat/lon allowed)\n"
                    "- Dataset: 6312 training rows, SE Australia, Nov 2017 - Nov 2019\n"
                    "- Current best LB: 0.8846\n"
                    "- Gap to top 10: ~0.08"
                )
            else:
                shape = config.get("data_shape", {}) or {}
                extra_ctx = f"- Dataset: {shape.get('n_train', 0)} training rows, {shape.get('n_cols', 0)} columns"
        except Exception:
            pass

        prompt = f"""You are a senior ML competition strategist.

Below are search results from multiple queries about winning ML competition approaches
for a {domain_name} problem ({comp_name}).

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
      "implementation_complexity": "high|medium|low",
      "relevant_to_tc_only_features": true
    }}
  ],
  "feature_hypotheses": [
    {{
      "hypothesis": "Specific new feature to create",
      "ecological_basis": "Why this should predict the target",
      "variables_needed": ["list of variables or features needed"],
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
- Mark relevance to the feature constraints explicitly
- Return ONLY valid JSON, no preamble, no markdown fences
"""
        response = gemini_client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
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
    mapping = {
        "high": 0.9,
        "medium": 0.75,
        "low": 0.6,
    }
    return mapping.get(str(impact).strip().lower(), 0.5)


def _build_code_miner_cache(
    entries: list[dict],
    queries: list[tuple[str, str, str]],
    domain: str,
    synthesis: dict,
) -> dict:
    query_texts = [query for query, _, _ in queries]
    successful = sum(1 for entry in entries if entry.get("status") == "success")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "OK" if successful else "EMPTY",
        "model": MODEL_NAME,
        "source_types": ["kaggle", "github", "huggingface"],
        "domain": domain,
        "query_count": len(query_texts),
        "queries": query_texts,
        "raw_count": len(entries),
        "successful_count": successful,
        "synthesis_summary": {
            "top_tricks": len(synthesis.get("top_3_actionable_tricks", [])),
            "feature_hypotheses": len(synthesis.get("feature_hypotheses", [])),
        },
    }


def _build_code_miner_patterns(
    entries: list[dict], synthesis: dict, domain: str
) -> dict:
    patterns = []

    for index, trick in enumerate(synthesis.get("top_3_actionable_tricks", []), 1):
        impact = trick.get("expected_impact", "unknown")
        pattern = {
            "pattern_id": f"cm_{index:03d}",
            "source_type": "synthesized",
            "source_ref": "gemini_synthesis",
            "category": "strategy",
            "technique_name": trick.get("trick", "unknown"),
            "problem_shape": f"{domain.title()} binary classification",
            "implementation_steps": [
                trick.get("trick", "Implement the synthesized trick"),
                f"Focus on {domain}-appropriate feature engineering and validation",
            ],
            "leakage_risk": "unknown",
            "expected_gain": impact,
            "confidence": _confidence_from_impact(impact),
        }
        patterns.append(pattern)

    base_index = len(patterns)
    for offset, hypothesis in enumerate(synthesis.get("feature_hypotheses", []), 1):
        variables = hypothesis.get("variables_needed", [])
        pattern = {
            "pattern_id": f"cm_{base_index + offset:03d}",
            "source_type": "synthesized",
            "source_ref": "gemini_synthesis",
            "category": "feature_engineering",
            "technique_name": hypothesis.get("hypothesis", "unknown"),
            "problem_shape": f"{domain.title()} binary classification",
            "implementation_steps": [
                hypothesis.get(
                    "ecological_basis", "Use the synthesized basis to shape features"
                ),
                f"Construct features from: {', '.join(variables) if variables else 'available TC variables'}",
            ],
            "leakage_risk": "low",
            "expected_gain": hypothesis.get("complexity", "unknown"),
            "confidence": 0.7,
        }
        patterns.append(pattern)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "OK" if patterns else "EMPTY",
        "patterns_count": len(patterns),
        "patterns": patterns,
    }


def write_markdown_report(
    entries: list[dict],
    synthesis: dict,
    report_path: Path,
    domain: str,
) -> None:
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

    tricks = synthesis.get("top_3_actionable_tricks", [])
    if tricks:
        for i, t in enumerate(tricks, 1):
            lines.append(f"### {i}. {t.get('trick', 'unknown')}")
            lines.append(f"- **Impact**: {t.get('expected_impact', '?')}")
            lines.append(f"- **Complexity**: {t.get('implementation_complexity', '?')}")
            lines.append(
                f"- **TC-only compatible**: {t.get('relevant_to_tc_only_features', '?')}"
            )
            lines.append("")
    else:
        lines.append("*No tricks extracted*")
        lines.append("")

    lines += [
        "## Feature Hypotheses",
        "",
    ]
    hypotheses = synthesis.get("feature_hypotheses", [])
    if hypotheses:
        for h in hypotheses:
            lines.append(f"### {h.get('hypothesis', 'unknown')}")
            lines.append(f"- **Basis**: {h.get('ecological_basis', '?')}")
            lines.append(f"- **Variables**: {h.get('variables_needed', [])}")
            lines.append(f"- **Complexity**: {h.get('complexity', '?')}")
            lines.append("")
    else:
        lines.append("*No hypotheses extracted*")
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

    lines += [
        "---",
        "",
        "## Raw Search Results",
        "",
    ]
    for e in entries:
        lines.append(f"### {e['query_label']} ({e['status']})")
        for trick in e.get("tricks", []):
            lines.append(f"- **Trick**: {trick}")
        for feat in e.get("feature_ideas", []):
            lines.append(f"- **Feature**: {feat}")
        for warn in e.get("warnings", []):
            lines.append(f"- ⚠️  {warn}")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ Report written → {report_path}")


def run(
    domain: str = "geospatial",
    dry_run: bool = False,
) -> dict:
    print(f"\n{'=' * 60}")
    print("SKILL 19 — The Code Miner")
    print(f"{'=' * 60}\n")
    print(f"Domain  : {domain}")
    print(f"Dry run : {dry_run}")

    paths = resolve_competition_paths()
    config = ChallengeConfig.load()

    if not GEMINI_AVAILABLE and not dry_run:
        print("❌ google-genai not installed.")
        print("   Run: pip install google-genai")
        return {"status": "ERROR", "message": "google-genai not installed"}

    if not dry_run:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("❌ GEMINI_API_KEY not found in environment.")
            print("   Add to .env: GEMINI_API_KEY=your_key_here")
            print("   Get free key: https://aistudio.google.com/app/apikey")
            return {"status": "ERROR", "message": "GEMINI_API_KEY not set"}

        gemini_model = genai.Client(api_key=api_key)
        print("✅ Gemini Flash initialized")
    else:
        gemini_model = None

    # Dynamically select tabular or other template if domain is default or generic
    active_domain = domain
    if active_domain == "geospatial" or active_domain is None:
        try:
            cfg_domain = config.domain
            if cfg_domain in SEARCH_TEMPLATES:
                active_domain = cfg_domain
            elif config.get("data_modality") == "tabular":
                active_domain = "tabular"
        except Exception:
            pass

    if active_domain == "all":
        queries = []
        for d, q_list in SEARCH_TEMPLATES.items():
            queries.extend([(q, label, d) for q, label in q_list])
    else:
        template = SEARCH_TEMPLATES.get(active_domain, SEARCH_TEMPLATES["geospatial"])
        queries = [(q, label, active_domain) for q, label in template]

    print(f"\nRunning {len(queries)} queries…\n")

    entries = []
    for i, (query, label, q_domain) in enumerate(queries):
        print(f"  [{i + 1}/{len(queries)}] {label}")

        entry = query_gemini(gemini_model, query, label, q_domain, dry_run)
        entry["id"] = f"CM_{i + 1:03d}"

        entries.append(entry)

        if not dry_run and i < len(queries) - 1:
            time.sleep(4.0)

    print(f"\nSynthesizing {len(entries)} results…")
    if not dry_run:
        time.sleep(3.0)

    synthesis = synthesize_results(gemini_model, entries, dry_run)

    if paths.competition_dir is None:
        return {
            "status": "ERROR",
            "message": "Competition directory is not configured",
        }

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
    print(f"✅ JSON saved → {json_path}")

    legacy_cache_path = reports_dir / "code_miner_cache.json"
    legacy_patterns_path = reports_dir / "code_miner_patterns.json"
    legacy_cache = _build_code_miner_cache(entries, queries, domain, synthesis)
    legacy_patterns = _build_code_miner_patterns(entries, synthesis, domain)
    legacy_cache_path.write_text(json.dumps(legacy_cache, indent=2), encoding="utf-8")
    legacy_patterns_path.write_text(
        json.dumps(legacy_patterns, indent=2), encoding="utf-8"
    )
    print(f"✅ Legacy cache saved → {legacy_cache_path}")
    print(f"✅ Legacy patterns saved → {legacy_patterns_path}")

    report_path = reports_dir / "code_miner_report.md"
    write_markdown_report(entries, synthesis, report_path, domain)

    state_store = SkillStateStore(paths.state_path)
    state_store.update(
        code_miner_last_run=datetime.now(timezone.utc).isoformat(),
        code_miner_domain=domain,
        code_miner_queries=len(queries),
        last_updated=datetime.now(timezone.utc).isoformat(),
    )

    print(f"\n{'=' * 60}")
    print("SYNTHESIS — TOP ACTIONABLE TRICKS")
    print(f"{'=' * 60}")
    for i, trick in enumerate(synthesis.get("top_3_actionable_tricks", []), 1):
        print(f"  {i}. {trick.get('trick', '?')}")
        print(
            f"     Impact: {trick.get('expected_impact', '?')} | "
            f"Complexity: {trick.get('implementation_complexity', '?')}"
        )

    print(f"\nValidation: {synthesis.get('validation_recommendation', '?')}")
    print(f"Ensemble  : {synthesis.get('ensemble_recommendation', '?')}")

    return {
        "status": "OK",
        "domain": domain,
        "queries_run": len(queries),
        "entries": len(entries),
        "synthesis": synthesis,
        "json_path": str(json_path),
        "report_path": str(report_path),
    }


def run_code_miner(
    domain: str = "geospatial",
    dry_run: bool = False,
) -> dict:
    return run(domain=domain, dry_run=dry_run)


if __name__ == "__main__":
    domain = "geospatial"
    dry_run = False

    for arg in sys.argv[1:]:
        if arg.startswith("--domain="):
            domain = arg.split("=", 1)[1]
        elif arg == "--dry-run":
            dry_run = True

    valid_domains = list(SEARCH_TEMPLATES.keys()) + ["all"]
    if domain not in valid_domains:
        print(f"❌ Unknown domain '{domain}'. Choose from: {valid_domains}")
        sys.exit(1)

    result = run_code_miner(domain=domain, dry_run=dry_run)
    printable = {k: v for k, v in result.items() if k not in ("entries", "synthesis")}
    print(json.dumps(printable, indent=2))
