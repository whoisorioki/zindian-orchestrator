"""
Skill 19 — The Code Miner
=========================
Searches for winning ML pipeline patterns from public competition writeups
and GitHub repositories using Gemini Flash's built-in web search capability.

Two problems served:
  Problem 1 (Generic Agent): Extract reusable ML prior art for any competition
  Problem 2 (EY Frogs):      Find geospatial species distribution tricks

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

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any

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
            "Geospatial SDM — validation and feature tricks"
        ),
        (
            "Kaggle geospatial competition spatial cross-validation block CV "
            "geographic leakage winning solution writeup",
            "Spatial CV strategies from competition winners"
        ),
        (
            "tabular binary classification imbalanced geospatial presence absence "
            "LightGBM Random Forest ensemble top solution",
            "Ensemble strategies for presence/absence classification"
        ),
        (
            "environmental feature engineering climate variables species occurrence "
            "temporal lag window features predictive model",
            "Temporal feature engineering for climate-species models"
        ),
        (
            "TerraClimate WorldClim bioclimatic variables species distribution "
            "machine learning feature importance SHAP",
            "Climate variable feature selection for SDMs"
        ),
    ],

    "tabular": [
        (
            "Kaggle tabular competition winning solution feature engineering "
            "LightGBM CatBoost ensemble stacking approach",
            "General tabular ML winning strategies"
        ),
        (
            "tabular binary classification threshold optimization F1 score "
            "imbalanced dataset handling top solution",
            "F1 optimization and threshold calibration"
        ),
        (
            "Kaggle solution cross-validation strategy leakage prevention "
            "out-of-fold predictions ensemble",
            "CV strategy and OOF ensemble patterns"
        ),
    ],

    "frog_ecology": [
        (
            "frog species occurrence prediction southeastern Australia "
            "environmental predictors machine learning",
            "Frog ecology ML — Australia specific"
        ),
        (
            "amphibian habitat suitability model climate variables "
            "precipitation temperature breeding trigger",
            "Amphibian habitat modeling signals"
        ),
        (
            "FrogID citizen science species distribution model "
            "TerraClimate environmental predictors",
            "FrogID-specific modeling approaches"
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

Extract and return a JSON object with exactly these fields:

{\
  "tricks": [
    "One concrete ML trick found (e.g. target-encoded stratified k-fold)"
  ],
  "validation_strategies": [
    "One validation strategy found (e.g. geographic block CV with KMeans clusters)"
  ],
  "feature_ideas": [
    "One feature engineering idea found (e.g. 3-month lagged precipitation sum)"
  ],
  "ensemble_patterns": [
    "One ensembling pattern found (e.g. OOF stacking with LGB+RF+XGB)"
  ],
  "warnings": [
    "One known pitfall or failure mode found"
  ],
  "sources": [
    "URL or reference where this was found"
  ],
  "confidence": "high|medium|low",
  "relevance_to_geospatial_species": "high|medium|low|not_applicable"
}\

Rules:
- Only include items actually found in search results
- Do not invent or hallucinate tricks
- If nothing relevant found, return empty arrays
- Keep each item to one sentence maximum
- Return ONLY valid JSON, no preamble, no markdown fences
"""


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

        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]

        parsed = json.loads(raw_text)

        entry["tricks"]                = parsed.get("tricks", [])
        entry["validation_strategies"] = parsed.get("validation_strategies", [])
        entry["feature_ideas"]         = parsed.get("feature_ideas", [])
        entry["ensemble_patterns"]     = parsed.get("ensemble_patterns", [])
        entry["warnings"]              = parsed.get("warnings", [])
        entry["raw_summary"]           = parsed.get("sources", [])
        entry["status"]                = "success"
        entry["confidence"]            = parsed.get("confidence", "unknown")
        entry["relevance"]             = parsed.get(
            "relevance_to_geospatial_species", "unknown"
        )

    except json.JSONDecodeError as e:
        entry["status"]      = "parse_error"
        entry["raw_summary"] = raw_text[:500] if raw_text else None
        entry["warnings"]    = [f"JSON parse failed: {e}"]
        print(f"  ⚠️  Parse error on '{query_label}': {e}")

    except Exception as e:
        entry["status"]   = "api_error"
        entry["warnings"] = [f"API error: {e}"]
        print(f"  ❌ API error on '{query_label}': {e}")

    return entry


SYNTHESIS_PROMPT = """
You are a senior ML competition strategist.

Below are search results from multiple queries about winning ML competition approaches
for a geospatial species distribution problem (frog presence/absence in SE Australia).

Competition context:
- Target: binary classification (frog present = 1, absent = 0)
- Metric: F1 score (maximize)
- Features: 52 TerraClimate climate variables (no lat/lon allowed)
- Dataset: 6312 training rows, SE Australia, Nov 2017 - Nov 2019
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
      "ecological_basis": "Why this should predict frog presence",
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
            "top_3_actionable_tricks": [{"trick": "[dry run]", "expected_impact": "unknown",
                                          "implementation_complexity": "unknown",
                                          "relevant_to_tc_only_features": True}],
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
        trimmed.append({
            "label":                e["query_label"],
            "tricks":               e["tricks"][:3],
            "validation":           e["validation_strategies"][:2],
            "features":             e["feature_ideas"][:3],
            "ensemble":             e["ensemble_patterns"][:2],
            "warnings":             e["warnings"][:2],
        })

    try:
        prompt   = SYNTHESIS_PROMPT.format(results_json=json.dumps(trimmed, indent=2))
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


def write_markdown_report(
    entries:   list[dict],
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
            lines.append(f"- **TC-only compatible**: {t.get('relevant_to_tc_only_features', '?')}")
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
    domain:  str  = "geospatial",
    dry_run: bool = False,
) -> dict:
    print(f"\n{'='*60}")
    print(f"SKILL 19 — The Code Miner")
    print(f"{'='*60}\n")
    print(f"Domain  : {domain}")
    print(f"Dry run : {dry_run}")

    paths  = resolve_competition_paths()
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
        print(f"✅ Gemini Flash initialized")
    else:
        gemini_model = None

    if domain == "all":
        queries = []
        for d, q_list in SEARCH_TEMPLATES.items():
            queries.extend([(q, label, d) for q, label in q_list])
    else:
        template = SEARCH_TEMPLATES.get(domain, SEARCH_TEMPLATES["geospatial"])
        queries  = [(q, label, domain) for q, label in template]

    print(f"\nRunning {len(queries)} queries…\n")

    entries = []
    for i, (query, label, q_domain) in enumerate(queries):
        print(f"  [{i+1}/{len(queries)}] {label}")

        entry    = query_gemini(gemini_model, query, label, q_domain, dry_run)
        entry["id"] = f"CM_{i+1:03d}"

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
        "skill":      "skill_19_code_miner",
        "competition": config.slug,
        "domain":     domain,
        "generated":  datetime.now(timezone.utc).isoformat(),
        "query_count": len(queries),
        "entries":    entries,
        "synthesis":  synthesis,
    }
    json_path = reports_dir / "ml_priorart.json"
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"✅ JSON saved → {json_path}")

    report_path = reports_dir / "code_miner_report.md"
    write_markdown_report(entries, synthesis, report_path, domain)

    state_store = SkillStateStore(paths.state_path)
    state_store.update(
        code_miner_last_run=datetime.now(timezone.utc).isoformat(),
        code_miner_domain=domain,
        code_miner_queries=len(queries),
        last_updated=datetime.now(timezone.utc).isoformat(),
    )

    print(f"\n{'='*60}")
    print("SYNTHESIS — TOP ACTIONABLE TRICKS")
    print(f"{'='*60}")
    for i, trick in enumerate(synthesis.get("top_3_actionable_tricks", []), 1):
        print(f"  {i}. {trick.get('trick', '?')}")
        print(f"     Impact: {trick.get('expected_impact', '?')} | "
              f"Complexity: {trick.get('implementation_complexity', '?')}")

    print(f"\nValidation: {synthesis.get('validation_recommendation', '?')}")
    print(f"Ensemble  : {synthesis.get('ensemble_recommendation', '?')}")

    return {
        "status":        "OK",
        "domain":        domain,
        "queries_run":   len(queries),
        "entries":       len(entries),
        "synthesis":     synthesis,
        "json_path":     str(json_path),
        "report_path":   str(report_path),
    }


def run_code_miner(
    domain: str = "geospatial",
    dry_run: bool = False,
) -> dict:
    return run(domain=domain, dry_run=dry_run)


if __name__ == "__main__":
    domain  = "geospatial"
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
    printable = {k: v for k, v in result.items()
                 if k not in ("entries", "synthesis")}
    print(json.dumps(printable, indent=2))
