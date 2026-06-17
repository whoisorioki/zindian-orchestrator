"""
Skill 18 — The Librarian
Canonical librarian implementation for literature mining and prior-art tracking.
"""

from __future__ import annotations
import tabula.skill_state_autopatch  # noqa

import json
import time
from typing import Any
import requests
from pathlib import Path
from zindian.paths import resolve_competition_paths
from zindian.config import ChallengeConfig
from zindian.state import SkillStateStore

from zindian.constants import TC_VARIABLES as _CANONICAL_TC

# Allow challenge config to override the canonical values if provided
TC_VARIABLES: list[str] = _CANONICAL_TC
try:
    cfg = ChallengeConfig.load()
    tc_conf = cfg.get("tc_variables")
    if isinstance(tc_conf, list) and tc_conf:
        TC_VARIABLES = tc_conf
except Exception:
    pass

SEMANTIC_SCHOLAR_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "paperId,title,abstract,year,authors,tldr"
MAX_RETRIES = 5
BASE_BACKOFF = 2

QUERY_TEMPLATES = [
    "TerraClimate species distribution modelling",
    "TerraClimate species occurrence prediction",
    "{var} climate variable species distribution",
    "monthly climate features species distribution machine learning",
    "TerraClimate temporal features species modelling",
    "precipitation temperature species suitability",
]

_SS_CLIENT: Any = None
try:
    from zindian.clients.semantic_scholar import SemanticScholarClient

    try:
        _SS_CLIENT = SemanticScholarClient()
    except Exception:
        _SS_CLIENT = None
except Exception:
    _SS_CLIENT = None


def build_queries(tc_variables: list[str]) -> list[str]:
    keywords = []
    comp_name = "Species distribution modelling"
    comp_domain = "biodiversity"
    target_col = "Occurrence Status"
    slug = ""
    try:
        cfg = ChallengeConfig.load()
        if cfg.get("domain_keywords"):
            keywords = cfg.get("domain_keywords")
        if cfg.get("name"):
            comp_name = cfg.get("name")
        if cfg.get("domain"):
            comp_domain = cfg.get("domain")
        if cfg.get("target_col"):
            target_col = cfg.get("target_col")
        if cfg.get("slug"):
            slug = cfg.get("slug")
    except Exception:
        pass

    # 1. Fallback cascade if no explicit keywords are defined
    if not isinstance(keywords, list) or not keywords:
        sources = [comp_name, target_col, slug]
        try:
            paths = resolve_competition_paths()
            sources.append(paths.config_path.parent.name)
        except Exception:
            pass

        found = []
        stopwords = {
            "the", "and", "for", "with", "from", "challenge", "competition",
            "modelling", "modeling", "prediction", "predict", "predictive", "status",
            "occurrence", "distribution", "value", "target", "label",
            "column", "series", "study", "jam", "june", "volume", "forecasting", "transaction"
        }
        for source_str in sources:
            if source_str:
                # Find all alphabetical words of length >= 3
                for w in re.findall(r"\b[a-zA-Z]{3,}\b", source_str.lower()):
                    if w not in stopwords:
                        found.append(w)
                # also split by separators (dashes, underscores)
                for part in re.split(r"[-_ ]", source_str.lower()):
                    w_clean = re.sub(r"[^a-z]", "", part)
                    if len(w_clean) >= 3 and w_clean not in stopwords:
                        found.append(w_clean)
        keywords = sorted(list(set(found)))

    # 2. Ultimate fallback to prevent empty keywords
    if not keywords:
        keywords = ["biodiversity", "species"]

    # 3. Generate query set dynamically using the keywords
    queries = []
    
    # Base generic ML queries
    if comp_name:
        queries.extend([
            f"{comp_name} machine learning pipeline",
            f"{comp_name} feature engineering techniques",
            f"{comp_name} winning solution writeup",
            f"{comp_name} cross validation strategy",
        ])

    # Dynamic queries based on extracted/configured domain keywords
    for kw in keywords:
        queries.extend([
            f"TerraClimate {kw} occurrence prediction",
            f"precipitation temperature {kw} suitability",
            f"TerraClimate temporal features {kw} modelling",
        ])
        for var in ["ppt", "tmax", "tmin", "aet", "pdsi"]:
            queries.append(f"{var} climate variable {kw} distribution")

    return sorted(list(set(queries)))



def fetch_papers(query: str, limit: int = 5) -> list[dict]:
    """Fetch papers using Semantic Scholar client if available, otherwise fallback to requests."""
    # Prefer the Semantic Scholar client if available (reads API key and rate-limits)
    if _SS_CLIENT is not None:
        try:
            resp = _SS_CLIENT.search_papers(query, limit=limit)
            if isinstance(resp, dict):
                return resp.get("data", [])
        except Exception as e:
            print(f"[Librarian] SemanticScholar client error: {e}")

    params: dict[str, Any] = {"query": query, "limit": limit, "fields": FIELDS}
    time.sleep(3.0)
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(SEMANTIC_SCHOLAR_SEARCH, params=params, timeout=15)
            if r.status_code == 429:
                wait = (attempt + 1) * 15
                print(
                    f"[Librarian] Rate limited. Cooled down requirement triggered. Waiting {wait}s..."
                )
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json().get("data", [])
        except requests.RequestException as e:
            print(f"[Librarian] Request error ({query[:40]}): {e}")
            time.sleep(5)
    return []


def extract_abstract(paper: dict) -> str | None:
    if paper.get("tldr") and paper["tldr"].get("text"):
        return paper["tldr"]["text"]
    return paper.get("abstract")


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _build_domain_hypotheses(entries: list[dict]) -> list[dict]:
    hypotheses = []
    seen_signatures = set()
    fallback_variables = [
        "ppt",
        "tmax",
        "tmin",
        "aet",
        "def",
        "pdsi",
        "pet",
        "q",
        "soil",
        "srad",
        "vap",
        "vpd",
    ]

    keyword_map = [
        ("precipitation", ["precip", "rain", "rainfall", "wet"], ["ppt"]),
        ("temperature", ["temp", "thermal", "heat", "cold"], ["tmax", "tmin", "pet"]),
        (
            "moisture stress",
            ["moisture", "soil", "drought", "arid", "dry"],
            ["soil", "aet", "def", "pdsi", "q"],
        ),
        ("radiation", ["radiation", "solar", "insolation", "sun"], ["srad"]),
        (
            "atmospheric demand",
            ["vapour", "vapor", "vpd", "evap", "evapotranspiration"],
            ["vap", "vpd", "aet", "pet"],
        ),
        (
            "seasonality",
            ["season", "monthly", "temporal", "lag"],
            ["ppt", "tmax", "tmin", "srad"],
        ),
        (
            "habitat suitability",
            ["habitat", "occurrence", "distribution", "suitability", "species"],
            ["ppt", "tmax", "tmin", "srad", "vap", "vpd"],
        ),
    ]

    for paper in entries:
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
        signals = []
        variables = []
        for signal_name, keywords, tc_variables in keyword_map:
            if any(keyword in text for keyword in keywords):
                signals.append(signal_name)
                variables.extend(tc_variables)

        variables = [
            variable
            for variable in _dedupe_preserve_order(variables)
            if variable in TC_VARIABLES
        ]
        if not variables:
            variables = fallback_variables[:4]

        if not signals:
            signals = ["climate variability"]

        signal = ", ".join(_dedupe_preserve_order(signals))
        signature = (signal, tuple(variables))
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        abstract = paper.get("abstract") or paper.get("title") or "Domain evidence"
        hypotheses.append(
            {
                "signal": signal,
                "rationale": abstract[:240],
                "variables_needed": variables,
                "paper_title": paper.get("title"),
                "year": paper.get("year"),
            }
        )

        if len(hypotheses) >= 16:
            break

    if not hypotheses:
        hypotheses = [
            {
                "signal": "climate variability",
                "rationale": "Fallback domain signal derived from TerraClimate-only competition constraints.",
                "variables_needed": ["ppt", "tmax", "tmin", "aet"],
                "paper_title": None,
                "year": None,
            }
        ]

    return hypotheses


def run_librarian(
    config_path: str | None = None, cache_path: str | None = None
) -> dict:
    queries = build_queries(TC_VARIABLES)
    seen_ids = set()
    entries = []

    for q in queries:
        papers = fetch_papers(q)
        for p in papers:
            pid = p.get("paperId")
            if not pid or pid in seen_ids:
                continue
            abstract = extract_abstract(p)
            if not abstract:
                continue
            seen_ids.add(pid)
            entries.append(
                {
                    "paper_id": pid,
                    "title": p.get("title"),
                    "year": p.get("year"),
                    "query": q,
                    "abstract": abstract,
                }
            )
        time.sleep(1.2)

    cache = {
        "status": "COMPLETE",
        "tc_variables": TC_VARIABLES,
        "region": "southeastern Australia",
        "temporal_window": "2017-11 to 2019-11",
        "query_count": len(queries),
        "paper_count": len(entries),
        "entries": entries,
    }
    if cache_path is None:
        paths = resolve_competition_paths(require_competition=True)
        cache_path = str(paths.reports_dir / "literature_cache.json")

    Path(cache_path).write_text(json.dumps(cache, indent=2))
    print(f"[Librarian] Cached {len(entries)} unique abstracts → {cache_path}")

    domain_hypotheses_path = Path(cache_path).with_name("domain_hypotheses.json")
    domain_hypotheses = _build_domain_hypotheses(entries)
    domain_hypotheses_path.write_text(
        json.dumps(domain_hypotheses, indent=2), encoding="utf-8"
    )
    print(
        f"[Librarian] Wrote {len(domain_hypotheses)} domain hypotheses → {domain_hypotheses_path}"
    )

    return cache


def run(config: dict, state_store: SkillStateStore) -> None:
    """Standard entry point wrapper that logs a warning or executes librarian."""
    print("WARNING: Standard skill_18 entry point run() called. This skill utilizes run_librarian() instead.")
    # Log a "Not Implemented" warning through the state store as requested
    state_store.update(
        librarian_warning="skill_18 run() called but is not implemented in the standard loop; execute via run_librarian() instead."
    )


if __name__ == "__main__":
    paths = resolve_competition_paths(require_competition=True)
    run_librarian(
        config_path=str(paths.config_path),
        cache_path=str(paths.reports_dir / "literature_cache.json"),
    )
