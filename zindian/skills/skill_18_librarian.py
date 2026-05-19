"""
Skill 18 — The Librarian
Copied from legacy `skill_02_4_librarian.py` to canonical skill number.
"""
from __future__ import annotations

import json, time, requests
from pathlib import Path
try:
    from zindian.skills.skill_07_features import TC_VARIABLES
except Exception:
    import ast, re
    p = Path(__file__).resolve().parent / "skill_07_features.py"
    txt = p.read_text(encoding="utf-8")
    m = re.search(r"TC_VARIABLES\s*=\s*(\[[^\]]*\])", txt, re.S)
    if m:
        TC_VARIABLES = ast.literal_eval(m.group(1))
    else:
        TC_VARIABLES = [
            "aet", "def", "pdsi", "pet", "ppt",
            "q", "soil", "srad", "swe",
            "tmax", "tmin", "vap", "vpd",
        ]

SEMANTIC_SCHOLAR_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "paperId,title,abstract,year,authors,tldr"
MAX_RETRIES = 5
BASE_BACKOFF = 2

QUERY_TEMPLATES = [
    "TerraClimate species distribution modelling Australia",
    "TerraClimate amphibian occurrence prediction",
    "{var} climate variable frog distribution southeastern Australia",
    "monthly climate features species distribution machine learning",
    "TerraClimate temporal features biodiversity modelling",
    "precipitation temperature amphibian habitat suitability Australia",
]

try:
    from zindian.clients.semantic_scholar import SemanticScholarClient
    try:
        _SS_CLIENT = SemanticScholarClient()
    except Exception:
        _SS_CLIENT = None
except Exception:
    _SS_CLIENT = None


def build_queries(tc_variables: list[str]) -> list[str]:
    queries = []
    for tmpl in QUERY_TEMPLATES:
        if "{var}" in tmpl:
            for var in ["ppt", "tmax", "tmin", "aet", "pdsi"]:
                queries.append(tmpl.format(var=var))
        else:
            queries.append(tmpl)
    return queries


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

    params = {"query": query, "limit": limit, "fields": FIELDS}
    time.sleep(3.0)
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(SEMANTIC_SCHOLAR_SEARCH, params=params, timeout=15)
            if r.status_code == 429:
                wait = (attempt + 1) * 15
                print(f"[Librarian] Rate limited. Cooled down requirement triggered. Waiting {wait}s...")
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
    fallback_variables = ["ppt", "tmax", "tmin", "aet", "def", "pdsi", "pet", "q", "soil", "srad", "vap", "vpd"]

    keyword_map = [
        ("precipitation", ["precip", "rain", "rainfall", "wet"], ["ppt"]),
        ("temperature", ["temp", "thermal", "heat", "cold"], ["tmax", "tmin", "pet"]),
        ("moisture stress", ["moisture", "soil", "drought", "arid", "dry"], ["soil", "aet", "def", "pdsi", "q"]),
        ("radiation", ["radiation", "solar", "insolation", "sun"], ["srad"]),
        ("atmospheric demand", ["vapour", "vapor", "vpd", "evap", "evapotranspiration"], ["vap", "vpd", "aet", "pet"]),
        ("seasonality", ["season", "monthly", "temporal", "lag"], ["ppt", "tmax", "tmin", "srad"]),
        ("habitat suitability", ["habitat", "occurrence", "distribution", "suitability", "species"], ["ppt", "tmax", "tmin", "srad", "vap", "vpd"]),
    ]

    for paper in entries:
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
        signals = []
        variables = []
        for signal_name, keywords, tc_variables in keyword_map:
            if any(keyword in text for keyword in keywords):
                signals.append(signal_name)
                variables.extend(tc_variables)

        variables = [variable for variable in _dedupe_preserve_order(variables) if variable in TC_VARIABLES]
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
        hypotheses.append({
            "signal": signal,
            "rationale": abstract[:240],
            "variables_needed": variables,
            "paper_title": paper.get("title"),
            "year": paper.get("year"),
        })

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

def run_librarian(config_path: str, cache_path: str) -> dict:
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
            entries.append({
                "paper_id": pid,
                "title":    p.get("title"),
                "year":     p.get("year"),
                "query":    q,
                "abstract": abstract,
            })
        time.sleep(1.2)

    cache = {
        "status":      "COMPLETE",
        "tc_variables": TC_VARIABLES,
        "region":      "southeastern Australia",
        "temporal_window": "2017-11 to 2019-11",
        "query_count": len(queries),
        "paper_count": len(entries),
        "entries":     entries,
    }
    Path(cache_path).write_text(json.dumps(cache, indent=2))
    print(f"[Librarian] Cached {len(entries)} unique abstracts → {cache_path}")

    domain_hypotheses_path = Path(cache_path).with_name("domain_hypotheses.json")
    domain_hypotheses = _build_domain_hypotheses(entries)
    domain_hypotheses_path.write_text(json.dumps(domain_hypotheses, indent=2), encoding="utf-8")
    print(f"[Librarian] Wrote {len(domain_hypotheses)} domain hypotheses → {domain_hypotheses_path}")

    return cache


if __name__ == "__main__":
    run_librarian(
        config_path = "competitions/ey-frogs/challenge_config.json",
        cache_path  = "competitions/ey-frogs/reports/literature_cache.json",
    )
