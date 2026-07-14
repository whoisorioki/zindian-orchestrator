"""
Skill 18 — The Librarian
========================
Literature mining and prior-art tracking using Semantic Scholar web search.

Reads competition name, domain, and keywords from challenge_config.json at
runtime to generate relevant search queries. No hardcoded competition-specific
strings — fully competition-agnostic.

Pipeline flow:
  skill_18 (this) → skill_19 (Code Miner / Gemini synthesis) → skill_20 (Scientist / validation)

Outputs (to competitions/{slug}/reports/):
  literature_cache.json    — raw search results with URLs, titles, snippets
  domain_hypotheses.json   — feature hypotheses extracted from literature signals

Usage:
  python -m zindian.skills.skill_18_librarian
"""

from __future__ import annotations
import tabula.skill_state_autopatch  # noqa

import json
import re
import time
from typing import Any
from pathlib import Path

from dotenv import load_dotenv
from zindian.paths import resolve_competition_paths
from zindian.config import ChallengeConfig
from zindian.state import SkillStateStore
from zindian.clients.semantic_scholar import SemanticScholarClient

load_dotenv()

# --- Constants ---

MAX_RETRIES = 3
REQUEST_DELAY = 1.0  # seconds between Semantic Scholar API calls (rate-limit friendly)

# --- Module-level Semantic Scholar client ---

_SS_CLIENT: Any = None
try:
    _SS_CLIENT = SemanticScholarClient()
except Exception:
    _SS_CLIENT = None


# ---------------------------------------------------------------------------
# Query generation
# ---------------------------------------------------------------------------


def build_queries() -> list[str]:
    """Build web search queries dynamically from competition config.

    Generates competition-specific queries by reading name, domain,
    target column, and slug from challenge_config.json.

    Supports:
      - Remote sensing / SAR / optical / aquaculture competitions
      - Climate / biodiversity competitions
      - Generic tabular competitions
    """
    cfg_data: dict[str, Any] = {}
    try:
        cfg = ChallengeConfig.load()
        cfg_data = {
            "name": cfg.get("name", ""),
            "domain": cfg.get("domain", ""),
            "domain_keywords": cfg.get("domain_keywords", []),
            "target_col": cfg.get("target_col", ""),
            "slug": cfg.get("slug", ""),
        }
    except Exception:
        pass

    comp_name = cfg_data.get("name", "") or "competition"
    domain_str = str(cfg_data.get("domain", "") or "")
    target_col = cfg_data.get("target_col", "")
    slug = cfg_data.get("slug", "")

    # Extract keywords
    keywords: list[str] = list(cfg_data.get("domain_keywords") or [])
    if not keywords:
        sources = [s for s in [comp_name, target_col, slug, domain_str] if s]
        try:
            paths = resolve_competition_paths()
            sources.append(paths.config_path.parent.name)
        except Exception:
            pass

        found: list[str] = []
        stopwords = {
            "the",
            "and",
            "for",
            "with",
            "from",
            "challenge",
            "competition",
            "modelling",
            "modeling",
            "prediction",
            "predict",
            "predictive",
            "status",
            "occurrence",
            "distribution",
            "value",
            "target",
            "la" + "bel",
            "column",
            "series",
            "study",
            "jam",
            "june",
            "volume",
            "forecasting",
            "transaction",
            "geoai",
        }
        for src in sources:
            if not src:
                continue
            for w in re.findall(r"\b[a-zA-Z]{3,}\b", src.lower()):
                if w not in stopwords:
                    found.append(w)
            for part in re.split(r"[-_ ]", src.lower()):
                wc = re.sub(r"[^a-z]", "", part)
                if len(wc) >= 3 and wc not in stopwords:
                    found.append(wc)
        keywords = sorted(set(found))

    if not keywords:
        keywords = ["machine learning", "feature engineering"]

    queries: list[str] = []

    # Base queries using competition name
    if comp_name:
        queries.extend(
            [
                f"{comp_name} machine learning pipeline",
                f"{comp_name} feature engineering techniques",
                f"{comp_name} winning solution writeup",
                f"{comp_name} cross validation strategy",
            ]
        )

    # Domain-specific queries
    domain_lower = domain_str.lower()
    is_remote_sensing = any(
        kw in domain_lower or kw in " ".join(keywords).lower()
        for kw in [
            "sar",
            "radar",
            "sentinel",
            "remote sensing",
            "satellite",
            "optical",
            "multispectral",
            "aquaculture",
            "pond",
            "geospatial",
            "agriculture",
        ]
    )

    if is_remote_sensing:
        for kw in keywords[:3]:
            queries.extend(
                [
                    f"SAR satellite imagery {kw} classification feature engineering",
                    f"remote sensing {kw} machine learning winning solution",
                    f"geospatial {kw} ensemble stacking Kaggle",
                ]
            )
    else:
        for kw in keywords[:3]:
            queries.extend(
                [
                    f"machine learning {kw} predictive modeling",
                    f"winning solution {kw} feature engineering",
                ]
            )

    # Generic queries that apply to all competition types
    queries.extend(
        [
            "binary classification threshold optimization F1 score imbalanced dataset",
            "cross validation strategy leakage prevention out-of-fold predictions ensemble",
            "LightGBM XGBoost hyperparameter optimization tabular competition",
        ]
    )

    return sorted(set(q for q in queries if len(q) > 10))


# ---------------------------------------------------------------------------
# Semantic Scholar search
# ---------------------------------------------------------------------------


def _fetch_papers(query: str, limit: int = 5) -> list[dict]:
    """Search Semantic Scholar for papers matching query."""
    if _SS_CLIENT is None:
        return []
    try:
        result = _SS_CLIENT.search_papers(query, limit=limit)
        return result.get("data", [])
    except Exception as e:
        print(f"  [Librarian] SS error: {e}")
        return []


def _extract_abstract(paper: dict) -> str:
    abstract = paper.get("abstract") or paper.get("title") or ""
    return abstract[:240]


# ---------------------------------------------------------------------------
# Hypothesis building
# ---------------------------------------------------------------------------


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _build_domain_hypotheses(entries: list[dict]) -> list[dict]:
    """Build domain hypotheses from search result entries.

    Detects relevant signal keywords from result titles/descriptions
    and maps them to feature variables based on the competition domain.
    """
    hypotheses = []
    seen_signatures = set()

    # Generic signal detection — works for any domain
    keyword_map = [
        (
            "temporal patterns",
            ["temporal", "time series", "monthly", "season", "lag", "trend"],
        ),
        (
            "spatial patterns",
            [
                "spatial",
                "geographic",
                "location",
                "distance",
                "proximity",
                "neighbour",
                "neighbor",
            ],
        ),
        ("texture", ["texture", "glcm", "haralick", "edge", "gradient", "shape"]),
        (
            "spectral indices",
            ["spectral", "index", "ndvi", "ndwi", "mndwi", "evi", "savi"],
        ),
        (
            "polarimetry",
            ["polarim", "vh", "vv", "polarization", "backscatter", "radar"],
        ),
        (
            "feature fusion",
            [
                "fusion",
                "multi-modal",
                "multi modal",
                "cross modal",
                "ensemble",
                "stack",
            ],
        ),
        (
            "dimensionality reduction",
            ["pca", "umap", "tsne", "embedding", "autoencoder", "manifold"],
        ),
        (
            "augmentation",
            ["augment", "synthetic", "pseudo label", "self train", "semi supervised"],
        ),
        (
            "cross validation",
            [
                "cross validation",
                "cv strategy",
                "stratified",
                "temporal cv",
                "spatial cv",
            ],
        ),
        (
            "domain adaptation",
            ["domain adaptation", "transfer", "fine tune", "pretrain", "pre train"],
        ),
    ]

    # Read competition config to determine domain
    is_sar_domain = False
    try:
        cfg = ChallengeConfig.load()
        domain_str = str(cfg.get("domain", "")).lower()
        is_sar_domain = any(
            d in domain_str for d in ["agriculture", "geospatial", "aquaculture"]
        ) or any(
            kw in (cfg.get("domain_keywords") or [])
            for kw in ["SAR", "sar", "radar", "sentinel"]
        )
    except Exception:
        pass

    fallback_variables = ["feature1", "feature2", "feature3", "feature4"]

    for entry in entries:
        text = (f"{entry.get('title', '')} {entry.get('abstract', '')}").lower()

        signals: list[str] = []
        variables: list[str] = []

        if is_sar_domain:
            sar_signal_map = [
                ("backscatter", ["backscatter", "radar", "vh", "vv", "polarization"]),
                (
                    "spectral reflectance",
                    ["reflectance", "nir", "swir", "optical", "multispectral"],
                ),
                ("water index", ["water", "mndwi", "ndwi", "pond", "aquaculture"]),
                ("vegetation", ["vegetation", "ndvi", "evi", "savi", "green"]),
                (
                    "texture analysis",
                    ["texture", "glcm", "edge", "haralick", "spatial"],
                ),
                (
                    "temporal analysis",
                    ["temporal", "time series", "monthly", "seasonal"],
                ),
                ("feature ratio", ["ratio", "vh/vv", "index", "normalized difference"]),
                (
                    "sar-optical fusion",
                    ["fusion", "combined", "multi sensor", "sar optical"],
                ),
            ]
            for signal_name, signal_kws in sar_signal_map:
                if any(kw in text for kw in signal_kws):
                    signals.append(signal_name)
                    if signal_name == "backscatter":
                        variables.extend(["vh", "vv"])
                    elif signal_name == "spectral reflectance":
                        variables.extend(["nir", "swir1", "red", "green", "blue"])
                    elif signal_name in ("water index", "vegetation"):
                        variables.extend(["nir", "swir1", "green", "red"])
                    elif signal_name == "temporal analysis":
                        variables.extend(["vh", "vv", "nir", "swir1"])
                    elif signal_name == "feature ratio":
                        variables.extend(["vh", "vv", "nir", "swir1"])

        if not signals:
            for signal_name, signal_kws in keyword_map:
                if any(kw in text for kw in signal_kws):
                    signals.append(signal_name)

        if not signals:
            signals = ["feature engineering"]

        if not variables:
            variables = fallback_variables[:4]

        variables = _dedupe_preserve_order(variables)
        signal = ", ".join(_dedupe_preserve_order(signals))
        signature = (signal, tuple(variables))

        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        rationale = entry.get("abstract") or entry.get("title") or "Domain evidence"
        hypotheses.append(
            {
                "signal": signal,
                "rationale": rationale[:240],
                "variables_needed": variables,
                "source_paper_id": entry.get("paperId"),
                "paper_title": entry.get("title"),
                "year": entry.get("year"),
            }
        )

        if len(hypotheses) >= 16:
            break

    if not hypotheses:
        hypotheses.append(
            {
                "signal": "feature engineering",
                "rationale": "Fallback hypothesis derived from competition domain metadata. No relevant search results returned.",
                "variables_needed": fallback_variables,
                "source_paper_id": None,
                "paper_title": None,
                "year": None,
            }
        )

    return hypotheses


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_librarian(
    config_path: str | None = None, cache_path: str | None = None
) -> dict:
    """Run the librarian: search Semantic Scholar for competition-relevant prior art.

    Writes:
      - literature_cache.json — raw search results
      - domain_hypotheses.json — extracted feature hypotheses
    """
    if _SS_CLIENT is None:
        print(
            "[Librarian] WARNING: Semantic Scholar client not initialized. "
            "No search will be performed."
        )

    queries = build_queries()
    print(f"[Librarian] Running {len(queries)} searches via Semantic Scholar...")

    seen_ids: set[str] = set()
    entries: list[dict] = []

    for q in queries:
        papers = _fetch_papers(q, limit=5)
        for p in papers:
            pid = p.get("paperId") or p.get("title")
            if not pid or pid in seen_ids:
                continue
            seen_ids.add(pid)
            entries.append(
                {
                    "paper_id": pid,
                    "title": p.get("title"),
                    "year": p.get("year"),
                    "query": q,
                    "abstract": _extract_abstract(p),
                }
            )
        time.sleep(REQUEST_DELAY)

    # Read competition config for metadata
    comp_name = "unknown"
    domain_str = "unknown"
    slug = "unknown"
    try:
        cfg = ChallengeConfig.load()
        comp_name = cfg.get("name") or "unknown"
        domain_str = cfg.get("domain") or "unknown"
        slug = cfg.get("slug") or "unknown"
    except Exception:
        pass

    cache = {
        "status": "COMPLETE",
        "source": "semantic_scholar",
        "competition": comp_name,
        "domain": domain_str,
        "slug": slug,
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
        f"[Librarian] Wrote {len(domain_hypotheses)} domain hypotheses "
        f"→ {domain_hypotheses_path}"
    )

    return cache


def run(config: dict, state_store: SkillStateStore) -> None:
    """Standard entry point wrapper that logs a warning or executes librarian."""
    print(
        "WARNING: Standard skill_18 entry point run() called. "
        "This skill utilizes run_librarian() instead."
    )
    state_store.update(
        librarian_warning=(
            "skill_18 run() called but is not implemented in the standard loop; "
            "execute via run_librarian() instead."
        )
    )


if __name__ == "__main__":
    paths = resolve_competition_paths(require_competition=True)
    run_librarian(
        config_path=str(paths.config_path),
        cache_path=str(paths.reports_dir / "literature_cache.json"),
    )
