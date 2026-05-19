"""Simple Semantic Scholar API client with basic rate-limiting.

Reads `SEMANTIC_SCHOLAR_API_KEY` from environment if not provided.
This wrapper is intentionally small and synchronous; callers should
ensure they respect the overall project rate limits (1 req/sec).
"""
from __future__ import annotations

import os
import time
import threading
from typing import Optional, Dict, Any

import requests
from dotenv import load_dotenv


load_dotenv()


class SemanticScholarClient:
    def __init__(self, api_key: Optional[str] = None, rate_limit_per_sec: float = 1.0) -> None:
        api_key = api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        if not api_key:
            raise ValueError("SEMANTIC_SCHOLAR_API_KEY not set in environment")

        self.session = requests.Session()
        self.session.headers.update({"x-api-key": api_key, "Accept": "application/json"})
        self._lock = threading.Lock()
        self._min_interval = 1.0 / float(rate_limit_per_sec) if rate_limit_per_sec > 0 else 0.0
        self._last_call = 0.0
        self.max_retries = 3
        self.backoff_factor = 2.0

    def _throttle(self) -> None:
        with self._lock:
            now = time.time()
            elapsed = now - self._last_call
            wait = self._min_interval - elapsed
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time()

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"https://api.semanticscholar.org{path}"
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                resp = self.session.get(url, params=params, timeout=15)
                if resp.status_code == 429:
                    wait_time = (self.backoff_factor ** attempt) + (attempt * 5)
                    print(f"  [SS] Rate limited (429). Backing off {wait_time:.0f}s...")
                    time.sleep(wait_time)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise
                wait_time = (self.backoff_factor ** attempt) + (attempt * 5)
                print(f"  [SS] Request error: {e}. Backing off {wait_time:.0f}s...")
                time.sleep(wait_time)
        raise RuntimeError(f"Failed after {self.max_retries} retries to {url}")

    def get_paper(self, paper_id: str, fields: str = "title,abstract,authors,year") -> Dict[str, Any]:
        """Fetch a paper by Semantic Scholar paper id (Graph API).

        Example paper_id: "CorpusID:12345" or S2PaperId.
        """
        path = f"/graph/v1/paper/{paper_id}"
        return self._get(path, params={"fields": fields})

    def search_papers(self, query: str, limit: int = 10, fields: str = "title,abstract,authors,year") -> Dict[str, Any]:
        path = "/graph/v1/paper/search"
        return self._get(path, params={"query": query, "limit": limit, "fields": fields})
