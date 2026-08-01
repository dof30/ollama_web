#!/usr/bin/env python3
"""
search — pluggable search backends.

The interface is one function:

    search(query, max_results) -> [{"title","url","snippet"}, ...]

so a keyed API (Brave, Tavily) or a self-hosted SearXNG can be added later by
writing one class and setting MESH_SEARCH_BACKEND. Nothing above this file knows
which engine it is talking to.

**Why queries are not run in parallel.** It is tempting to fire five searches at
once, but DuckDuckGo's HTML endpoint is scraped, not an API — concurrent bursts
get rate-limited or blocked, and then you have zero results instead of five. So
queries are serialised with a jittered gap.

That costs much less than it sounds like. A query returns in ~1 s and a turn
issues 2-4 of them; the expensive part of research is *fetching and reading*
pages, and that is fully parallel (see fetch.py). Search is not the bottleneck.
"""

import random
import threading
import time

from . import config


class SearchError(RuntimeError):
    pass


class DDGBackend:
    """DuckDuckGo via the `ddgs` package. No key, no account, no quota."""

    name = "duckduckgo"

    # Minimum gap between queries, seconds. Jittered so repeated turns don't
    # produce a machine-regular request pattern.
    MIN_GAP = 1.2
    JITTER = 0.6

    def __init__(self):
        self._lock = threading.Lock()
        self._last = 0.0

    def _throttle(self):
        with self._lock:
            gap = self.MIN_GAP + random.uniform(0, self.JITTER)
            wait = self._last + gap - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()

    def search(self, query, max_results):
        from ddgs import DDGS  # imported lazily so config/import errors stay local

        self._throttle()
        n = max(1, min(int(max_results), 10))
        try:
            with DDGS() as ddgs:
                rows = list(ddgs.text(query, max_results=n))
        except Exception as e:  # ddgs raises a wide variety of transport errors
            raise SearchError(f"search failed: {e}") from e

        out = []
        for r in rows:
            url = r.get("href") or r.get("url") or ""
            if not url:
                continue
            out.append(
                {
                    "title": (r.get("title") or "").strip(),
                    "url": url,
                    "snippet": (r.get("body") or r.get("snippet") or "").strip(),
                }
            )
        return out


_BACKENDS = {"ddg": DDGBackend}
_instance = None
_instance_lock = threading.Lock()


def backend():
    """The configured backend, created once (it holds throttle state)."""
    global _instance
    with _instance_lock:
        if _instance is None:
            cls = _BACKENDS.get(config.SEARCH_BACKEND)
            if cls is None:
                raise SearchError(
                    f"unknown backend {config.SEARCH_BACKEND!r}; "
                    f"have {sorted(_BACKENDS)}"
                )
            _instance = cls()
        return _instance


def search(query, max_results=None):
    return backend().search(query, max_results or config.RESULTS_PER_QUERY)


def search_many(queries, max_results=None):
    """Run several queries and merge, keeping first-seen order and dropping dupes.

    Serialised by the backend's throttle; the point of batching here is that the
    *caller* gets to express "these N queries" in one round instead of taking N
    round-trips through the big model to ask for them one at a time.
    """
    merged, seen, errors = [], set(), []
    for q in queries:
        try:
            rows = search(q, max_results)
        except SearchError as e:
            errors.append(str(e))
            continue
        for r in rows:
            if r["url"] in seen:
                continue
            seen.add(r["url"])
            r["query"] = q
            merged.append(r)
    if not merged and errors:
        raise SearchError(errors[0])
    return merged
