"""
Offline tests for the shared OpenRouter embedding client — the query-embedding LRU cache.
No network / no API key: we bypass __init__ and stub embed() to count calls.
"""

import sys
import threading
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embedding.or_client import EmbeddingClient  # noqa: E402


def _offline_client(cache_size: int = 3) -> EmbeddingClient:
    """An EmbeddingClient with a counting embed() stub and no constructor (no key/network)."""
    c = EmbeddingClient.__new__(EmbeddingClient)
    c._qcache = OrderedDict()
    c._qcache_size = cache_size
    c._qlock = threading.Lock()
    c.calls = []
    c.embed = lambda texts: (c.calls.append(tuple(texts)) or [[0.1, 0.2, 0.3]])
    return c


def test_embed_one_caches_repeat_query():
    c = _offline_client()
    first = c.embed_one("MAP của bệnh nhân là bao nhiêu?")
    second = c.embed_one("MAP của bệnh nhân là bao nhiêu?")
    assert first == second
    assert len(c.calls) == 1  # second served from cache, no extra API call


def test_embed_one_distinct_queries_each_embed_once():
    c = _offline_client()
    c.embed_one("q-a")
    c.embed_one("q-b")
    assert len(c.calls) == 2  # different text -> different cache key


def test_embed_one_lru_eviction():
    c = _offline_client(cache_size=3)
    for q in ("q1", "q2", "q3"):
        c.embed_one(q)
    c.embed_one("q1")          # refresh q1 -> q2 becomes least-recently-used
    c.embed_one("q4")          # inserts q4, evicts q2
    n = len(c.calls)
    c.embed_one("q1")          # still cached
    assert len(c.calls) == n   # no new call
    c.embed_one("q2")          # was evicted -> re-embeds
    assert len(c.calls) == n + 1


def test_embed_one_cache_disabled_when_size_zero():
    c = _offline_client(cache_size=0)
    c.embed_one("q")
    c.embed_one("q")
    assert len(c.calls) == 2  # no caching when size is 0
