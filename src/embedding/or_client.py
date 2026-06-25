"""
Shared OpenRouter embedding client.

OpenRouter exposes an OpenAI-compatible `/embeddings` endpoint (even though
embedding models are not listed in `/models`). Verified working models:
    qwen/qwen3-embedding-8b          (4096-dim) — used for indexing
    openai/text-embedding-3-small    (1536-dim) — used for Task 1.5 comparison

Reused by embedding_eval.py, embedder.py and retriever.py.
"""

import math
import os
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Windows consoles default to cp1252 and choke on Vietnamese — force UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ on sys.path
from paths import ENV_FILE  # noqa: E402

load_dotenv(ENV_FILE)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "qwen/qwen3-embedding-8b"
DEFAULT_CHAT_MODEL = "qwen/qwen3.6-flash"


def _normalize(vec: list[float]) -> list[float]:
    """L2-normalize a vector so ChromaDB cosine scores are clean."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


class EmbeddingClient:
    """Thin wrapper over the OpenRouter embeddings endpoint."""

    def __init__(self, model_name: str = DEFAULT_MODEL, query_cache_size: int = 512):
        api_key = os.getenv("OPEN_ROUTER_KEY")
        if not api_key:
            raise RuntimeError(
                "OPEN_ROUTER_KEY not found in environment / .env at repo root."
            )
        self.model_name = model_name
        self.client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
        self._dim: int | None = None
        # Bounded LRU cache for single-text (query) embeddings — see embed_one(). Lock guards it
        # because the pipeline now runs retrieval in a worker thread (concurrent requests share one
        # embedder), and OrderedDict mutation is not atomic across move_to_end/popitem.
        self._qcache: "OrderedDict[str, list[float]]" = OrderedDict()
        self._qcache_size = query_cache_size
        self._qlock = threading.Lock()

    @property
    def dim(self) -> int | None:
        return self._dim

    def embed(self, texts: list[str], max_retries: int = 3) -> list[list[float]]:
        """Embed a batch of texts. Returns L2-normalized vectors."""
        if not texts:
            return []
        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                resp = self.client.embeddings.create(
                    model=self.model_name, input=texts
                )
                vecs = [_normalize(d.embedding) for d in resp.data]
                if self._dim is None and vecs:
                    self._dim = len(vecs[0])
                return vecs
            except Exception as err:  # transient network / rate-limit
                last_err = err
                wait = 2 ** attempt
                print(f"  [embed retry {attempt + 1}/{max_retries}] {err} "
                      f"-> sleeping {wait}s")
                time.sleep(wait)
        raise RuntimeError(f"Embedding failed after {max_retries} attempts: {last_err}")

    def embed_one(self, text: str) -> list[float]:
        """Embed a single text (the query path) with a bounded LRU cache. The query vector is
        deterministic per (model, text), so re-embedding the same query — eval re-runs, repeated
        questions, the contraindication 'embed once query twice' path — skips the ~0.8s API call.
        Indexing goes through embed() (batch) and is intentionally NOT cached."""
        with self._qlock:
            cached = self._qcache.get(text)
            if cached is not None:
                self._qcache.move_to_end(text)
                return cached
        vec = self.embed([text])[0]  # network call outside the lock
        if self._qcache_size > 0:
            with self._qlock:
                self._qcache[text] = vec
                if len(self._qcache) > self._qcache_size:
                    self._qcache.popitem(last=False)  # evict least-recently-used
        return vec


class ChatClient:
    """Thin wrapper over the OpenRouter chat-completions endpoint."""

    def __init__(self, model_name: str = DEFAULT_CHAT_MODEL,
                 reasoning: bool | None = None):
        api_key = os.getenv("OPEN_ROUTER_KEY")
        if not api_key:
            raise RuntimeError(
                "OPEN_ROUTER_KEY not found in environment / .env at repo root."
            )
        self.model_name = model_name
        # reasoning: None = provider default (on for reasoning models); False = disable the
        # hidden chain-of-thought (huge latency win on qwen3.6 — see config.GEN_REASONING_ENABLED).
        self.reasoning = reasoning
        self.client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
        # Token usage of the most recent successful chat() call, or None. Lets callers
        # (latency_probe) split generation into prefill (prompt_tokens) vs decode
        # (completion_tokens) without changing chat()'s return type.
        self.last_usage: dict | None = None

    def chat(self, messages: list[dict], temperature: float = 0.1,
             max_tokens: int = 1200, max_retries: int = 3) -> str:
        """Send a chat-completion request; returns the assistant text."""
        extra = ({"extra_body": {"reasoning": {"enabled": False}}}
                 if self.reasoning is False else {})
        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **extra,
                )
                content = resp.choices[0].message.content
                if content is None:
                    raise RuntimeError("empty completion")
                u = getattr(resp, "usage", None)
                self.last_usage = ({"prompt_tokens": getattr(u, "prompt_tokens", None),
                                    "completion_tokens": getattr(u, "completion_tokens", None)}
                                   if u else None)
                return content
            except Exception as err:  # transient network / rate-limit
                last_err = err
                wait = 2 ** attempt
                print(f"  [chat retry {attempt + 1}/{max_retries}] {err} "
                      f"-> sleeping {wait}s", file=sys.stderr)
                time.sleep(wait)
        raise RuntimeError(f"Chat failed after {max_retries} attempts: {last_err}")

    def chat_stream(self, messages: list[dict], temperature: float = 0.1,
                    max_tokens: int = 1200):
        """Yield assistant text deltas as they arrive (stream=True), setting last_usage at the end.

        This is the TTFT primitive — first token typically lands in ~1s vs ~4.5s for the full
        answer. NOTE for the RAG path: the generator's answer is structured JSON and is gated by
        the post-generation verifier (fail-closed), so deltas must NOT be shown verbatim to the
        clinician — stream them into a clearly-marked "đang soạn / chưa xác minh" draft state and
        swap in the verified answer once verify_answer() clears it. See docs/RAG_PERF_REPORT.md.
        No retry: a streamed connection can't be transparently resumed; callers fall back to chat().
        """
        extra = ({"extra_body": {"reasoning": {"enabled": False}}}
                 if self.reasoning is False else {})
        self.last_usage = None
        stream = self.client.chat.completions.create(
            model=self.model_name, messages=messages, temperature=temperature,
            max_tokens=max_tokens, stream=True,
            stream_options={"include_usage": True}, **extra)
        for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if choices:
                delta = getattr(choices[0], "delta", None)
                piece = getattr(delta, "content", None) if delta else None
                if piece:
                    yield piece
            u = getattr(chunk, "usage", None)  # final chunk carries usage (include_usage)
            if u:
                self.last_usage = {"prompt_tokens": getattr(u, "prompt_tokens", None),
                                   "completion_tokens": getattr(u, "completion_tokens", None)}
