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
import time
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


def _normalize(vec: list[float]) -> list[float]:
    """L2-normalize a vector so ChromaDB cosine scores are clean."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


class EmbeddingClient:
    """Thin wrapper over the OpenRouter embeddings endpoint."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        api_key = os.getenv("OPEN_ROUTER_KEY")
        if not api_key:
            raise RuntimeError(
                "OPEN_ROUTER_KEY not found in environment / .env at repo root."
            )
        self.model_name = model_name
        self.client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
        self._dim: int | None = None

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
        return self.embed([text])[0]
