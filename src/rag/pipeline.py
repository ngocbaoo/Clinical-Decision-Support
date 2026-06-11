"""
RAG pipeline orchestration (PRD §7.2):

  query -> router -> [retrieval (safety-priority by intent) ∥ patient context
  + scores] -> allergy gate -> grounded generation -> cited response

`RAGPipeline.ask()` is the single entry point used by the CLI and the
evaluation harness. Per-stage latencies are returned alongside the response.
"""

import sys
import time
from pathlib import Path

import chromadb

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ on sys.path
from paths import CHROMA_PATH  # noqa: E402
from embedding.or_client import ChatClient, EmbeddingClient, DEFAULT_MODEL  # noqa: E402
from embedding.retriever import COLLECTION_NAME, retrieve  # noqa: E402
from rag.config import GEN_MODEL, TOP_K  # noqa: E402
from rag.generator import generate  # noqa: E402
from rag.query_router import route  # noqa: E402
from rag.safety import (check_allergies, check_contraindications,  # noqa: E402
                        check_drug_interactions)


class RAGPipeline:
    def __init__(self, gen_model: str = GEN_MODEL):
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        self.collection = client.get_collection(COLLECTION_NAME)
        self.embedder = EmbeddingClient(DEFAULT_MODEL)
        self.chat = ChatClient(gen_model)

    def retrieve_for_intent(self, query: str, intent: str,
                            n_results: int = TOP_K) -> list[dict]:
        """Safety-priority retrieval driven by ROUTED intent, not keywords."""
        if intent == "contraindication":
            vec = self.embedder.embed_one(query)  # embed once, query twice
            safety = retrieve(query, self.collection, self.embedder,
                              n_results=n_results, query_embedding=vec,
                              chunk_type_filter="contraindication", min_score=0.0)
            rest = retrieve(query, self.collection, self.embedder,
                            n_results=n_results, query_embedding=vec,
                            min_score=0.0)
            seen, merged = set(), []
            for c in safety + rest:  # contraindication chunks first
                key = c["text"][:80]
                if key not in seen:
                    seen.add(key)
                    merged.append(c)
            return merged[:n_results]
        return retrieve(query, self.collection, self.embedder,
                        n_results=n_results, min_score=0.0)

    def ask(self, query: str, patient_context: dict | None = None,
            calc: dict | None = None) -> dict:
        """Full pipeline run; returns {response, routing, timings_s}."""
        timings = {}

        t = time.perf_counter()
        routing = route(query, self.chat)
        timings["router"] = round(time.perf_counter() - t, 2)

        chunks = []
        if routing["intent"] != "off_topic":
            t = time.perf_counter()
            try:
                chunks = self.retrieve_for_intent(query, routing["intent"])
            except Exception as err:
                # Embedding/API outage -> no chunks -> generator falls back
                # (never crashes the caller, never answers uncited).
                print(f"  [retrieval failed] {err}", file=sys.stderr)
            timings["retrieval"] = round(time.perf_counter() - t, 2)

        t = time.perf_counter()
        alerts = check_allergies(routing["drugs"], patient_context or {})
        alerts += check_contraindications(routing["drugs"], patient_context or {})
        alerts += check_drug_interactions(routing["drugs"], patient_context or {})
        timings["safety"] = round(time.perf_counter() - t, 2)

        t = time.perf_counter()
        response = generate(query, routing["intent"], chunks,
                            patient_context or {}, calc or {}, alerts, self.chat)
        timings["generation"] = round(time.perf_counter() - t, 2)
        timings["total"] = round(sum(timings.values()), 2)

        return {"response": response, "routing": routing, "timings_s": timings,
                "chunks": chunks}
