"""
RAG pipeline orchestration (PRD §7.2):

  query -> router -> [retrieval (safety-priority by intent) ∥ patient context
  + scores] -> allergy gate -> grounded generation -> cited response

`RAGPipeline.ask()` is the single entry point used by the CLI and the
evaluation harness. Per-stage latencies are returned alongside the response.
"""

import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chromadb

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ on sys.path
from paths import CHROMA_PATH  # noqa: E402
from embedding.or_client import ChatClient, EmbeddingClient, DEFAULT_MODEL  # noqa: E402
from embedding.retriever import COLLECTION_NAME, retrieve  # noqa: E402
from rag.config import (GEN_MODEL, GEN_REASONING_ENABLED, TOP_K,  # noqa: E402
                        VERIFIER_BACKEND, VERIFIER_MODEL, VERIFY_ENABLED,
                        VERIFY_EVIDENCE_NLI)
from rag.generator import generate  # noqa: E402
from rag.logging_utils import log_request  # noqa: E402
from rag.query_router import route  # noqa: E402
from rag.safety import (check_allergies, check_contraindications,  # noqa: E402
                        check_drug_interactions)


class RAGPipeline:
    def __init__(self, gen_model: str = GEN_MODEL, *, verify: bool = VERIFY_ENABLED,
                 backend: str = VERIFIER_BACKEND):
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        self.collection = client.get_collection(COLLECTION_NAME)
        self.embedder = EmbeddingClient(DEFAULT_MODEL)
        self.gen_model = gen_model
        # Reasoning disabled on the generation/router client = ~7x faster generation on
        # qwen3.6 (the hidden CoT dominates latency); the verifier client keeps its default.
        self.chat = ChatClient(gen_model, reasoning=GEN_REASONING_ENABLED)
        # Verifier uses a DIFFERENT family from the qwen generator (gpt-5.4-mini) to avoid
        # correlated errors. local_nli needs no chat client. Disabled -> no verification.
        self.backend = backend
        self.verify = verify
        self.verifier = (ChatClient(VERIFIER_MODEL)
                         if verify and backend in ("llm", "hybrid") else None)
        # Local int8 NLI (mDeBERTa-XNLI) for the local_nli/hybrid backends. Lazy-loaded on first
        # use so the torch-free offline path / tests never import optimum-onnxruntime.
        self._nli = None

    def _get_nli(self):
        """Lazy-load the local NLI model once, only when a backend needs it. Loaded for the
        local_nli/hybrid chunk-level backends AND for the Lever-2 evidence-span entailment check
        (VERIFY_EVIDENCE_NLI), which runs regardless of the chunk-level backend."""
        needs_nli = self.backend in ("local_nli", "hybrid") or VERIFY_EVIDENCE_NLI
        if self._nli is None and self.verify and needs_nli:
            from rag.nli_local import LocalNLI
            self._nli = LocalNLI()
        return self._nli

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
        t0 = time.perf_counter()

        t = time.perf_counter()
        routing = route(query, self.chat)
        timings["router"] = round(time.perf_counter() - t, 2)

        # Retrieval (embedding API + vector search) and safety (OpenFDA API + local checks) both
        # depend only on the routed output and are independent of each other — run them concurrently
        # so the OpenFDA tail hides under retrieval instead of stacking after it. Both are I/O-bound,
        # so threads (GIL released on network) give the win without process overhead.
        def _retrieve() -> tuple[list, float]:
            s = time.perf_counter()
            ch: list = []
            if routing["intent"] != "off_topic":
                try:
                    ch = self.retrieve_for_intent(query, routing["intent"])
                except Exception as err:  # API outage -> no chunks -> generator falls back
                    print(f"  [retrieval failed] {err}", file=sys.stderr)
            return ch, round(time.perf_counter() - s, 2)

        def _safety() -> tuple[list, float]:
            s = time.perf_counter()
            a = check_allergies(routing["drugs"], patient_context or {})
            a += check_contraindications(routing["drugs"], patient_context or {})
            a += check_drug_interactions(routing["drugs"], patient_context or {})
            return a, round(time.perf_counter() - s, 2)

        with ThreadPoolExecutor(max_workers=2) as pool:
            f_chunks = pool.submit(_retrieve)
            f_alerts = pool.submit(_safety)
            chunks, timings["retrieval"] = f_chunks.result()
            alerts, timings["safety"] = f_alerts.result()

        t = time.perf_counter()
        verifier_chat = self.verifier if self.verify else None
        response = generate(query, routing["intent"], chunks,
                            patient_context or {}, calc or {}, alerts, self.chat,
                            verifier_chat=verifier_chat,
                            backend=self.backend if self.verify else "llm",
                            nli=self._get_nli())
        timings["generation"] = round(time.perf_counter() - t, 2)
        # Real end-to-end wall-clock (not a sum of stages — retrieval/safety now overlap).
        timings["total"] = round(time.perf_counter() - t0, 2)
        # verify is a sub-stage of generation; reported separately, not double-summed.
        timings["verify"] = (response.get("verify") or {}).get("elapsed_s", 0.0)

        request_id = uuid.uuid4().hex[:12]
        result = {"request_id": request_id, "response": response, "routing": routing,
                  "timings_s": timings, "chunks": chunks}
        self._log(query, result, patient_context)
        return result

    def _log(self, query: str, result: dict, patient_context: dict | None) -> None:
        resp = result["response"]
        log_request({
            "request_id": result["request_id"],
            "query": query,
            "has_patient": bool(patient_context),
            "routing": result["routing"],
            "retrieved": [{"source": c.get("source"), "title": c.get("title"),
                           "score": round(c.get("score", 0.0), 4),
                           "chunk_type": c.get("chunk_type")}
                          for c in result["chunks"]],
            "alerts": [{"type": a.get("type", "allergy"),
                        "drug": a.get("drug") or a.get("drug_a")} for a in resp["alerts"]],
            "verify": resp.get("verify"),
            "fallback": resp["fallback"],
            "fallback_reason": resp["fallback_reason"],
            "citations": resp["citations"],
            "answer": resp["answer"],
            "timings_s": result["timings_s"],
            "models": {"gen": self.gen_model,
                       "verifier_backend": self.backend if self.verify else None},
        })
