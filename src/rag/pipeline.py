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
                        VERIFY_EVIDENCE_NLI, COMORBIDITY_RETRIEVAL, RRF_K,
                        COMORBIDITY_SLOTS, COMORBIDITY_MIN_SCORE,
                        MAX_COMORBIDITY_QUERIES, RETRIEVE_CANDIDATES,
                        COMORBIDITY_QUERY_TEMPLATE, RERANK_ENABLED, RERANK_CANDIDATES,
                        COMORBIDITY_GATE_ENABLED)
from rag.fusion import comorbidity_fuse, comorbidity_names, comorbidity_queries  # noqa: E402
from rag.comorbidity_gate import apply_comorbidity_gate  # noqa: E402
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
        # Cross-encoder reranker (bge-reranker-v2-m3). Lazy-loaded (torch) only when reranking is on.
        self._reranker = None

    def _get_reranker(self):
        """Lazy-load the cross-encoder reranker once, only when RERANK_ENABLED. Keeps torch out of
        the offline path (same contract as _get_nli). Degrades gracefully: if the model isn't in the
        local cache (or fails to load), reranking is skipped and retrieval falls back to the
        bi-encoder — never hang a live request on a multi-GB download."""
        if not RERANK_ENABLED or self._reranker is False:
            return None
        if self._reranker is None:
            from rag.reranker import CrossEncoderReranker, model_is_cached
            if not model_is_cached():
                print("  [reranker] model not cached → bi-encoder only "
                      "(download BAAI/bge-reranker-v2-m3 to enable)", file=sys.stderr)
                self._reranker = False
                return None
            try:
                self._reranker = CrossEncoderReranker()
            except Exception as err:  # noqa: BLE001 — load failure must not break retrieval
                print(f"  [reranker] load failed → bi-encoder only: {err}", file=sys.stderr)
                self._reranker = False
                return None
        return self._reranker

    def _get_nli(self):
        """Lazy-load the local NLI model once, only when a backend needs it. Loaded for the
        local_nli/hybrid chunk-level backends AND for the Lever-2 evidence-span entailment check
        (VERIFY_EVIDENCE_NLI), which runs regardless of the chunk-level backend."""
        needs_nli = self.backend in ("local_nli", "hybrid") or VERIFY_EVIDENCE_NLI
        if self._nli is None and self.verify and needs_nli:
            from rag.nli_local import LocalNLI
            self._nli = LocalNLI()
        return self._nli

    def _primary_retrieval(self, query: str, intent: str, cand_n: int) -> list[dict]:
        """The intent-driven primary list (contraindication-first for safety intents). Pulls a wide
        bi-encoder pool (RERANK_CANDIDATES when reranking, else `cand_n`), then a cross-encoder
        reranks it by joint (query, chunk) relevance — fixing the bi-encoder near-ties that let an
        off-topic chunk win rank-1. Each sub-list is reranked independently so the contra-first
        safety ordering is preserved by construction."""
        rr = self._get_reranker()
        pool_n = RERANK_CANDIDATES if rr is not None else cand_n

        def _rr(lst: list[dict]) -> list[dict]:
            return rr.rerank(query, lst) if rr is not None and lst else lst

        if intent == "contraindication":
            vec = self.embedder.embed_one(query)  # embed once, query twice
            safety = retrieve(query, self.collection, self.embedder,
                              n_results=pool_n, query_embedding=vec,
                              chunk_type_filter="contraindication", min_score=0.0)
            rest = retrieve(query, self.collection, self.embedder,
                            n_results=pool_n, query_embedding=vec, min_score=0.0)
            seen, merged = set(), []
            for c in _rr(safety) + _rr(rest):  # contraindication chunks first, each reranked
                key = c["text"][:80]
                if key not in seen:
                    seen.add(key)
                    merged.append(c)
            return merged
        return _rr(retrieve(query, self.collection, self.embedder,
                            n_results=pool_n, min_score=0.0))

    def retrieve_for_intent(self, query: str, intent: str,
                            comorbidities: list[str] | None = None,
                            n_results: int = TOP_K) -> list[dict]:
        """Safety-priority retrieval driven by ROUTED intent, fused with comorbidity-aware
        auxiliary retrieval (RRF) so the patient's background pathology is represented in context.

        Without comorbidities (or with the flag off) this returns exactly the old top-K primary
        list. With comorbidities it runs one auxiliary query per comorbidity and weighted-RRF-fuses
        them under the primary list (primary weight 1.0 > aux RRF_AUX_WEIGHT), keeping the total at
        n_results — comorbidity chunks displace only weak primary-tail chunks, never the primary
        top hits. See src/rag/config.py / fusion.py and the guardrail eval."""
        cand_n = max(n_results, RETRIEVE_CANDIDATES)
        primary = self._primary_retrieval(query, intent, cand_n)
        if not (COMORBIDITY_RETRIEVAL and comorbidities):
            return primary[:n_results]

        aux_lists = [
            retrieve(cq, self.collection, self.embedder, n_results=cand_n, min_score=0.0)
            for cq in comorbidity_queries(comorbidities, COMORBIDITY_QUERY_TEMPLATE,
                                          MAX_COMORBIDITY_QUERIES)
        ]
        return comorbidity_fuse(primary, aux_lists, n_results=n_results,
                                comorbidity_slots=COMORBIDITY_SLOTS, k=RRF_K,
                                min_score=COMORBIDITY_MIN_SCORE)

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
        # Comorbidity-aware retrieval needs the patient's active conditions (RRF-fused as
        # auxiliary queries). Derived generically from context; [] when no patient -> plain retrieval.
        comorbidities = comorbidity_names(patient_context)

        def _retrieve() -> tuple[list, float]:
            s = time.perf_counter()
            ch: list = []
            if routing["intent"] != "off_topic":
                try:
                    ch = self.retrieve_for_intent(query, routing["intent"],
                                                  comorbidities=comorbidities)
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

        # Deterministic comorbidity-conflict enforcement gate (Risk #1 backstop): catch a dangerous
        # recommendation × the patient's comorbidity that grounded generation can't (the
        # fluid-bolus-in-cirrhosis class). Attaches a mandatory warning; never deletes the answer.
        if COMORBIDITY_GATE_ENABLED:
            response = apply_comorbidity_gate(response, patient_context or {})

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
