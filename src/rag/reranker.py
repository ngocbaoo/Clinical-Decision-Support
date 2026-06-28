"""
Cross-encoder reranker — second-stage retrieval precision.

The bi-encoder (embedding + Chroma ANN) ranks by cosine of two INDEPENDENTLY embedded vectors:
recall-friendly but weak on precision. For a vague query ("sốt cao cần làm gì tiếp theo") every
candidate clusters near the fallback threshold (~0.40) and an off-topic chunk can win rank-1 — e.g.
the mislabeled antivenom adverse-reaction section that poisoned the pt-007 answer (corticoid /
diphenhydramin / paracetamol-for-fever). A cross-encoder reads the (query, chunk) pair JOINTLY and
emits a single relevance logit, resolving exactly that near-tie. The pipeline retrieves a larger
bi-encoder pool, reranks it here, and keeps the top-K.

Model: BAAI/bge-reranker-v2-m3 (cross-lingual, strong on Vietnamese), a sequence-classification head
→ one relevance score. fp16 on CUDA, fp32 on CPU. Torch is ISOLATED to this module and it is
lazy-imported only in the live pipeline, so the offline catalog/scoring/test path stays torch-free —
the same contract as rag/nli_local.py and asr/*. `order_by_rerank` is a pure helper (no torch) so the
ordering logic is unit-testable offline.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ on sys.path
from rag.config import RERANK_MODEL, RERANK_MAX_LENGTH, RERANK_BATCH  # noqa: E402


def model_is_cached(model_id: str = RERANK_MODEL) -> bool:
    """True if the reranker weights are already in the local HF cache. Network-free and torch-free,
    so the pipeline can fast-skip reranking (→ bi-encoder only) when the model hasn't been downloaded
    yet, instead of hanging on a multi-GB fetch during the first live request."""
    try:
        from huggingface_hub import try_to_load_from_cache
        for fn in ("model.safetensors", "pytorch_model.bin", "model.onnx"):
            if isinstance(try_to_load_from_cache(model_id, fn), str):
                return True
    except Exception:  # noqa: BLE001 — any cache/hub error -> treat as unavailable
        return False
    return False


def order_by_rerank(chunks: list[dict], scores: list[float], *,
                    top_k: int | None = None) -> list[dict]:
    """Pure: return chunks sorted by `scores` (desc), each annotated with 'rerank_score'.

    Stable — equal scores keep their original (bi-encoder) order. Input dicts are not mutated.
    Torch-free, so the wiring is testable offline with injected scores.
    """
    order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    out = [{**chunks[i], "rerank_score": float(scores[i])} for i in order]
    return out[:top_k] if top_k is not None else out


class CrossEncoderReranker:
    """bge-reranker cross-encoder. `rerank(query, chunks)` reorders by joint relevance."""

    def __init__(self, model_id: str = RERANK_MODEL, max_length: int = RERANK_MAX_LENGTH,
                 batch_size: int = RERANK_BATCH):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, dtype=dtype).to(self.device).eval()
        self.max_length = max_length
        self.batch_size = batch_size

    def scores(self, query: str, docs: list[str]) -> list[float]:
        """Relevance logit per doc vs query (higher = more relevant). Batched to bound GPU memory."""
        if not docs:
            return []
        torch = self._torch
        out: list[float] = []
        for i in range(0, len(docs), self.batch_size):
            batch = docs[i:i + self.batch_size]
            enc = self.tokenizer([query] * len(batch), batch, padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits.view(-1).float().cpu().tolist()
            out.extend(logits)
        return out

    def rerank(self, query: str, chunks: list[dict], *, top_k: int | None = None,
               text_key: str = "text") -> list[dict]:
        """Return `chunks` sorted by cross-encoder relevance (desc), annotated 'rerank_score'."""
        if not chunks:
            return []
        scs = self.scores(query, [c.get(text_key, "") for c in chunks])
        return order_by_rerank(chunks, scs, top_k=top_k)
