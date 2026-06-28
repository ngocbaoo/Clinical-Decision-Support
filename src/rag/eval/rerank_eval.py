"""
Cross-encoder reranker — before/after eval (one-variable experiment).

The reranker targets PRECISION (which chunk is rank-1), not recall. So:
  - Hit@1 should RISE (the cross-encoder resolves the bi-encoder near-ties that let an off-topic
    chunk win rank-1).
  - Recall@K and Safety-priority@K must HOLD (reranking a pool that already contains the relevant
    chunk must not lose it; the contra-first ordering is preserved per sub-list).
  - Anchor case: the pt-007 "sốt cao" query whose bi-encoder rank-1 is the mislabeled antivenom
    adverse-reaction chunk ("XÉT NGHIỆM ĐỊNH LƯỢNG…") — the reranker should demote it out of rank-1.

Mirrors production: same `retrieve`, same pool size (RERANK_CANDIDATES), same CrossEncoderReranker.
Run:  python src/rag/eval/rerank_eval.py   (needs OPEN_ROUTER_KEY in .env + the reranker model)
"""

import json
import sys
from pathlib import Path

import chromadb

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/ on sys.path
from paths import CHROMA_PATH  # noqa: E402
from embedding.or_client import EmbeddingClient, DEFAULT_MODEL  # noqa: E402
from embedding.retriever import COLLECTION_NAME, retrieve  # noqa: E402
from rag.eval.retrieval_eval import GOLD_FILE, is_relevant  # noqa: E402
from rag.config import TOP_K, RERANK_CANDIDATES  # noqa: E402

ANCHOR_QUERY = "bệnh nhân đột nhiên sốt cao cần làm gì tiếp theo"
ANCHOR_JUNK = "ĐỊNH LƯỢNG"  # title fragment of the mislabeled antivenom chunk


def _topk(query, collection, model, rr, contra=False):
    """Production-mirroring top-K: wide bi-encoder pool -> (optional) rerank -> cut to TOP_K.
    contra=True front-loads contraindication chunks (each sub-list reranked), like the pipeline."""
    pool_n = RERANK_CANDIDATES if rr is not None else TOP_K
    if contra:
        vec = model.embed_one(query)
        safety = retrieve(query, collection, model, n_results=pool_n, query_embedding=vec,
                          chunk_type_filter="contraindication", min_score=0.0)
        rest = retrieve(query, collection, model, n_results=pool_n, query_embedding=vec, min_score=0.0)
        if rr is not None:
            safety, rest = rr.rerank(query, safety), rr.rerank(query, rest)
        seen, merged = set(), []
        for c in safety + rest:
            k = c["text"][:80]
            if k not in seen:
                seen.add(k); merged.append(c)
        return merged[:TOP_K]
    cand = retrieve(query, collection, model, n_results=pool_n, min_score=0.0)
    if rr is not None:
        cand = rr.rerank(query, cand)
    return cand[:TOP_K]


def evaluate() -> dict:
    gold = json.loads(GOLD_FILE.read_text(encoding="utf-8"))["queries"]
    collection = chromadb.PersistentClient(path=str(CHROMA_PATH)).get_collection(COLLECTION_NAME)
    model = EmbeddingClient(DEFAULT_MODEL)
    from rag.reranker import CrossEncoderReranker
    rr = CrossEncoderReranker()

    in_scope = [q for q in gold if q["group"] != "oos" and q["relevant"]]
    safety = [q for q in gold if q["safety"]]

    def recall_hit(reranker):
        rec = hit = 0
        for q in in_scope:
            chunks = _topk(q["query"], collection, model, reranker)
            ranks = [i + 1 for i, c in enumerate(chunks) if is_relevant(c, q["relevant"])]
            rec += any(r <= TOP_K for r in ranks)
            hit += bool(ranks) and ranks[0] == 1
        n = len(in_scope)
        return round(rec / n, 3), round(hit / n, 3)

    def safety_pri(reranker):
        ok = 0
        for q in safety:
            chunks = _topk(q["query"], collection, model, reranker, contra=True)
            ok += any(c["chunk_type"] == "contraindication" and is_relevant(c, q["relevant"])
                      for c in chunks)
        return round(ok / len(safety), 3) if safety else None

    base_recall, base_hit = recall_hit(None)
    rr_recall, rr_hit = recall_hit(rr)
    base_sp, rr_sp = safety_pri(None), safety_pri(rr)

    anchor_base = _topk(ANCHOR_QUERY, collection, model, None)
    anchor_rr = _topk(ANCHOR_QUERY, collection, model, rr)

    return {"n_in_scope": len(in_scope), "n_safety": len(safety),
            "base_recall": base_recall, "rr_recall": rr_recall,
            "base_hit": base_hit, "rr_hit": rr_hit, "base_sp": base_sp, "rr_sp": rr_sp,
            "anchor_base": [c["title"][:40] for c in anchor_base],
            "anchor_rr": [c["title"][:40] for c in anchor_rr]}


def main() -> None:
    r = evaluate()
    print(f"\n=== Cross-encoder rerank (pool={RERANK_CANDIDATES} -> TOP_K={TOP_K}) ===")
    print(f"  in-scope n={r['n_in_scope']}, safety n={r['n_safety']}\n")
    print(f"  {'metric':<18}{'bi-encoder':>12}{'+rerank':>10}")
    print(f"  {'Hit@1':<18}{r['base_hit']:>12}{r['rr_hit']:>10}   "
          f"{'↑' if r['rr_hit'] > r['base_hit'] else '=' if r['rr_hit']==r['base_hit'] else '↓ REGRESS'}")
    print(f"  {'Recall@'+str(TOP_K):<18}{r['base_recall']:>12}{r['rr_recall']:>10}   "
          f"{'OK' if r['rr_recall'] >= r['base_recall'] else 'REGRESSION'}")
    print(f"  {'Safety-prio@'+str(TOP_K):<18}{str(r['base_sp']):>12}{str(r['rr_sp']):>10}   "
          f"{'OK' if (r['rr_sp'] or 0) >= (r['base_sp'] or 0) else 'REGRESSION'}")
    print(f"\n  Anchor case ('{ANCHOR_QUERY}'):")
    jb = any(ANCHOR_JUNK in t for t in r["anchor_base"][:1])
    jr = any(ANCHOR_JUNK in t for t in r["anchor_rr"][:1])
    print(f"    bi-encoder rank-1: {r['anchor_base'][0]}  {'<-- JUNK' if jb else ''}")
    print(f"    +rerank   rank-1: {r['anchor_rr'][0]}  {'<-- JUNK (still!)' if jr else '(junk demoted)'}")
    print("\nVerdict: SHIP if Hit@1 ↑ (or =) while Recall@K and Safety-prio@K hold.")


if __name__ == "__main__":
    main()
