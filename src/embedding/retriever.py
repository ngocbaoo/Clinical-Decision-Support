"""
Task 3 — Retrieval interface for the RAG pipeline.

- retrieve(): semantic search with optional chunk_type filter + min score.
- retrieve_with_safety_priority(): surfaces contraindication chunks first when
  the query contains safety keywords.
- run_evaluation(): 10 fixed test queries, formatted output.

Run:  python src/embedding/retriever.py
"""

import sys
from pathlib import Path

import chromadb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ on sys.path
from paths import CHROMA_PATH  # noqa: E402
from embedding.or_client import EmbeddingClient, DEFAULT_MODEL  # noqa: E402

COLLECTION_NAME = "clinical_knowledge"

SAFETY_KEYWORDS = ("chống chỉ định", "không được dùng", "contraindication", "nguy hiểm")

# (query, expected_top_chunk_type) per Task 3 acceptance criteria.
EVAL_QUERIES = [
    ("quy trình điều trị sốc nhiễm khuẩn", "procedure"),
    ("chống chỉ định đặt nội khí quản", "contraindication"),
    ("các bước tiến hành lọc máu liên tục", "procedure"),
    ("theo dõi sau thở máy", "procedure"),
    ("xử trí tai biến chọc hút dịch màng phổi", "contraindication"),
    ("chăm sóc bệnh nhân hôn mê", "procedure"),
    # Blood transfusion is time-critical and intentionally out of scope for this
    # RAG system — excluded from the pass/fail count (reported as N/A).
    ("quy trình truyền máu", "out-of-scope"),
    ("cấp cứu ngừng tuần hoàn", "procedure"),
    ("điều trị tăng kali máu", "procedure"),
    ("chống chỉ định lọc máu", "contraindication"),
]

# procedure_section chunks are procedure content and satisfy a "procedure" expectation.
PROCEDURE_FAMILY = {"procedure", "procedure_section"}


def _open_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_collection(COLLECTION_NAME)


def retrieve(query: str, collection, model: EmbeddingClient, n_results: int = 3,
             chunk_type_filter: str | None = None, min_score: float = 0.5,
             query_embedding: list[float] | None = None) -> list[dict]:
    """Semantic search; returns results with score >= min_score.

    Pass query_embedding to reuse an already-computed vector (saves an API call
    when the same query is retrieved with different filters).
    """
    where = {"chunk_type": chunk_type_filter} if chunk_type_filter else None
    res = collection.query(
        query_embeddings=[query_embedding or model.embed_one(query)],
        n_results=n_results,
        where=where,
    )
    out = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        score = 1 - dist
        if score < min_score:
            continue
        out.append({
            "text": doc,
            "title": meta.get("title", ""),
            "source": meta.get("source", ""),
            "chunk_type": meta.get("chunk_type", ""),
            "score": score,
        })
    return out


def retrieve_with_safety_priority(query: str, collection, model: EmbeddingClient,
                                  n_results: int = 3) -> dict:
    """If the query is safety-related, fetch contraindication chunks first."""
    is_safety = any(kw in query.lower() for kw in SAFETY_KEYWORDS)
    if is_safety:
        safety = retrieve(query, collection, model, n_results=n_results,
                          chunk_type_filter="contraindication", min_score=0.0)
        procedures = retrieve(query, collection, model, n_results=n_results,
                             chunk_type_filter="procedure", min_score=0.0)
    else:
        safety = []
        procedures = retrieve(query, collection, model, n_results=n_results,
                             min_score=0.0)
    return {"safety_chunks": safety, "procedure_chunks": procedures, "query": query}


def _top3_for_eval(query: str, collection, model: EmbeddingClient) -> list[dict]:
    """Top-3 results, applying safety-priority routing for safety queries."""
    if any(kw in query.lower() for kw in SAFETY_KEYWORDS):
        res = retrieve_with_safety_priority(query, collection, model, n_results=3)
        merged = res["safety_chunks"] + res["procedure_chunks"]
        return merged[:3]
    return retrieve(query, collection, model, n_results=3, min_score=0.0)


def run_evaluation(collection, model: EmbeddingClient) -> int:
    print(f"Running {len(EVAL_QUERIES)} test queries...\n")
    passed = 0
    scored = 0  # queries that count toward pass rate (out-of-scope excluded)
    for i, (query, expected) in enumerate(EVAL_QUERIES, start=1):
        results = _top3_for_eval(query, collection, model)
        top = results[0] if results else None

        if expected == "out-of-scope":
            verdict = "N/A "
        else:
            scored += 1
            if top is None:
                ok = False
            elif expected == "contraindication":
                ok = top["chunk_type"] == "contraindication"
            else:  # procedure family
                ok = top["chunk_type"] in PROCEDURE_FAMILY
            passed += int(ok)
            verdict = "PASS" if ok else "FAIL"

        print(f'Query {i}: "{query}"   [expect: {expected}]  {verdict}')
        for rank, r in enumerate(results, start=1):
            ctype = {"contraindication": "contraindic."}.get(
                r["chunk_type"], r["chunk_type"]
            )
            print(f"  {rank}. [{r['score']:.2f}] {ctype:<17} | {r['title'][:50]}")
        print()

    print(f"Pass rate: {passed}/{scored} "
          f"({len(EVAL_QUERIES) - scored} query out-of-scope, excluded)")
    return passed


def main() -> None:
    collection = _open_collection()
    model = EmbeddingClient(DEFAULT_MODEL)
    run_evaluation(collection, model)


if __name__ == "__main__":
    main()
