"""
TOP_K sweep — one-variable experiment: does passing fewer chunks to the generator
(config.TOP_K) keep the relevant chunk in context, and keep contraindications surfaced?

Generation latency is ~96% of wall-clock and ~92% prefill (docs/RAG_EVAL_REPORT.md). The
cheapest lever to cut prefill is feeding fewer chunks. This script measures the GUARDRAIL for
that change WITHOUT touching production: it retrieves top-5 once per gold query, then reports
Recall@K, Hit@1 and safety-priority@K (contraindication in top-K via the contra-first merge)
for K in {1,3,5}. Only lower config.TOP_K if Recall@K and safety-priority@K hold vs K=5.

Run:  python src/rag/eval/topk_sweep.py
Reuses: retrieval_eval.is_relevant + embedding.retriever.retrieve (same path as production).
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

KS = (1, 3, 4, 5)
RETRIEVE_N = 5  # we always pull 5, then evaluate cutoffs offline


def evaluate() -> dict:
    gold = json.loads(GOLD_FILE.read_text(encoding="utf-8"))["queries"]
    collection = chromadb.PersistentClient(path=str(CHROMA_PATH)).get_collection(COLLECTION_NAME)
    model = EmbeddingClient(DEFAULT_MODEL)

    in_scope = [q for q in gold if q["group"] != "oos" and q["relevant"]]
    safety = [q for q in gold if q["safety"]]

    # recall@K / hit@1 over in-scope queries
    rel_ranks_by_q = {}
    for q in in_scope:
        top = retrieve(q["query"], collection, model, n_results=RETRIEVE_N, min_score=0.0)
        rel_ranks_by_q[q["id"]] = [i + 1 for i, c in enumerate(top)
                                   if is_relevant(c, q["relevant"])]

    # safety-priority@K: contra-first merge (as the pipeline does), then check top-K
    contra_ranks_by_q = {}
    for q in safety:
        top = retrieve(q["query"], collection, model, n_results=RETRIEVE_N, min_score=0.0)
        contra = retrieve(q["query"], collection, model, n_results=3,
                          chunk_type_filter="contraindication", min_score=0.0)
        merged = contra + [c for c in top if c["chunk_type"] != "contraindication"]
        contra_ranks_by_q[q["id"]] = [
            i + 1 for i, c in enumerate(merged)
            if c["chunk_type"] == "contraindication" and is_relevant(c, q["relevant"])]

    rows = []
    n_in, n_safe = len(in_scope), len(safety)
    for k in KS:
        recall = sum(any(r <= k for r in rr) for rr in rel_ranks_by_q.values()) / n_in
        hit1 = sum(rr and rr[0] == 1 for rr in rel_ranks_by_q.values()) / n_in
        safety_pri = (sum(any(r <= k for r in cr) for cr in contra_ranks_by_q.values())
                      / n_safe) if n_safe else None
        rows.append({"k": k, "recall@k": round(recall, 3), "hit@1": round(hit1, 3),
                     "safety_priority@k": round(safety_pri, 3) if safety_pri is not None else None})
    return {"n_in_scope": n_in, "n_safety": n_safe, "rows": rows}


def main() -> None:
    res = evaluate()
    print(f"=== TOP_K sweep (in-scope n={res['n_in_scope']}, safety n={res['n_safety']}) ===")
    print(f"{'K':>3}{'Recall@K':>11}{'Hit@1':>9}{'Safety-priority@K':>20}")
    for r in res["rows"]:
        sp = "—" if r["safety_priority@k"] is None else r["safety_priority@k"]
        print(f"{r['k']:>3}{r['recall@k']:>11}{r['hit@1']:>9}{str(sp):>20}")
    base = next(r for r in res["rows"] if r["k"] == 5)
    print("\nGuardrail: lower config.TOP_K to K only if Recall@K and Safety-priority@K")
    print(f"hold vs K=5 (Recall@5={base['recall@k']}, Safety@5={base['safety_priority@k']}).")
    print("Targets: Recall@K >= 0.85, Safety-priority@K = 1.0.")


if __name__ == "__main__":
    main()
