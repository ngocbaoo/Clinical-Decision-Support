"""
R6a — Retrieval quality evaluation on the gold set (LLM-free, embedding calls only).

Metrics (plan §4.2):
  Hit@1, Recall@5, MRR@5         per group + overall (oos excluded)
  Safety-priority rate           contraindication chunk in top-3 (safety groups)
  Keyword-detection rate         which safety queries the legacy keyword router catches
  OOS rejection + threshold calibration   top-1 score distributions in- vs out-of-scope

Run:  python src/rag/eval/retrieval_eval.py
Out:  chunks/rag_retrieval_eval.json (consumed by answer_eval.py's report)
"""

import json
import sys
import time
from pathlib import Path

import chromadb

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/ on sys.path
from paths import CHUNKS_DIR  # noqa: E402
from paths import CHROMA_PATH  # noqa: E402
from embedding.or_client import EmbeddingClient, DEFAULT_MODEL  # noqa: E402
from embedding.retriever import COLLECTION_NAME, SAFETY_KEYWORDS, retrieve  # noqa: E402

GOLD_FILE = Path(__file__).resolve().parent / "gold_retrieval.json"
OUT_FILE = CHUNKS_DIR / "rag_retrieval_eval.json"
TOP_K = 5


def is_relevant(chunk: dict, specs: list[dict]) -> bool:
    src = (chunk.get("source") or "").casefold()
    title = (chunk.get("title") or "").casefold()
    for spec in specs:
        s, t = spec["source"].casefold(), spec["title"].casefold()
        if (not s or s in src) and (not t or t in title):
            return True
    return False


def evaluate() -> dict:
    gold = json.loads(GOLD_FILE.read_text(encoding="utf-8"))["queries"]
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_collection(COLLECTION_NAME)
    model = EmbeddingClient(DEFAULT_MODEL)

    rows = []
    t0 = time.perf_counter()
    for q in gold:
        top = retrieve(q["query"], collection, model, n_results=TOP_K, min_score=0.0)
        rel_ranks = [i + 1 for i, c in enumerate(top) if is_relevant(c, q["relevant"])]

        # Safety-priority: contraindication-first merge (intent-driven, as the
        # pipeline does for routed contraindication intent).
        contra_top3_hit = None
        if q["safety"]:
            safety_chunks = retrieve(q["query"], collection, model, n_results=3,
                                     chunk_type_filter="contraindication",
                                     min_score=0.0)
            merged = safety_chunks + [c for c in top
                                      if c["chunk_type"] != "contraindication"]
            contra_top3_hit = any(
                c["chunk_type"] == "contraindication" and is_relevant(c, q["relevant"])
                for c in merged[:3])

        rows.append({
            "id": q["id"], "group": q["group"], "query": q["query"],
            "top1_score": round(top[0]["score"], 4) if top else 0.0,
            "top1_title": top[0]["title"][:60] if top else "",
            "hit1": bool(rel_ranks and rel_ranks[0] == 1) if q["relevant"] else None,
            "recall5": bool(rel_ranks) if q["relevant"] else None,
            "rr": (1.0 / rel_ranks[0]) if rel_ranks else 0.0,
            "contra_top3": contra_top3_hit,
            "keyword_detected": any(kw in q["query"].lower() for kw in SAFETY_KEYWORDS)
                                if q["safety"] else None,
        })
    elapsed = round(time.perf_counter() - t0, 1)

    in_scope = [r for r in rows if r["group"] != "oos"]
    oos = [r for r in rows if r["group"] == "oos"]

    def agg(items):
        n = len(items)
        if not n:
            return {}
        return {
            "n": n,
            "hit@1": round(sum(r["hit1"] for r in items) / n, 3),
            "recall@5": round(sum(r["recall5"] for r in items) / n, 3),
            "mrr@5": round(sum(r["rr"] for r in items) / n, 3),
        }

    groups = {}
    for g in ("vn_procedure", "safety_kw", "safety_para", "cross_lingual"):
        groups[g] = agg([r for r in in_scope if r["group"] == g])

    safety_rows = [r for r in rows if r["contra_top3"] is not None]
    in_scores = sorted(r["top1_score"] for r in in_scope)
    oos_scores = sorted(r["top1_score"] for r in oos)
    # Calibration: a threshold must reject every OOS query; report the gap.
    max_oos = max(oos_scores) if oos_scores else 0.0
    min_in = min(in_scores) if in_scores else 0.0
    suggested = round((max_oos + min_in) / 2, 3) if max_oos < min_in else max_oos + 0.01

    summary = {
        "overall": agg(in_scope),
        "by_group": groups,
        "safety_priority_rate": round(
            sum(r["contra_top3"] for r in safety_rows) / len(safety_rows), 3)
            if safety_rows else None,
        "keyword_detection_rate": round(
            sum(r["keyword_detected"] for r in safety_rows) / len(safety_rows), 3)
            if safety_rows else None,
        "oos": {
            "n": len(oos),
            "top1_scores": oos_scores,
            "max_top1": max_oos,
        },
        "in_scope_min_top1": min_in,
        "suggested_threshold": suggested,
        "threshold_separates": max_oos < min_in,
        "elapsed_s": elapsed,
    }
    return {"summary": summary, "rows": rows}


def main() -> None:
    result = evaluate()
    s = result["summary"]
    print("=== Retrieval evaluation ===")
    print(f"Overall (in-scope, n={s['overall']['n']}): "
          f"Hit@1={s['overall']['hit@1']}  Recall@5={s['overall']['recall@5']}  "
          f"MRR@5={s['overall']['mrr@5']}")
    for g, m in s["by_group"].items():
        if m:
            print(f"  {g:<14} n={m['n']:<3} Hit@1={m['hit@1']}  "
                  f"Recall@5={m['recall@5']}  MRR@5={m['mrr@5']}")
    print(f"Safety-priority rate (contra in top-3): {s['safety_priority_rate']}")
    print(f"Legacy keyword detection on safety queries: {s['keyword_detection_rate']}")
    print(f"OOS max top-1 score: {s['oos']['max_top1']:.3f} | "
          f"in-scope min top-1: {s['in_scope_min_top1']:.3f}")
    print(f"Suggested CONF_THRESHOLD: {s['suggested_threshold']} "
          f"(clean separation: {s['threshold_separates']})")

    fails = [r for r in result["rows"]
             if r["group"] != "oos" and r["recall5"] is False]
    if fails:
        print("\nMisses (no relevant chunk in top-5):")
        for r in fails:
            print(f"  {r['id']} [{r['group']}] \"{r['query']}\" "
                  f"-> top1: {r['top1_title']} ({r['top1_score']})")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\nSaved: {OUT_FILE}")


if __name__ == "__main__":
    main()
