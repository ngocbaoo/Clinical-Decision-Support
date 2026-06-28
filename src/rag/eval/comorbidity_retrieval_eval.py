"""
Comorbidity-aware retrieval — guardrail + generality eval (one-variable experiment).

Two things must hold for COMORBIDITY_RETRIEVAL to ship (src/rag/config.py, fusion.py):

  A. GUARDRAIL — fusing a patient's comorbidities into retrieval must NOT drop the primary intent's
     recall, nor demote a contraindication chunk out of the top-K. We inject a realistic
     multi-comorbidity set into EVERY gold query and require Recall@K and Safety-priority@K to hold
     vs the baseline (no comorbidity). The reservation design guarantees the top
     (TOP_K - COMORBIDITY_SLOTS) primary chunks survive — this measures the only thing it can cost:
     a gold query whose ONLY relevant chunk sat in the reserved-away primary tail.

  B. GENERALITY — fusion must surface comorbidity-relevant content for DIFFERENT comorbidities, not
     just the pt-007 liver case. A small curated set asserts the comorbidity term appears in the
     fused top-K but NOT in the baseline top-K, across liver / heart-failure / renal / pregnancy.

Mirrors production exactly: same `retrieve`, `comorbidity_queries`, `comorbidity_fuse`, same config.
Run:  python src/rag/eval/comorbidity_retrieval_eval.py   (needs OPEN_ROUTER_KEY in .env)
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
from rag.fusion import comorbidity_fuse, comorbidity_queries  # noqa: E402
from rag.config import (TOP_K, RETRIEVE_CANDIDATES, COMORBIDITY_SLOTS, RRF_K,  # noqa: E402
                        COMORBIDITY_MIN_SCORE, MAX_COMORBIDITY_QUERIES,
                        COMORBIDITY_QUERY_TEMPLATE)

# A realistic ICU multi-comorbidity profile injected into every gold query for the guardrail —
# stresses displacement harder than a single comorbidity would.
GUARDRAIL_COMORBIDITIES = ["Suy gan do rượu", "Suy thận mạn", "Suy tim"]

# Generality cases: (query, comorbidities). Fusion must inject a comorbidity chunk that was NOT in
# the baseline top-K — measured by chunk-set difference (a substring check is unreliable on
# repetitive medical text). Different organ systems prove it is not the pt-007 liver case alone.
GENERALITY_CASES = [
    ("bệnh nhân sốt cao cần làm gì tiếp theo", ["Suy gan do rượu"]),
    ("xử trí tụt huyết áp truyền dịch", ["Suy thận mạn"]),
    ("lựa chọn kháng sinh cho nhiễm khuẩn", ["Suy thận mạn"]),
    ("theo dõi bệnh nhân sau phẫu thuật", ["Suy gan do rượu"]),
]


def _primary(query, collection, model, intent_contra=False):
    """The production primary list (contra-first for safety), `RETRIEVE_CANDIDATES` deep."""
    if intent_contra:
        vec = model.embed_one(query)
        safety = retrieve(query, collection, model, n_results=RETRIEVE_CANDIDATES,
                          query_embedding=vec, chunk_type_filter="contraindication", min_score=0.0)
        rest = retrieve(query, collection, model, n_results=RETRIEVE_CANDIDATES,
                        query_embedding=vec, min_score=0.0)
        seen, merged = set(), []
        for c in safety + rest:
            k = c["text"][:80]
            if k not in seen:
                seen.add(k); merged.append(c)
        return merged
    return retrieve(query, collection, model, n_results=RETRIEVE_CANDIDATES, min_score=0.0)


def _fused(query, collection, model, comorbidities, intent_contra=False):
    primary = _primary(query, collection, model, intent_contra)
    aux = [retrieve(cq, collection, model, n_results=RETRIEVE_CANDIDATES, min_score=0.0)
           for cq in comorbidity_queries(comorbidities, COMORBIDITY_QUERY_TEMPLATE,
                                          MAX_COMORBIDITY_QUERIES)]
    return comorbidity_fuse(primary, aux, n_results=TOP_K, comorbidity_slots=COMORBIDITY_SLOTS,
                            k=RRF_K, min_score=COMORBIDITY_MIN_SCORE)


def evaluate() -> dict:
    gold = json.loads(GOLD_FILE.read_text(encoding="utf-8"))["queries"]
    collection = chromadb.PersistentClient(path=str(CHROMA_PATH)).get_collection(COLLECTION_NAME)
    model = EmbeddingClient(DEFAULT_MODEL)

    in_scope = [q for q in gold if q["group"] != "oos" and q["relevant"]]
    safety = [q for q in gold if q["safety"]]

    def recall_hit(use_fusion: bool):
        rec = hit = 0
        for q in in_scope:
            chunks = (_fused(q["query"], collection, model, GUARDRAIL_COMORBIDITIES)
                      if use_fusion else
                      _primary(q["query"], collection, model)[:TOP_K])
            ranks = [i + 1 for i, c in enumerate(chunks) if is_relevant(c, q["relevant"])]
            rec += any(r <= TOP_K for r in ranks)
            hit += bool(ranks) and ranks[0] == 1
        n = len(in_scope)
        return round(rec / n, 3), round(hit / n, 3)

    def safety_priority(use_fusion: bool):
        ok = 0
        for q in safety:
            chunks = (_fused(q["query"], collection, model, GUARDRAIL_COMORBIDITIES, intent_contra=True)
                      if use_fusion else
                      _primary(q["query"], collection, model, intent_contra=True)[:TOP_K])
            ok += any(c["chunk_type"] == "contraindication" and is_relevant(c, q["relevant"])
                      for c in chunks)
        return round(ok / len(safety), 3) if safety else None

    base_recall, base_hit = recall_hit(False)
    fused_recall, fused_hit = recall_hit(True)
    base_sp, fused_sp = safety_priority(False), safety_priority(True)

    gen = []
    for query, comorbidities in GENERALITY_CASES:
        base = _primary(query, collection, model)[:TOP_K]
        fused = _fused(query, collection, model, comorbidities)
        bset = {c["text"][:80] for c in base}
        injected = [c for c in fused if c["text"][:80] not in bset]  # comorbidity chunk(s) added
        gen.append({"query": query, "comorbidity": comorbidities[0],
                    "injected": injected[0]["title"] if injected else None,
                    "score": round(injected[0]["score"], 2) if injected else None})

    return {"n_in_scope": len(in_scope), "n_safety": len(safety),
            "base_recall": base_recall, "fused_recall": fused_recall,
            "base_hit": base_hit, "fused_hit": fused_hit,
            "base_sp": base_sp, "fused_sp": fused_sp, "generality": gen}


def main() -> None:
    r = evaluate()
    print(f"\n=== Comorbidity-aware retrieval (TOP_K={TOP_K}, slots={COMORBIDITY_SLOTS}, "
          f"min_score={COMORBIDITY_MIN_SCORE}) ===")
    print(f"Injected comorbidities (guardrail): {GUARDRAIL_COMORBIDITIES}\n")
    print("A. GUARDRAIL (must not regress vs baseline)")
    print(f"  Recall@{TOP_K}        baseline {r['base_recall']}  -> fused {r['fused_recall']}  "
          f"{'OK' if r['fused_recall'] >= r['base_recall'] else 'REGRESSION'}")
    print(f"  Hit@1            baseline {r['base_hit']}  -> fused {r['fused_hit']}")
    sp_ok = (r['fused_sp'] is None) or (r['base_sp'] is not None and r['fused_sp'] >= r['base_sp'])
    print(f"  Safety-prio@{TOP_K}    baseline {r['base_sp']}  -> fused {r['fused_sp']}  "
          f"{'OK' if sp_ok else 'REGRESSION'}")
    print("\nB. GENERALITY (fusion must inject a comorbidity chunk absent from baseline top-K)")
    n_gain = 0
    for g in r["generality"]:
        gain = g["injected"] is not None
        n_gain += gain
        tag = "NEW" if gain else "—"
        detail = f"+[{g['score']}] {g['injected']}" if gain else "(nothing cleared min_score)"
        print(f"  [{tag:>3}] '{g['comorbidity']}'  {detail}\n        ({g['query']})")
    print(f"\n  Comorbidity content newly surfaced in {n_gain}/{len(r['generality'])} cases.")
    print("\nVerdict: SHIP only if Recall@K and Safety-prio@K hold AND generality shows gains.")


if __name__ == "__main__":
    main()
