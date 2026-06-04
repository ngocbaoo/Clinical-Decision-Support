"""
Task 1.5 — Embedding model comparison.

Compares two OpenRouter embedding models on a 20-chunk sample using 4 test
cases (Hit@1, avg top-1 score, embed time). Writes Section 1 of the report.

The indexing model stays qwen/qwen3-embedding-8b regardless of outcome
(per project decision); this comparison is informational.

Run:  python src/embedding/embedding_eval.py
"""

import json
import sys
import time
from pathlib import Path

import chromadb

sys.path.insert(0, str(Path(__file__).parent))
from or_client import EmbeddingClient  # noqa: E402

WEEK2_DIR = Path(__file__).resolve().parents[2]
CHUNKS_FILE = WEEK2_DIR / "chunks" / "icu_chunks.json"
REPORT_FILE = WEEK2_DIR / "chunks" / "evaluation_report.md"

MODELS = ["qwen/qwen3-embedding-8b", "openai/text-embedding-3-small"]
SAMPLE_SIZE = 20

TEST_CASES = [
    {"query": "quy trình đặt nội khí quản", "expected_keyword": "nội khí quản"},
    {"query": "chống chỉ định lọc máu", "expected_keyword": "chống chỉ định"},
    {"query": "điều trị septic shock", "expected_keyword": "septic shock"},
    {"query": "Vancomycin liều suy thận creatinine cao", "expected_keyword": "thận"},
]


def _sample_chunks() -> list[dict]:
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    # Spread the sample across the corpus so multiple procedures are represented.
    step = max(1, len(chunks) // SAMPLE_SIZE)
    return chunks[::step][:SAMPLE_SIZE]


def evaluate_model(model_name: str, sample: list[dict]) -> dict:
    print(f"\n--- {model_name} ---")
    client = EmbeddingClient(model_name)

    t0 = time.time()
    doc_vecs = client.embed([c["text"] for c in sample])
    load_time = time.time() - t0
    print(f"  embedded {len(sample)} chunks in {load_time:.1f}s (dim={client.dim})")

    # Temporary in-memory collection (dims differ per model).
    coll = chromadb.Client().create_collection(
        name=f"eval_{abs(hash(model_name))}",
        metadata={"hnsw:space": "cosine"},
    )
    coll.add(
        ids=[c["id"] for c in sample],
        embeddings=doc_vecs,
        documents=[c["text"] for c in sample],
    )

    hits, scores = 0, []
    for case in TEST_CASES:
        qv = client.embed_one(case["query"])
        res = coll.query(query_embeddings=[qv], n_results=3)
        top_doc = res["documents"][0][0]
        top_score = 1 - res["distances"][0][0]
        scores.append(top_score)
        hit = case["expected_keyword"].lower() in top_doc.lower()
        hits += int(hit)
        print(f"  {'HIT ' if hit else 'MISS'} [{top_score:.3f}] {case['query']}")

    return {
        "model": model_name,
        "hit_at_1": hits,
        "total": len(TEST_CASES),
        "avg_score": sum(scores) / len(scores),
        "load_time": load_time,
    }


def write_report_section(results: list[dict], chosen: str) -> None:
    header = [
        "# RAG Knowledge Base — Evaluation Report",
        f"**Date:** {time.strftime('%Y-%m-%d')}",
        "**Indexing model:** qwen/qwen3-embedding-8b (via OpenRouter)",
        "**Vector DB:** ChromaDB",
        "",
    ]
    section1 = [
        "## 1. Embedding Model Comparison (Task 1.5)",
        "",
        f"Sample: {SAMPLE_SIZE} chunks · {len(TEST_CASES)} cross-lingual test queries.",
        "",
        "| Model | Hit@1 | Avg Score | Embed time |",
        "|-------|-------|-----------|------------|",
    ]
    for r in results:
        section1.append(
            f"| {r['model']} | {r['hit_at_1']}/{r['total']} | "
            f"{r['avg_score']:.3f} | {r['load_time']:.1f}s |"
        )
    section1 += [
        "",
        f"**Decision:** indexing uses `qwen/qwen3-embedding-8b` "
        f"(higher Hit@1 was `{chosen}`; qwen3 is used for the pipeline regardless).",
        "",
    ]

    # Preserve Sections 2+ if a full report already exists.
    tail = ""
    if REPORT_FILE.exists():
        existing = REPORT_FILE.read_text(encoding="utf-8")
        idx = existing.find("## 2.")
        if idx != -1:
            tail = existing[idx:]

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(header + section1)
    REPORT_FILE.write_text(body + (tail if tail else ""), encoding="utf-8")
    print(f"\nWrote report Section 1 -> {REPORT_FILE}")


def main() -> None:
    sample = _sample_chunks()
    print(f"Comparing {len(MODELS)} models on {len(sample)} sample chunks...")
    results = [evaluate_model(m, sample) for m in MODELS]

    print("\n" + "=" * 60)
    print(f"{'Model':<34}| Hit@1 | Avg Score | Embed time")
    print("-" * 60)
    for r in results:
        print(f"{r['model']:<34}|  {r['hit_at_1']}/{r['total']}  |   "
              f"{r['avg_score']:.3f}   |  {r['load_time']:.1f}s")

    best = max(results, key=lambda r: (r["hit_at_1"], -r["load_time"]))
    write_report_section(results, best["model"])


if __name__ == "__main__":
    main()
