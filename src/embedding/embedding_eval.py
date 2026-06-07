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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ on sys.path
from paths import CHUNKS_FILE, REPORT_FILE  # noqa: E402
from embedding.or_client import EmbeddingClient  # noqa: E402

MODELS = ["qwen/qwen3-embedding-8b", "openai/text-embedding-3-small"]
SAMPLE_SIZE = 50

# Full test set (10 queries) with an expected keyword per query for Hit@1.
TEST_CASES = [
    {"query": "quy trình điều trị sốc nhiễm khuẩn", "expected_keyword": "nhiễm khuẩn"},
    {"query": "chống chỉ định đặt nội khí quản", "expected_keyword": "nội khí quản"},
    {"query": "các bước tiến hành lọc máu liên tục", "expected_keyword": "lọc máu"},
    {"query": "theo dõi sau thở máy", "expected_keyword": "thở máy"},
    {"query": "xử trí tai biến chọc hút dịch màng phổi", "expected_keyword": "màng phổi"},
    {"query": "chăm sóc bệnh nhân hôn mê", "expected_keyword": "chăm sóc"},
    {"query": "quy trình truyền máu", "expected_keyword": "máu"},
    {"query": "cấp cứu ngừng tuần hoàn", "expected_keyword": "tuần hoàn"},
    {"query": "điều trị tăng kali máu", "expected_keyword": "kali"},
    {"query": "chống chỉ định lọc máu", "expected_keyword": "lọc máu"},
]


def _sample_chunks() -> list[dict]:
    """
    Build a SAMPLE_SIZE pool that contains at least one relevant chunk per query
    (so Hit@1 measures ranking quality, not sample luck) plus stride-sampled
    distractors spread across the corpus. Both models are scored on the same pool.
    """
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    selected: dict[str, dict] = {}

    for case in TEST_CASES:
        kw = case["expected_keyword"].lower()
        for c in chunks:
            if kw in c["text"].lower():
                selected[c["id"]] = c
                break

    step = max(1, len(chunks) // SAMPLE_SIZE)
    for c in chunks[::step] + chunks:  # stride distractors, then sequential fill
        if len(selected) >= SAMPLE_SIZE:
            break
        selected.setdefault(c["id"], c)

    return list(selected.values())[:SAMPLE_SIZE]


def evaluate_model(model_name: str, sample: list[dict]) -> dict:
    print(f"\n--- {model_name} ---")
    client = EmbeddingClient(model_name)

    t0 = time.time()
    # Batch to stay under the OpenRouter per-request token limit.
    texts = [c["text"] for c in sample]
    doc_vecs: list[list[float]] = []
    for i in range(0, len(texts), 16):
        doc_vecs.extend(client.embed(texts[i:i + 16]))
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
        "# Báo cáo Đánh giá — RAG Knowledge Base",
        f"**Ngày:** {time.strftime('%Y-%m-%d')}",
        "**Mô hình index:** qwen/qwen3-embedding-8b (qua OpenRouter)",
        "**Vector DB:** ChromaDB",
        "",
    ]
    section1 = [
        "## 1. So sánh mô hình embedding (Task 1.5)",
        "",
        f"Mẫu thử: {SAMPLE_SIZE} chunks · {len(TEST_CASES)} truy vấn (bộ test đầy đủ). "
        f"Pool gồm ít nhất một chunk liên quan cho mỗi truy vấn + các chunk gây nhiễu; "
        f"cả hai mô hình được chấm trên cùng một pool.",
        "",
        "| Mô hình | Hit@1 | Điểm TB | Thời gian embed |",
        "|---------|-------|---------|-----------------|",
    ]
    for r in results:
        section1.append(
            f"| {r['model']} | {r['hit_at_1']}/{r['total']} | "
            f"{r['avg_score']:.3f} | {r['load_time']:.1f}s |"
        )
    section1 += [
        "",
        f"**Kết luận:** pipeline dùng `qwen/qwen3-embedding-8b` "
        f"(Hit@1 cao hơn thuộc về `{chosen}`).",
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
