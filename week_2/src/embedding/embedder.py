"""
Task 2 — Embed chunks via OpenRouter and index into ChromaDB.

Model:       qwen/qwen3-embedding-8b (L2-normalized, cosine space)
Collection:  clinical_knowledge  (persistent at ./chroma_db)
Idempotent:  the collection is dropped and recreated on every run.

Run:  python src/embedding/embedder.py
"""

import json
import sys
from pathlib import Path

import chromadb

sys.path.insert(0, str(Path(__file__).parent))
from or_client import EmbeddingClient, DEFAULT_MODEL  # noqa: E402

WEEK2_DIR = Path(__file__).resolve().parents[2]
CHUNKS_FILE = WEEK2_DIR / "chunks" / "icu_chunks.json"
CHROMA_PATH = WEEK2_DIR / "chroma_db"
COLLECTION_NAME = "clinical_knowledge"


def load_embedding_model(model_name: str = DEFAULT_MODEL) -> EmbeddingClient:
    client = EmbeddingClient(model_name)
    # Probe the dimension with a tiny call so we can report it.
    client.embed_one("dimension probe")
    print(f"Loading {model_name}...")
    print(f"Model loaded on: OpenRouter API (dim={client.dim})")
    return client


def init_chromadb(persist_path: str | Path = CHROMA_PATH) -> tuple:
    print("\nInitializing ChromaDB...")
    client = chromadb.PersistentClient(path=str(persist_path))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # didn't exist
    collection = client.create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    print(f"Collection '{COLLECTION_NAME}' ready.")
    return client, collection


def _flatten_metadata(chunk: dict) -> dict:
    """ChromaDB metadata must be flat scalars (no nested dicts)."""
    m = chunk["metadata"]
    return {
        "title": chunk["title"],
        "source": chunk["source"],
        "language": chunk["language"],
        "chunk_type": chunk["chunk_type"],
        "procedure_title": m["procedure_title"],
        "has_contraindication": bool(m["has_contraindication"]),
        "has_steps": bool(m["has_steps"]),
        "is_partial": bool(m["is_partial"]),
        "char_count": int(m["char_count"]),
        "type": m["type"],
    }


def embed_and_index(chunks: list[dict], collection, model: EmbeddingClient,
                    batch_size: int = 16) -> None:
    total = len(chunks)
    print("\nIndexing chunks...")
    for start in range(0, total, batch_size):
        batch = chunks[start:start + batch_size]
        vecs = model.embed([c["text"] for c in batch])
        collection.add(
            ids=[c["id"] for c in batch],
            embeddings=vecs,
            documents=[c["text"] for c in batch],
            metadatas=[_flatten_metadata(c) for c in batch],
        )
        done = min(start + batch_size, total)
        print(f"Indexed {done}/{total}{' ✅' if done == total else '...'}")


def verify_index(collection, model: EmbeddingClient) -> None:
    print("\nVerification:")
    print(f"Total documents: {collection.count()}")
    query = "quy trình đặt nội khí quản"
    print(f'Sample query: "{query}"')
    res = collection.query(query_embeddings=[model.embed_one(query)], n_results=3)
    for rank, (doc, meta, dist) in enumerate(
        zip(res["documents"][0], res["metadatas"][0], res["distances"][0]), start=1
    ):
        title = (meta.get("title") or doc)[:60]
        print(f"  {rank}. [{1 - dist:.3f}] {title}...")


def main() -> None:
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    model = load_embedding_model()
    _, collection = init_chromadb()
    embed_and_index(chunks, collection, model)
    verify_index(collection, model)
    print(f"\nDone. ChromaDB saved to: {CHROMA_PATH}")


if __name__ == "__main__":
    main()
