# RAG Knowledge Base — Evaluation Report
**Date:** 2026-06-04
**Indexing model:** qwen/qwen3-embedding-8b (via OpenRouter)
**Vector DB:** ChromaDB

## 1. Embedding Model Comparison (Task 1.5)

Sample: 20 chunks · 4 cross-lingual test queries.

| Model | Hit@1 | Avg Score | Embed time |
|-------|-------|-----------|------------|
| qwen/qwen3-embedding-8b | 3/4 | 0.583 | 9.3s |
| openai/text-embedding-3-small | 2/4 | 0.466 | 2.0s |

**Decision:** indexing uses `qwen/qwen3-embedding-8b` (higher/›= Hit@1 was `qwen/qwen3-embedding-8b`; qwen3 chosen for the pipeline by project decision regardless).
