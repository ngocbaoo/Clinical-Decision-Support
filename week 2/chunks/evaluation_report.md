# RAG Knowledge Base — Evaluation Report
**Date:** 2026-06-04
**Indexing model:** qwen/qwen3-embedding-8b (4096-dim, via OpenRouter)
**Vector DB:** ChromaDB (cosine, persistent at `./chroma_db`)

> **Architecture note.** Embeddings are served by the OpenRouter `/embeddings`
> endpoint rather than a local model. No `torch` / `sentence-transformers` / GPU is
> used; ChromaDB stores precomputed, L2-normalized vectors. The SQLite lookup DB
> (`db/clinical_db.sqlite`, LOINC + ICD-10) was built in Task 0 and is a standalone
> artifact — it is **not** wired into the vector retriever.

## 1. Embedding Model Comparison (Task 1.5)

Sample: 20 chunks · 4 cross-lingual test queries.

| Model | Hit@1 | Avg Score | Embed time |
|-------|-------|-----------|------------|
| qwen/qwen3-embedding-8b | 3/4 | 0.583 | 9.3s |
| openai/text-embedding-3-small | 2/4 | 0.466 | 2.0s |

**Decision:** `qwen/qwen3-embedding-8b` — higher Hit@1 and stronger cross-lingual /
medical-term separation (both models missed only "điều trị septic shock", whose
target procedure is absent from the 20-chunk sample). text-embedding-3-small is
~4× faster but consistently lower-scoring on Vietnamese clinical text.

## 2. Knowledge Base Statistics

- Total chunks indexed: **420**
  - Procedure chunks: **101**
  - Procedure-section chunks: **212**
  - Contraindication chunks: **107**
- Chunk size (chars): min **162** / avg **3350** / max **6496**
- `chunks/icu_chunks.json`: ~2.0 MB
- ChromaDB index (`chroma_db/`): ~22 MB
- Lookup DB (`db/clinical_db.sqlite`): ~1.8 MB — 28 LOINC rows, 7,900 ICD-10 rows

> Note on counts: the task brief anticipated ~232 procedure chunks with few
> section chunks. In reality the source procedures average ~7,000 chars (half
> exceed 6,000), so larger procedures are packed into ≤6,000-char
> `procedure_section` chunks to keep embeddings within a sensible token budget and
> improve retrieval granularity. `procedure` + `procedure_section` together
> represent the 187 procedures that passed the structure filter.

### Retrieval Quality (10 test queries)

Safety-keyword queries are routed contraindication-first; `procedure_section`
counts as a `procedure`-family top result.

| # | Query | Top-1 type | Score | Expected | Pass |
|---|-------|-----------|-------|----------|------|
| 1 | quy trình điều trị sốc nhiễm khuẩn | procedure_section | 0.54 | procedure | ✅* |
| 2 | chống chỉ định đặt nội khí quản | contraindication | 0.75 | contraindication | ✅ |
| 3 | các bước tiến hành lọc máu liên tục | procedure_section | 0.71 | procedure | ✅ |
| 4 | theo dõi sau thở máy | procedure_section | 0.68 | procedure | ✅ |
| 5 | xử trí tai biến chọc hút dịch màng phổi | procedure | 0.66 | contraindication | ❌ |
| 6 | chăm sóc bệnh nhân hôn mê | procedure | 0.66 | procedure | ✅ |
| 7 | quy trình truyền máu | contraindication | 0.60 | procedure | ❌ |
| 8 | cấp cứu ngừng tuần hoàn | procedure | 0.71 | procedure | ✅ |
| 9 | điều trị tăng kali máu | procedure | 0.72 | procedure | ✅ |
| 10 | chống chỉ định lọc máu | contraindication | 0.72 | contraindication | ✅ |

**Pass rate: 8/10** (meets the Definition of Done ≥ 8/10).

\* Q1 matches the expected *type* but the top document is only weakly relevant
(score 0.54 < 0.6) — see failure analysis.

## 3. Failure Analysis

- **Q1 "sốc nhiễm khuẩn" (septic shock)** — passes on chunk_type but the top hit
  (DẪN LƯU NÃO THẤT) is irrelevant and scores 0.54. The BYT 2014 manual is a
  *technique* catalogue and contains no dedicated septic-shock management protocol;
  "nhiễm khuẩn" appears only as scattered mentions, so no chunk dominates.
- **Q7 "truyền máu" (blood transfusion)** — fails: there is no blood-transfusion
  procedure in the corpus, so the query drifts to plasma-exchange (THAY HUYẾT TƯƠNG)
  contraindication chunks at ~0.60.
- **Q5 "xử trí tai biến chọc hút dịch màng phổi"** — fails against the brief's
  expectation of a *contraindication* top result, but semantically the query asks
  about handling **complications** (TAI BIẾN), which live in procedure chunks, not
  CHỐNG CHỈ ĐỊNH. The expectation in the brief looks mismatched; the retriever's
  procedure-type answer is arguably correct behaviour (though the specific top hit,
  "truyền dịch", is still off-target).

## 4. Known Limitations

- **Corpus coverage**: pure ICU *techniques*; disease-management topics
  (septic shock, transfusion protocols) are not first-class entries.
- **Cross-lingual retrieval untested at scale**: only the 4 Task-1.5 probes touched
  VI→EN; the 10-query eval is VI→VI.
- **OpenRouter dependency**: indexing and querying both require network access and
  the `OPEN_ROUTER_KEY`; there is no local fallback.
- **ICD-10 parse is best-effort**: the bilingual source wraps mid-description, so
  some `icd10_codes` rows carry interleaved EN/VI text or appended Incl./Excl. notes.
- **Title noise**: a few `## ` headings merge title with body; titles are cut at the
  first roman-numeral section marker, which is heuristic.
- **Score calibration**: relevant top-1 scores cluster ~0.6–0.75; a single global
  `min_score` is coarse across query types.

## 5. Next Steps (Tuần 3)

- Integrate with the FHIR pipeline (map retrieved procedures → LOINC/ICD lookups in
  `clinical_db.sqlite`).
- Add English guidelines (SSC 2021 for sepsis, NEWS2) to close the septic-shock and
  scoring gaps surfaced by Q1.
- Connect retrieval to the ASR module for spoken clinical queries.
- Add a hybrid path: detect LOINC/ICD codes in queries and enrich vector results
  from the SQLite lookup tables.
- Per-type score thresholds and a re-ranking pass to lift weak top-1 cases (Q1, Q7).
