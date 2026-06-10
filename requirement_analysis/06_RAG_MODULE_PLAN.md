# 06 · RAG Module Build Plan + Evaluation Metrics
# ASR + RAG Clinical Assistant — ICU Decision Support
**Ngày lập:** 2026-06-10 · **Trạng thái:** Draft
**Cơ sở:** PRD §7.2 (RAG stack), Evaluation Plan Tier 2 (scenario rubric), code hiện có trong `src/`

---

## 1. Mục tiêu và phạm vi

Nối ba thành phần đã hoàn thiện thành một pipeline trả lời câu hỏi có trích dẫn:

| Đã có | Thiếu (phạm vi plan này) |
|-------|--------------------------|
| Retriever + safety-priority routing (`src/embedding/retriever.py`) | Query orchestration (intent → retrieval) |
| `patient_context` từ FHIR (`src/fhir/fhir_client.py`) | Context assembly (chunks + patient + scores → prompt) |
| 5 calculator (`src/scoring/calculator.py` — `calculate_all()`) | LLM generation với citation bắt buộc + fallback |
| ChromaDB 4 nguồn guideline, SQLite LOINC/ICD-10 | Allergy-first safety gate |
| 17 mock patients + index.json | Evaluation harness (retrieval + answer quality) |

**Out of scope:** ASR (Tier 1 riêng), drug-interaction API (DrugBank — tách thành task sau, pipeline để sẵn hook), UI.

---

## 2. Kiến trúc module — `src/rag/`

```
Câu hỏi (text) + patient (--file data/mock/patient_X.json)
        │
        ▼
[1] query_router.py   — phân loại intent (LLM 1 call, JSON output):
        │               {procedure | contraindication | dosing | scoring | general | off-topic}
        │               + trích tên thuốc/thủ thuật được nhắc đến
        ▼
[2] Parallel context build:
        ├─ retriever.retrieve_with_safety_priority()  (top-k chunks, đã có)
        ├─ FHIRClient.from_file().build_patient_context()  (đã có)
        └─ calculate_all(patient_context)  (đã có)
        ▼
[3] safety.py         — allergy gate: so khớp thuốc trong câu hỏi với
        │               AllergyIntolerance → cảnh báo LUÔN ĐỨNG ĐẦU response (Safety Req #2)
        ▼
[4] context_builder.py — lắp prompt: chunks đánh số [1]..[k] kèm (source, title),
        │               patient summary, scores, missing-data list
        ▼
[5] generator.py      — LLM qua OpenRouter (mở rộng or_client thành chat client).
        │               Structured output: {answer, citations[], confidence, disclaimer}
        │               Fallback: top-1 score < THRESHOLD hoặc citations rỗng
        │               → "Không đủ thông tin để khuyến nghị" (F-RAG-09, US-05)
        ▼
[6] Response: alert (nếu có) → khuyến nghị ngắn gọn → citations → disclaimer
```

Quy ước hiện có giữ nguyên: bootstrap `src/` lên `sys.path`, import từ `paths`, UTF-8 stdout, status ra stderr để `--json` sạch.

### Nguyên tắc chống hallucination (PRD §7.3)
1. Prompt chỉ cho phép trả lời từ các chunk được đánh số; mọi câu khuyến nghị phải kèm `[n]`.
2. Post-check bằng code (không tin LLM): response không có citation hợp lệ → thay bằng fallback.
3. Confidence threshold trên retrieval score — **calibrate bằng eval, không chọn tay** (xem §4.3).
4. Disclaimer cố định: "Cần bác sĩ xác nhận trước khi thực hiện."

---

## 3. Kế hoạch thực hiện (6 task, ~2 tuần)

| # | Task | Nội dung | Acceptance |
|---|------|----------|------------|
| R1 | LLM chat client | Mở rộng `or_client.py`: `ChatClient` (OpenRouter chat completions, retry/backoff như embed, JSON mode) | Gọi được model qua `.env` key; unit test mock |
| R2 | Query router | Intent classification + entity extraction (thuốc, thủ thuật) — 1 LLM call, few-shot, JSON | ≥90% đúng intent trên 30 câu test tự tạo |
| R3 | Safety gate + context builder | Allergy match (exact + nhóm thuốc cơ bản, vd penicillin↔amoxicillin); prompt assembly với citation IDs; liệt kê data thiếu | S-01 luôn ra cảnh báo đầu tiên; prompt < 8k tokens |
| R4 | Generator + fallback | Structured output, citation post-check, threshold fallback, off-topic refusal (S-09) | Citation rate 100% trên mọi response không-fallback |
| R5 | CLI demo `src/rag/ask.py` | `python src/rag/ask.py --file data/mock/patient_A.json --query "..." [--json]` + đo latency từng stage | Chạy end-to-end offline trừ 2 API call |
| R6 | Evaluation harness `src/rag/eval/` | Gold set + retrieval metrics + answer metrics (§4, §5), xuất `chunks/rag_eval_report.md` | Báo cáo tự động sinh, so với Definition of Success |

Thứ tự: R1 → (R2 ∥ R3) → R4 → R5 → R6. Drug-interaction check (F-RAG-05) để hook trống trong safety.py, gắn DrugBank/RxNav sau.

---

## 4. Metrics — Retrieval quality

### 4.1 Gold set (điều kiện tiên quyết)
Bộ eval 10 query hiện tại chỉ check *chunk_type* — không đo đúng tài liệu. Cần nâng cấp:

- **~40 query** có nhãn: 10 query cũ + 15 query lâm sàng mới (từ 20 scenario Tier 2) + 5 query tiếng Anh/sepsis (đo cross-lingual với chunk SSC 2021) + 5 query paraphrase an toàn không chứa keyword "chống chỉ định" (đo lỗ hổng keyword routing) + 5 query out-of-scope/off-topic.
- Mỗi query gán **chunk IDs liên quan** (1–3 id/query, duyệt tay từ `icu_chunks.json`) → lưu `src/rag/eval/gold_retrieval.json`.
- ID chunk thay đổi khi re-chunk → gold set lưu thêm `(source, title)` để re-map.

### 4.2 Metrics

| Metric | Định nghĩa | Target |
|--------|-----------|--------|
| **Hit@1** | top-1 là chunk liên quan | ≥ 70% |
| **Recall@5** | ≥1 chunk liên quan trong top-5 | ≥ 85% |
| **MRR@5** | mean reciprocal rank của chunk liên quan đầu tiên | ≥ 0.75 |
| **Safety-priority rate** | query an toàn (kể cả paraphrase) có chunk contraindication trong top-3 | 100% trên query có keyword; báo cáo riêng nhóm paraphrase |
| **Cross-lingual Hit@5** | query EN/sepsis tìm được chunk SSC | ≥ 60% (báo cáo, chưa gate) |
| **Out-of-scope rejection** | query ngoài phạm vi có top-1 score < threshold | 100% |

### 4.3 Calibrate confidence threshold
Vẽ phân bố top-1 score của (a) query có nhãn liên quan vs (b) query out-of-scope trên gold set → chọn threshold tách hai nhóm (ưu tiên zero false-accept cho nhóm b). Threshold này dùng cho fallback F-RAG-09. Hiện `min_score=0.5` chưa từng được kiểm chứng — đây là cách kiểm chứng.

---

## 5. Metrics — Answer quality

### 5.1 Tự động (chạy mỗi lần sửa pipeline, không cần người)

| Metric | Cách đo | Target (DoS) |
|--------|---------|--------------|
| **Citation rate** | % response không-fallback có ≥1 citation hợp lệ (id tồn tại trong context) — check bằng code | 100% |
| **Citation precision** | % citation trỏ đúng chunk thực sự chứa nội dung được claim — LLM-as-judge + spot-check tay 20% | ≥ 90% |
| **Faithfulness / hallucination rate** | Tách response thành từng claim → judge LLM kiểm tra claim có được chunk trích dẫn hỗ trợ không. Hallucination = ≥1 claim không có căn cứ | 0% unsupported claim |
| **Refusal correctness** | Query off-topic/thiếu data → có fallback đúng không (false-answer = critical fail); query hợp lệ → không refuse oan (over-refusal rate) | False-answer 0%; over-refusal < 10% |
| **Allergy detection** | Scenario có dị ứng (S-01 pattern, chạy trên patient mock có AllergyIntolerance) → alert xuất hiện và đứng đầu | 100% (hard gate) |
| **Score consistency** | NEWS2/qSOFA/eGFR trong câu trả lời khớp `calculate_all()` (regex so số) | 100% |
| **Độ dài** | Response đọc được < 10s (~ ≤120 từ phần khuyến nghị) | ≥ 90% |
| **Latency** | per-stage (router, retrieval, generation) + end-to-end | RAG-only < 4.5s (chừa 2s cho ASR trong budget 5s tổng) |

> Judge LLM dùng model khác model generation (tránh self-preference), prompt judge kèm rubric + chunk gốc; mọi verdict "hallucination" phải được người xác nhận trước khi tính vào báo cáo cuối.

### 5.2 Human rubric (theo Evaluation Plan Tier 2, giữ nguyên)
20 scenario S-01…S-20, mỗi scenario chấm 5 tiêu chí × 0–2 điểm: **Correctness, Safety, Source citation, Completeness, Clarity.**

- Pass: ≥ 7/10 mỗi scenario; pass rate mục tiêu ≥ 85% (tối thiểu 70%).
- **S-01→S-05 (critical safety) bắt buộc 10/10** — không đạt thì không demo.
- Validator: clinician nếu có; nếu không → mentor + cross-check guideline gốc, ghi rõ limitation.
- Mỗi scenario map sẵn vào 1 mock patient (vd S-01 → patient có allergy Penicillin; S-02 → patient Creatinine cao; S-03 → patient sepsis). Nếu cohort thiếu, bổ sung 2–3 mock patient qua `generate_mock_patients.py`.

### 5.3 Failure analysis
Mọi case fail (tự động hoặc rubric) ghi theo template §8 của Evaluation Plan, gán root cause: `retrieval / generation / safety-gate / calculation / FHIR data` — để tuần optimization biết sửa tầng nào.

---

## 6. Definition of Done cho RAG module

1. `ask.py` chạy end-to-end trên 17 mock patient, citation rate 100%, allergy gate 100%.
2. Retrieval: Recall@5 ≥ 85%, MRR ≥ 0.75 trên gold set 40 query; threshold đã calibrate.
3. Answer: 0% hallucination trên bộ tự động; ≥ 14/20 scenario pass rubric; S-01→S-05 đạt 10/10.
4. `chunks/rag_eval_report.md` sinh tự động, kèm failure analysis.
5. Latency RAG-only < 4.5s p50 (đo trên 40 query).

## 7. Rủi ro chính

| Rủi ro | Mitigation |
|--------|------------|
| Judge LLM chấm faithfulness sai | Spot-check tay 20%, judge khác model generation, verdict hallucination phải người confirm |
| Gold set chunk ID trôi khi re-chunk | Lưu (source, title) kèm ID để re-map |
| Keyword safety routing bỏ sót paraphrase | Router R2 phân loại intent bằng LLM thay vì chỉ keyword; đo riêng nhóm paraphrase trong §4.2 |
| GPT-4o (PRD) vs OpenRouter | Dùng OpenRouter làm gateway (đã có key/hạ tầng), model chọn được qua config — thử 2–3 model trong R6 |
| Prompt vượt context với patient phức tạp | context_builder cắt theo ưu tiên: allergy > scores > top-3 chunks > meds > phần còn lại |
