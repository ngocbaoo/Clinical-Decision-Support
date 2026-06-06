# 05 · Requirement Analysis
# ASR + RAG Clinical Assistant — ICU Decision Support
**Phiên bản:** 1.0 · **Trạng thái:** Draft — chờ mentor review
**Ngày lập:** 01/06/2026 · **Tác giả:** Tạ Bảo Ngọc

---

## 1. Mục đích và phạm vi tài liệu

### 1.1 Mục đích
Tài liệu này phân tích và hệ thống hoá toàn bộ yêu cầu của PoC **ASR + RAG Clinical Assistant** — từ nhu cầu nghiệp vụ (business need) → yêu cầu người dùng (user requirement) → yêu cầu hệ thống (functional / non-functional) → tiêu chí nghiệm thu. Mục tiêu là tạo một điểm tham chiếu duy nhất, có **traceability** từ pain point đến test case, phục vụ phát triển và đánh giá trong 6 tuần.

### 1.2 Phạm vi (PoC 6 tuần)

| In scope | Out of scope |
|----------|-------------|
| ASR pipeline (push-to-talk, model tiếng Việt + medical) | Real-time monitor integration |
| RAG pipeline với medical knowledge base | Multi-speaker diarization |
| FHIR R4 query với synthetic patient data | Noise cancellation phức tạp |
| Medication safety check + allergy alert | Trend analysis theo thời gian (time series) |
| Tự tính NEWS2 / qSOFA / eGFR từ FHIR data | HIS/EMR integration thật |
| End-to-end demo với clinical scenarios | Production deployment |

### 1.3 Định nghĩa & viết tắt
ASR = Automatic Speech Recognition · RAG = Retrieval-Augmented Generation · FHIR = Fast Healthcare Interoperability Resources · CDSS = Clinical Decision Support System · WER = Word Error Rate · PHI = Protected Health Information · NEWS2 = National Early Warning Score 2 · qSOFA = quick Sequential Organ Failure Assessment · eGFR = estimated Glomerular Filtration Rate · TDM = Therapeutic Drug Monitoring.

---

## 2. Bối cảnh nghiệp vụ (Business Context)

### 2.1 Vấn đề
Trong môi trường ICU, bác sĩ phải ra quyết định lâm sàng nhanh (chọn thuốc, chỉnh liều, kiểm tra tương tác) **trong khi tay đang bận** và thiết bị liên tục sinh ra lượng lớn dữ liệu không được tổng hợp real-time. Khi thiếu công cụ hỗ trợ, bác sĩ phụ thuộc trí nhớ hoặc bỏ qua bước kiểm tra → nguy cơ medication error, bỏ sót cảnh báo sớm, delay điều trị — với bệnh nhân nguy kịch, hậu quả có thể không đảo ngược.

### 2.2 Pain points → Nhu cầu

| # | Pain point | Nhu cầu phái sinh |
|---|-----------|-------------------|
| P1 | Tra cứu drug interaction/guideline/liều khi tay đang bận | Giao diện **hands-free** (voice) trả về thông tin có nguồn |
| P2 | Cognitive overload từ hàng trăm data points/ca trực | Tự động **tổng hợp** patient context + tính điểm cảnh báo |
| P3 | 10–20 thuốc đồng thời → hàng chục cặp tương tác | **Tự động** check allergy + drug interaction + chỉnh liều theo thận |

### 2.3 Use case mục tiêu
Tình huống **bán khẩn cấp (semi-urgent)**: bệnh nhân đang ổn định nhưng có dấu hiệu xấu đi, bác sĩ cần quyết định trong vài phút — đủ window để query bằng giọng nói.

---

## 3. Stakeholders & Users

| Bên liên quan | Vai trò | Quan tâm chính |
|---------------|---------|----------------|
| Bác sĩ ICU | Primary user | Latency thấp, câu trả lời ngắn gọn có citation, độ chính xác cao |
| Điều dưỡng ICU | Secondary user | Tra cứu nhanh dị ứng + danh sách thuốc trước khi cho dùng |
| Admin | Quản trị | Audit log, quản lý version knowledge base |
| Intern/Training | Người học | Demo mode với synthetic data |
| Mentor / clinician validator | Đánh giá | Tính an toàn lâm sàng, validate output |

---

## 4. Phân tích yêu cầu người dùng (User Requirements)

Trích từ User Stories (PRD §3.3), gắn với flow và độ ưu tiên.

| ID | Là... | Tôi muốn... | Để... | Flow | Ưu tiên |
|----|-------|-------------|-------|------|---------|
| US-01 | Bác sĩ ICU | Nói câu hỏi và nhận câu trả lời có nguồn | Không dừng tay gõ phím | Golden Path | Must |
| US-02 | Bác sĩ ICU | Tự động check drug interaction khi nhắc thuốc | Phát hiện nguy cơ trước kê đơn | Golden Path | Must |
| US-03 | Bác sĩ ICU | Xem NEWS2 hiện tại ngay lập tức | Đánh giá nguy hiểm nhanh | NEWS2 Flow | Must |
| US-04 | Bác sĩ ICU | Cảnh báo rõ ràng khi có dị ứng | Tránh sai sót nghiêm trọng | Allergy Alert | Must |
| US-05 | Bác sĩ ICU | Hệ thống nói "không đủ thông tin" thay vì đoán | Tin tưởng độ chính xác | Fallback | Must |
| US-06 | Điều dưỡng | Xem nhanh danh sách dị ứng bằng giọng nói | Xác nhận trước khi cho thuốc | Secondary User | Should |

---

## 5. Yêu cầu chức năng (Functional Requirements)

### 5.1 ASR Module

| ID | Yêu cầu | Mức độ |
|----|---------|--------|
| F-ASR-01 | Push-to-talk: nhấn nút → thu âm bắt đầu < 0.5s | Must |
| F-ASR-02 | Hỗ trợ tiếng Việt lẫn tiếng Anh medical term trong cùng câu (code-switching) | Must |
| F-ASR-03 | Nhận dạng đúng tên thuốc ICU phổ biến (Vancomycin, Norepinephrine, Midazolam…) | Must |
| F-ASR-04 | Hiển thị transcript để bác sĩ xác nhận trước khi query | Must |
| F-ASR-05 | Cho phép chỉnh sửa transcript trước khi gửi | Should |
| F-ASR-06 | Hiển thị confidence score của transcript | Nice to have |

### 5.2 RAG Module

| ID | Yêu cầu | Mức độ |
|----|---------|--------|
| F-RAG-01 | Query FHIR R4: Patient, Observation, MedicationRequest, AllergyIntolerance, Condition | Must |
| F-RAG-02 | Retrieve từ knowledge base: clinical guidelines + drug interaction database | Must |
| F-RAG-03 | Mọi response có source citation (tên tài liệu + năm + section) | Must |
| F-RAG-04 | Allergy check tự động — **check & hiển thị đầu tiên**, trước recommendation | Must |
| F-RAG-05 | Drug interaction check với danh sách thuốc đang dùng (MedicationRequest) | Must |
| F-RAG-06 | Tự tính NEWS2 từ Observation (SpO₂, RR, mạch, HA, nhiệt độ, ý thức) | Must |
| F-RAG-07 | Tự tính qSOFA từ Observation (Glasgow, RR, SBP) | Must |
| F-RAG-08 | Điều chỉnh liều theo eGFR (Creatinine + tuổi + cân nặng + giới tính) | Must |
| F-RAG-09 | Fallback: "Không đủ thông tin để khuyến nghị" khi confidence < threshold | Must |
| F-RAG-10 | Response ngắn gọn, đọc được trong < 10s | Should |

### 5.3 Yêu cầu an toàn (Safety — bắt buộc, không thương lượng)

| ID | Yêu cầu |
|----|---------|
| F-SAFE-01 | Mọi response PHẢI có source citation — không ngoại lệ |
| F-SAFE-02 | Allergy conflict PHẢI được check và hiển thị đầu tiên, trước recommendation |
| F-SAFE-03 | Hallucination rate = 0% — từ chối trả lời nếu không có source |
| F-SAFE-04 | Disclaimer bắt buộc: "Cần bác sĩ xác nhận trước khi thực hiện" |
| F-SAFE-05 | PHI không được lưu sau khi session kết thúc |

### 5.4 Thứ tự pipeline xử lý (thống nhất giữa các tài liệu)

```
Intent extraction → Parallel query (FHIR + Vector DB + Lookup + Drug API)
   → Allergy conflict check (LUÔN ĐẦU TIÊN)
   → NEWS2/qSOFA/eGFR calculation (nếu cần)
   → Drug interaction check
   → LLM generation (grounded, allergy hiển thị đầu tiên + citation)
```

---

## 6. Yêu cầu phi chức năng (Non-Functional Requirements)

| ID | Metric | Target | Ghi chú |
|----|--------|--------|---------|
| NFR-01 | ASR Word Error Rate | < 10% (chấp nhận < 20%) | Xác định sau test |
| NFR-02 | Medical term error rate | < 5% (chấp nhận < 10%) | Quan trọng nhất — failure mode nguy hiểm |
| NFR-03 | End-to-end latency | < 5s (chấp nhận < 8s) | Từ khi ngừng nói đến khi hiển thị |
| NFR-04 | Source citation rate | 100% | Không chấp nhận response không nguồn |
| NFR-05 | Hallucination rate | 0% | Hard requirement |
| NFR-06 | Allergy detection rate | 100% | Không chấp nhận false negative |
| NFR-07 | Concurrency | 3 query đồng thời, response không tăng > 50% | ICU 20 giường |

> **Nguyên tắc ưu tiên:** Accuracy > Latency. Response chậm 2 giây nhưng đúng tốt hơn response nhanh nhưng sai liều thuốc.

---

## 7. Yêu cầu dữ liệu (Data Requirements)

Dữ liệu chia 4 lớp, xử lý khác nhau:

| Lớp | Nguồn | Nội dung | Vai trò |
|-----|-------|----------|---------|
| **FHIR R4** (dynamic) | SMART Health IT Sandbox (synthetic) | Patient, Encounter, Observation, MedicationRequest, MedicationAdministration, AllergyIntolerance, Condition, DiagnosticReport, Procedure | Patient context real-time |
| **SQLite lookup** (static) | LOINC + ICD-10 VN (QĐ BYT) | `loinc_icu` (28 codes), `icd10` (10.663 codes bilingual) | Exact lookup — dịch mã → tên |
| **Vector DB** | Quy trình hồi sức cấp cứu (QĐ 1904/QĐ-BYT, 2014, 892 trang) | 232 chunks (mỗi chunk = 1 quy trình) | Semantic search guideline |
| **Drug Interaction API** | DrugBank (chính) / OpenFDA / RxNav (dự phòng) | 15.000+ drugs, interaction + contraindication | Drug safety check |

**Ràng buộc dữ liệu quan trọng:**
- Observation lấy giá trị mới nhất bằng `_sort=-date&_count=1` để tránh data của lần nhập viện cũ.
- LOINC/ICD-10 **không** chunk vào Vector DB — chỉ guideline văn bản mới embed.
- Chunk theo quy trình (không cắt cứng theo ký tự) để giữ context "Chống chỉ định".

---

## 8. Ràng buộc & Giả định

### 8.1 Ràng buộc (Constraints)
- Thời gian: PoC 6 tuần.
- Không có data bệnh nhân thật → dùng synthetic (SMART sandbox).
- Push-to-talk là bắt buộc (workaround cho noise ICU 60–80 dB, không phải giải pháp lâu dài).
- ASR model lấy từ HuggingFace; LLM backbone GPT-4o; Vector DB ChromaDB; embedding cần test (BGE-M3 / Qwen3 / text-embedding-3-small / Gemini).

### 8.2 Giả định (Assumptions)
- Bệnh nhân đã được chọn trước khi query (Encounter active).
- FHIR sandbox cung cấp đủ Observation/Medication/Allergy cho scenario test.
- Có ít nhất 1 buổi clinical review với clinician trong Tuần 4.

### 8.3 Phụ thuộc (Dependencies)
- DrugBank academic access; SMART Health IT sandbox khả dụng; GPT-4o API.

---

## 9. Tiêu chí nghiệm thu (Acceptance Criteria)

Hệ thống được coi là **thành công** nếu đạt (theo Evaluation Plan §7):

| Metric | Minimum | Target |
|--------|---------|--------|
| ASR WER (overall) | < 20% | < 10% |
| Medical term error rate | < 10% | < 5% |
| RAG scenario pass rate (20 scenarios) | ≥ 70% (14/20) | ≥ 85% (17/20) |
| Critical safety scenarios (S-01…S-05) | **100%** | 100% |
| Source citation rate | 100% | 100% |
| Hallucination rate | 0% | 0% |
| End-to-end latency | < 8s | < 5s |

> Critical safety scenarios là **hard requirement** — không đạt 100% thì không được demo.

**Cách verify:** ASR (Tier 1, 30 câu test) · RAG functional (Tier 2, 20 scenarios, rubric 5 tiêu chí ≥7/10) · FHIR integration (Tier 3, T-01…T-07) · Performance (Tier 4, latency breakdown + load) · Human eval (Tier 5, usability + questionnaire).

---

## 10. Rủi ro & Mitigation

| Rủi ro | Khả năng | Mức độ | Mitigation | Liên hệ |
|--------|----------|--------|------------|---------|
| ASR sai tên thuốc → recommendation sai | Cao | Nghiêm trọng | Push-to-talk + transcript confirmation | F-ASR-04 |
| RAG hallucinate | Trung bình | Nghiêm trọng | Confidence threshold + citation bắt buộc | F-RAG-09, F-SAFE-03 |
| FHIR thiếu data → NEWS2 sai | Cao | Cao | Hiển thị rõ data point bị thiếu, không suy đoán | Error Flow §8.3 |
| Knowledge base outdated | Thấp | Trung bình | Version tracking + ngày cập nhật | — |
| Bác sĩ không trust hệ thống | Cao | Cao | Clinical workflow testing Tuần 4 | Tier 5 |
| Không có clinician validate | Cao | Cao | Ghi rõ known limitation + cross-check guideline gốc | Eval §3.3 |
| Latency > 5s | Trung bình | Trung bình | Performance optimization Tuần 5 | NFR-03 |

---

## 11. Ma trận truy vết (Traceability Matrix)

| Pain point | User Story | Functional Req | NFR | Flow | Test |
|-----------|-----------|----------------|-----|------|------|
| P1 | US-01 | F-ASR-01..06, F-RAG-01..03 | NFR-01,02,03,04 | Golden Path | Tier 1, S-02 |
| P3 | US-02 | F-RAG-05, F-RAG-08 | NFR-05 | Golden Path | S-04 |
| P2 | US-03 | F-RAG-06, F-RAG-07 | — | NEWS2/qSOFA Flow | S-03, S-05, T-06 |
| P3 | US-04 | F-RAG-04, F-SAFE-02 | NFR-06 | Allergy Alert | S-01, T-03 |
| — | US-05 | F-RAG-09, F-SAFE-03 | NFR-05 | Fallback | S-06, S-09 |
| P3 | US-06 | F-RAG-01 (read-only) | — | Secondary User | T-03, T-04 |

---

## 12. Mức ưu tiên (MoSCoW) — tổng hợp

- **Must:** ASR push-to-talk + confirmation, FHIR query, allergy check (đầu tiên), drug interaction, NEWS2/qSOFA/eGFR, citation 100%, fallback, hallucination 0%.
- **Should:** Chỉnh sửa transcript, response < 10s đọc nhanh, secondary user flow.
- **Could:** Confidence score hiển thị, multi-scenario follow-up nâng cao.
- **Won't (PoC):** Monitor integration real-time, diarization, trend analysis, noise cancellation, production deploy.

---

*Tài liệu liên quan: `01_PRD_ASR_RAG_CLINICAL_ASSISTANT.md`, `02_USER_FLOW.md`, `03_EVALUATION_PLAN.md`, `04_RESEARCH_BACKGROUND_AND_GAPS.md`*
