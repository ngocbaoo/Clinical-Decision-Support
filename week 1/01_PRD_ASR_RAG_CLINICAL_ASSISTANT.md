# 01 · Product Requirements Document (PRD)
# ASR + RAG Clinical Assistant — ICU Decision Support
**Phiên bản:** 1.0 · **Trạng thái:** Draft — chờ mentor review
**Ngày lập:** 01/06/2026 · **Người làm:** Tạ Bảo Ngọc

---

## 1. Tổng quan sản phẩm

### 1.1 Một câu mô tả
Hệ thống cho phép bác sĩ ICU **nói câu hỏi lâm sàng bằng giọng nói** và nhận lại **khuyến nghị điều trị có trích dẫn nguồn** trong vài giây — trong khi tay vẫn đang thực hiện thủ thuật.

### 1.2 Bối cảnh
Trong môi trường ICU, bác sĩ phải đưa ra quyết định lâm sàng nhanh — lựa chọn thuốc, điều chỉnh liều, kiểm tra drug interaction — trong khi tay đang bận. Đồng thời, các thiết bị y tế liên tục sinh ra lượng lớn dữ liệu (vital signs, lab results, ventilator logs) mà không có công cụ tổng hợp real-time.

Khi không có công cụ hỗ trợ, bác sĩ phụ thuộc vào trí nhớ hoặc bỏ qua bước kiểm tra — dẫn đến nguy cơ medication error, bỏ sót cảnh báo sớm, và delay trong quyết định điều trị. Với bệnh nhân nguy kịch, các sai sót này có thể không thể đảo ngược.

### 1.3 Phạm vi của PoC (6 tuần)
| In scope | Out of scope |
|-------------|-------------|
| ASR pipeline (push-to-talk, Whisper/MedASR) | Real-time monitor integration |
| RAG pipeline với medical knowledge base | Multi-speaker diarization |
| FHIR R4 query với synthetic patient data | Noise cancellation phức tạp |
| Medication safety check + allergy alert | Trend analysis theo thời gian |
| NEWS2 tự tính từ Observation data | HIS/EMR integration thật |
| End-to-end demo với clinical scenarios | Production deployment |

---

## 2. Vấn đề cần giải quyết

### 2.1 Pain Points

**Pain point 1 — Information retrieval trong thời gian thực**
Bác sĩ cần tra cứu drug interaction, guideline điều trị, dosage adjustment — nhưng tay đang thực hiện thủ thuật. Giải pháp hiện tại là nhờ đồng nghiệp, dựa vào trí nhớ, hoặc bỏ qua.

**Pain point 2 — Information synthesis từ data của máy móc**
Một bệnh nhân ICU sinh ra hàng trăm data points mỗi ca trực. Không có tool nào tổng hợp tự động → bác sĩ bị cognitive overload → bỏ sót pattern quan trọng (ví dụ: trend SpO₂ giảm dần kết hợp lactate tăng = dấu hiệu septic shock sớm).

**Pain point 3 — Medication safety với nhiều thuốc đồng thời**
Bệnh nhân ICU dùng trung bình 10–20 thuốc cùng lúc → có tới hàng chục cặp tương tác cần kiểm tra. Không ai nhớ hết trong điều kiện kiệt sức sau ca trực dài.

### 2.2 Target Use Case
Hệ thống được thiết kế cho **tình huống bán khẩn cấp (semi-urgent)** — bệnh nhân đang ổn định nhưng có dấu hiệu xấu đi, bác sĩ cần quyết định trong vài phút. Đây là window bác sĩ vừa đủ thời gian để query.

---

## 3. Target users

### 3.1 Primary User — Bác sĩ ICU
- **Bối cảnh:** Đứng tại bedside, tay đang làm thủ thuật hoặc kiểm tra bệnh nhân
- **Câu hỏi thường gặp:**
  - "Bệnh nhân đang dùng Vancomycin, thêm Gentamicin được không?"
  - "SpO₂ 88%, NEWS2 bao nhiêu điểm, cần làm gì tiếp theo?"
  - "Bệnh nhân suy thận, Creatinine 180 — điều chỉnh liều Vancomycin thế nào?"
- **Yêu cầu:** Latency < 5 giây, câu trả lời ngắn gọn, có source citation

### 3.2 Secondary User — Điều dưỡng ICU
- **Bối cảnh:** Chuẩn bị thuốc, cần xác nhận nhanh trước khi cho bệnh nhân dùng
- **Câu hỏi thường gặp:** "Bệnh nhân này dị ứng gì?", "Liều Norepinephrine hiện tại là bao nhiêu?"

### 3.3 User Stories

| ID | Vai trò | Tôi muốn... | Để... |
|----|---------|-------------|-------|
| US-01 | Bác sĩ ICU | Nói câu hỏi bằng giọng nói và nhận câu trả lời có nguồn | Không cần dừng tay để gõ bàn phím |
| US-02 | Bác sĩ ICU | Hệ thống tự động check drug interaction khi tôi đề cập thuốc | Phát hiện nguy cơ trước khi kê đơn |
| US-03 | Bác sĩ ICU | Xem NEWS2 score hiện tại của bệnh nhân ngay lập tức | Đánh giá mức độ nguy hiểm nhanh |
| US-04 | Bác sĩ ICU | Hệ thống cảnh báo rõ ràng khi phát hiện dị ứng | Tránh sai sót nghiêm trọng |
| US-05 | Bác sĩ ICU | Hệ thống nói "không đủ thông tin" thay vì đưa ra câu trả lời không có cơ sở | Tin tưởng vào độ chính xác của hệ thống |
| US-06 | Điều dưỡng | Xem nhanh danh sách dị ứng của bệnh nhân bằng giọng nói | Xác nhận trước khi cho thuốc |

---

## 4. Functional Requirements

### 4.1 ASR Module

| ID | Yêu cầu | Mức độ |
|----|---------|--------|
| F-ASR-01 | Push-to-talk: nhấn nút → thu âm bắt đầu trong < 0.5 giây | Must have |
| F-ASR-02 | Hỗ trợ tiếng Việt lẫn tiếng Anh medical terminology trong cùng câu | Must have |
| F-ASR-03 | Nhận dạng đúng tên thuốc ICU phổ biến (Vancomycin, Norepinephrine, Midazolam...) | Must have |
| F-ASR-04 | Hiển thị transcript để bác sĩ xác nhận trước khi query | Must have |
| F-ASR-05 | Cho phép chỉnh sửa transcript trước khi gửi | Should have |
| F-ASR-06 | Confidence score hiển thị để bác sĩ biết mức độ chắc chắn của transcript | Nice to have |

### 4.2 RAG Module

| ID | Yêu cầu | Mức độ |
|----|---------|--------|
| F-RAG-01 | Query FHIR R4: Patient, Observation, MedicationRequest, AllergyIntolerance, Condition | Must have |
| F-RAG-02 | Retrieve từ knowledge base: clinical guidelines + drug interaction database | Must have |
| F-RAG-03 | Mọi response phải có source citation (tên tài liệu + năm + section) | Must have |
| F-RAG-04 | Allergy check tự động — cảnh báo trước phần recommendation | Must have |
| F-RAG-05 | Drug interaction check với danh sách thuốc đang dùng từ MedicationRequest | Must have |
| F-RAG-06 | Tự tính NEWS2 score từ Observation data (SpO₂, nhịp thở, mạch, huyết áp, nhiệt độ, ý thức) | Must have |
| F-RAG-07 | Tự tính qSOFA từ Observation data (Glasgow, nhịp thở, huyết áp tâm thu) | Must have |
| F-RAG-08 | Điều chỉnh liều thuốc theo eGFR (tính từ Creatinine + tuổi + cân nặng + giới tính) | Must have |
| F-RAG-09 | Fallback: hiển thị "Không đủ thông tin để khuyến nghị" khi confidence < threshold | Must have |
| F-RAG-10 | Response ngắn gọn, có thể đọc trong < 10 giây | Should have |

### 4.3 Safety Requirements

```
1. Mọi response PHẢI có source citation — không có ngoại lệ
2. Allergy conflict PHẢI hiển thị đầu tiên, trước recommendation
3. Hallucination rate = 0% — từ chối trả lời nếu không có source
4. Disclaimer bắt buộc: "Cần bác sĩ xác nhận trước khi thực hiện"
5. PHI (thông tin bệnh nhân) không được lưu sau khi session kết thúc
```

---

## 5. Non-functional Requirements

| Metric | Target | Ghi chú |
|--------|--------|---------|
| ASR Word Error Rate | < [ X ]% | Xác định sau khi test thực tế |
| Medical term accuracy | < [ X ]% error | Đặc biệt quan trọng với tên thuốc, liều lượng |
| End-to-end latency | < 5 giây | Từ khi ngừng nói đến khi hiển thị response |
| RAG source citation rate | 100% | Không chấp nhận response không có source |
| Hallucination rate | 0% | Không chấp nhận hallucination |
| Allergy detection rate | 100% | Không chấp nhận false negative |

> **Lưu ý:** Accuracy > Latency. Một response chậm 2 giây nhưng đúng tốt hơn response nhanh nhưng sai liều thuốc.

---

## 6. Chiến lược dữ liệu

### 6.1 Patient Data (FHIR R4)
- **PoC:** SMART Health IT Sandbox (synthetic data) — `launch.smarthealthit.org`
- **Production (tương lai):** FHIR server của bệnh viện đối tác
- **FHIR Resources cần dùng:**

```
Patient                     → demographics (tuổi, cân nặng, giới tính cho eGFR)
Encounter                   → lần nhập viện hiện tại
Observation                 → vital signs (LOINC codes), lab results, NEWS2 components
Condition                   → chẩn đoán ICD-10
MedicationRequest           → danh sách thuốc đang dùng (drug interaction check)
MedicationAdministration    → thuốc đã dùng thực tế
AllergyIntolerance          → dị ứng (safety check — query đầu tiên)
DiagnosticReport            → kết quả xét nghiệm tổng hợp (sepsis workup)
Procedure                   → thủ thuật đã làm
```

> **Lưu ý:** `Encounter` không query trực tiếp — thay vào đó filter theo `_sort=-date&_count=1` để lấy giá trị mới nhất của từng Observation, tránh lấy data của lần nhập viện cũ.

---

### 6.2 Lookup Tables (SQLite)

Hai bảng structured data được lưu trong `clinical_db.sqlite` — dùng cho exact lookup, **không** chunk vào Vector DB.

**Bảng `loinc_icu` — 28 codes ICU-relevant**

Nguồn: LOINC full database (loinc.org), được filter lấy subset ICU.

| Category | Số codes | Ví dụ |
|----------|----------|-------|
| vital_sign | 7 | SpO₂ `59408-5`, HR `8867-4`, RR `9279-1` |
| lab_renal | 3 | Creatinine `2160-0` |
| lab_electrolyte | 3 | K `2823-3`, Na `2951-2`, HCO₃ `1963-8` |
| lab_hematology | 3 | WBC `26464-8`, Platelet `777-3`, Hgb `718-7` |
| lab_abg | 3 | PaO₂ `2703-7`, PaCO₂ `2019-8`, pH `2744-1` |
| lab_infection | 3 | CRP `1988-5`, PCT `33959-8`, Ferritin `2276-4` |
| lab_coagulation | 2 | PT `5902-2`, INR `6301-6` |
| lab_metabolic | 1 | Lactate `32693-4` |
| lab_liver | 1 | Bilirubin `1975-2` |

**Bảng `icd10` — 10,663 codes bilingual**

Nguồn: ICD-10 Việt Nam — Quyết định Bộ Y tế, parse từ PDF 860 trang.

| Thống kê | Số lượng |
|----------|----------|
| Tổng codes | 10,663 |
| Có tên tiếng Việt | 4,206 |
| Có tên tiếng Anh | 8,137 |
| Có cả hai | 1,680 |

**Cách dùng trong RAG pipeline:**

```python
# Translate LOINC code → tên để đưa vào LLM context
def get_loinc_name(code: str) -> str:
    row = conn.execute(
        "SELECT LONG_COMMON_NAME, EXAMPLE_UNITS FROM loinc_icu WHERE LOINC_NUM = ?",
        (code,)
    ).fetchone()
    return f"{row[0]} ({row[1]})" if row else code

# Translate ICD-10 → tên tiếng Việt ưu tiên
def get_icd10_name(code: str) -> str:
    row = conn.execute(
        "SELECT name_vi, name_en FROM icd10 WHERE code = ?", (code,)
    ).fetchone()
    return (row[0] or row[1]) if row else code
```

---

### 6.3 Knowledge Base (Vector DB)

Unstructured documents được chunk, embed và index vào Vector DB — dùng cho semantic search.

| Tài liệu | Nguồn | Năm | Ngôn ngữ | Loại | Trạng thái |
|----------|-------|-----|---------|------|------------|
| Quy trình kỹ thuật hồi sức cấp cứu (232 quy trình) | Bộ Y tế VN — QĐ 1904/QĐ-BYT | 2014 | Tiếng Việt | Unstructured PDF 892 trang | ✅ Có file |


**Chunking strategy cho `quy_trinh_icu_vn.pdf`:**

```
892 trang → Extract text → Parse boundary theo từng quy trình
                                    ↓
232 chunks — mỗi chunk = 1 quy trình (3–5 trang)
Mỗi chunk giữ nguyên cấu trúc:
  - Khái niệm
  - Chỉ định / Chống chỉ định  ← quan trọng cho contraindication check
  - Các bước tiến hành
  - Theo dõi và xử lý tai biến
                                    ↓
Embed với multilingual model → Index vào Vector DB
```

> **Lý do chunk theo quy trình thay vì chunk cứng theo số ký tự:** Mỗi quy trình là một đơn vị lâm sàng độc lập. Chunk cắt giữa quy trình sẽ mất context của "Chống chỉ định" hoặc "Các bước tiến hành" — dẫn đến RAG retrieve được chunk nhưng thiếu thông tin quan trọng.

---

### 6.4 Drug Interaction Database

Drug interaction **không** được lưu trong Vector DB hay SQLite tự build — thay vào đó dùng API từ nguồn bên ngoài:

| Nguồn | Loại | Coverage | Chi phí | Quyết định |
|-------|------|----------|---------|------------|
| DrugBank | API + download | 15,000+ drugs, interaction + contraindication | Free cho academic | Đề xuất dùng |
| OpenFDA Drug Label | REST API | FDA-approved drugs | Miễn phí | Dự phòng |
| RxNav (NIH) | REST API | RxNorm + interaction | Miễn phí | Dự phòng |


---

### 6.5 Tóm tắt kiến trúc dữ liệu

```
Query của bác sĩ
        ↓
┌─────────────────────────────────────────────────────┐
│                  Data Layer                         │
│                                                     │
│  FHIR R4 Sandbox    SQLite              Vector DB   │
│  ├── Patient        ├── loinc_icu       ├── quy_    │
│  ├── Observation    │   (28 codes)         trinh_   │
│  ├── Medication     └── icd10              icu_vn   │
│  ├── Allergy            (10,663 codes)              │
│  ├── Condition                                      │
│  ├── Diagnostic                                     │
│  └── Procedure                                      │
│                                                     │
│  Drug Interaction API (DrugBank / OpenFDA)          │
└─────────────────────────────────────────────────────┘
        ↓
RAG Pipeline → LLM → Response + Citation
```
---

## 7. Chiến lược AI

### 7.1 ASR Stack
- **Candidate models (chưa test):**

| Model | Tiếng Việt | Medical terms | Code-switching | Weights |
|-------|-----------|---------------|----------------|---------|
| Whisper large-v3 | Yes |  OOV | Chưa test | HuggingFace |
| MedASR (Google) | No |  Chưa test | Chưa test | HuggingFace |
| MultiMed-ST whisper-vi | Yes |  Yes | Chưa test | HuggingFace |
| MultiMed-ST whisper-multilingual | Yes | Yes | Chưa test | HuggingFace |

- **Decision:** [ Điền sau khi có kết quả test thực tế ]
- **Fallback strategy:** Fine-tune PhoWhisper-small trên dataset VietMed

### 7.2 RAG Stack
```
Input (text từ ASR)
    ↓
Intent extraction (LLM)
    ↓
Parallel query:
    ├── FHIR R4 → patient context
    └── Vector DB → relevant guidelines
    ↓
Allergy conflict check  ← LUÔN ĐẦU TIÊN (Safety Req #2)
    ↓
NEWS2 / qSOFA calculation (nếu cần)
    ↓
eGFR calculation (nếu có Creatinine)
    ↓
Drug interaction check
    ↓
LLM generation với grounded context (allergy hiển thị đầu tiên)
    ↓
Response với source citation
```

> **Allergy luôn được check và hiển thị đầu tiên**

- **Vector DB:** ChromaDB
- **Embedding model:** BGE-M3 / Qwen3-Embedding-8B / text-embedding-3-small / Google Gemini Embedding  (cần test)
- **LLM backbone:** GPT-4o

### 7.3 Hallucination Mitigation
1. **Source grounding bắt buộc:** Mọi claim phải trace về chunk cụ thể trong knowledge base
2. **Confidence threshold:** Nếu retrieved chunks có similarity < threshold → từ chối trả lời
3. **Structured output:** LLM được prompt để output theo format cố định, không tự thêm thông tin
4. **Clinical validation:** Output được so sánh với ground truth trong Evaluation Plan

---

## 8. Rủi ro và Mitigation

| Rủi ro | Khả năng xảy ra | Mức độ | Mitigation |
|--------|----------------|--------|------------|
| ASR sai tên thuốc → recommendation sai | Cao | Nghiêm trọng | Sử dụng Push-to-talk + transcript confirmation  |
| RAG hallucinate → thông tin sai | Trung bình | Nghiêm trọng | Confidence threshold + source citation bắt buộc |
| FHIR data không đủ → tính NEWS2 sai | Cao | Cao | Hiển thị rõ data points nào bị thiếu |
| Knowledge base outdated | Thấp | Trung bình | Version tracking + ngày cập nhật hiển thị |
| Bác sĩ không trust hệ thống | Cao | Cao | Tuần 4 clinical workflow testing với clinician |
| Latency > 5 giây | Trung bình | Trung bình | Tuần 5 performance optimization |

---

