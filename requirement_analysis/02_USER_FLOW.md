# 02 · User Flow
# ASR + RAG Clinical Assistant — ICU Decision Support

---

## 1. Tổng quan luồng hệ thống

Bác sĩ ICU **nói câu hỏi lâm sàng** và nhận lại **khuyến nghị có trích dẫn nguồn** trong < 5 giây (NFR — End-to-end latency), trong khi tay vẫn đang thực hiện thủ thuật.

```
[Bác sĩ] ──push-to-talk──▶ [ASR] ──transcript──▶ [Confirm] ──text──▶ [RAG Pipeline]
                                                                          │
                                                                          ▼
                                                                 [Intent extraction]
                                                                          │
                                          ┌───────────────────────────────┼───────────────────────────────┐
                                          ▼                               ▼                               ▼
                                    [FHIR R4 Query]                  [Vector DB]                  [Lookup / API]
                                    Patient                          quy_trinh_icu_vn             SQLite loinc_icu
                                    Observation                      (clinical guidelines)        SQLite icd10
                                    MedicationRequest                                             DrugBank API
                                    AllergyIntolerance
                                    Condition
                                          │
                                          ▼
                          [Allergy conflict check ← LUÔN ĐẦU TIÊN]
                                          │
                                          ▼
                          [Calculation: NEWS2 / qSOFA / eGFR]
                                          │
                                          ▼
                          [Drug interaction check]
                                          │
                                          ▼
                          [LLM generation — grounded + citation]
                                          │
                                          ▼
                                     [Bác sĩ]
```

> Thứ tự pipeline bám theo PRD Mục 7.2. **Allergy luôn được check và hiển thị đầu tiên** (Safety Req #2) — ngay sau khi có patient context, trước mọi calculation và recommendation khác.

---

## 2. Golden Path — Luồng chính (Happy Path)

Luồng bác sĩ ICU sử dụng hệ thống trong điều kiện lý tưởng. Hiện thực hoá **US-01, US-02**.

### Bước 0 · Tiền điều kiện
- Bệnh nhân đã được chọn (Encounter đang active) — hệ thống đã có patient context.
- `Encounter` không query trực tiếp; mỗi Observation lấy giá trị mới nhất bằng `_sort=-date&_count=1` để tránh lấy data của lần nhập viện cũ.

### Bước 1 · Kích hoạt (F-ASR-01)

```
Bác sĩ nhấn nút push-to-talk (thiết bị đeo tay hoặc bàn phím)
    ↓  (thu âm bắt đầu trong < 0.5 giây)
Đèn chỉ thị đỏ sáng — "Đang ghi âm"
    ↓
Bác sĩ nói câu hỏi (tiếng Việt hoặc hỗn hợp Việt-Anh — F-ASR-02)
    ↓
Bác sĩ thả nút → Thu âm dừng
```

### Bước 2 · Transcript Confirmation (F-ASR-04, F-ASR-05, F-ASR-06)

```
Hệ thống hiển thị transcript trong < 2 giây:

┌─────────────────────────────────────────────────────┐
│    Transcript:                          [conf 0.94] │
│  "Bệnh nhân đang dùng Vancomycin 1g, Creatinine     │
│   180, muốn thêm Gentamicin — có tương tác không?"  │
│                                                     │
│  [Xác nhận]   [Chỉnh sửa]   [Huỷ]                   │
└─────────────────────────────────────────────────────┘
```

**Lý do bắt buộc có bước này:** ASR có thể sai tên thuốc → recommendation sai (Rủi ro #1, mức Nghiêm trọng). Confirmation là safety layer. Confidence score giúp bác sĩ biết khi nào cần soát kỹ.

### Bước 3 · Processing (F-RAG-01, F-RAG-02)

```
Bác sĩ nhấn [Xác nhận]
    ↓
Hiển thị "Đang tra cứu..." (loading indicator)
    ↓
Intent extraction (LLM) — xác định: hỏi tương tác thuốc + cần điều chỉnh liều theo thận
    ↓
Parallel execution:
    ├── FHIR query : Patient + Observation + MedicationRequest + AllergyIntolerance + Condition
    ├── Vector search : quy_trinh_icu_vn (guidelines liên quan)
    ├── Lookup : loinc_icu (dịch LOINC→tên), icd10 (dịch mã→tên VN)
    └── DrugBank API : Vancomycin × Gentamicin interaction
    ↓
Allergy conflict check (F-RAG-04) ← LUÔN ĐẦU TIÊN
    ↓
Calculation : eGFR từ Creatinine + tuổi + cân nặng + giới tính (F-RAG-08)
    ↓
Drug interaction check (F-RAG-05)
```

### Bước 4 · Response Display (F-RAG-03, Safety Req #1–#4)

```
┌─────────────────────────────────────────────────────┐
│     CẢNH BÁO TƯƠNG TÁC THUỐC                        │
│  Vancomycin + Gentamicin = tăng nguy cơ             │
│  nephrotoxicity (tổn thương thận)                   │
│  → Nguồn: DrugBank, Drug Interactions (2024)        │
│                                                     │
│    TÌNH TRẠNG THẬN HIỆN TẠI                         │
│  Creatinine: 180 μmol/L (đo 08:00, LOINC 2160-0)    │
│  eGFR ước tính: ~35 mL/min → Suy thận nhẹ-vừa       │
│                                                     │
│    KHUYẾN NGHỊ                                      │
│  Thận trọng cao khi kết hợp hai thuốc này.          │
│  Nếu bắt buộc: theo dõi creatinine mỗi 12 giờ,      │
│  cân nhắc therapeutic drug monitoring.              │
│  → Nguồn: Quy trình hồi sức cấp cứu (QĐ 1904/       │
│           QĐ-BYT, 2014), Quy trình #__              │
│                                                     │
│  ─────────────────────────────────────────────────  │
│    Khuyến nghị cần bác sĩ xác nhận trước khi        │
│      thực hiện. Hệ thống chỉ hỗ trợ quyết định.     │
└─────────────────────────────────────────────────────┘
```

> Response thiết kế để đọc trong < 10 giây (F-RAG-10). Mọi block đều có source citation — không ngoại lệ (Safety Req #1).

### Bước 5 · Follow-up (tuỳ chọn)

```
Bác sĩ hỏi tiếp trong cùng context:
"Vậy nếu vẫn cần dùng thì liều Vancomycin điều chỉnh thế nào?"
    ↓
Hệ thống giữ nguyên FHIR + patient context, chỉ query thêm về dosing
    ↓
Trả lời với liều đã điều chỉnh theo eGFR đã tính ở trên
```

### Bước 6 · Kết thúc session (Safety Req #5)

```
Bác sĩ đóng session / chọn bệnh nhân khác
    ↓
PHI (transcript + patient context) bị xoá — KHÔNG lưu sau khi session kết thúc
```

---

## 3. Allergy Alert Flow (F-RAG-04, US-04, Safety Req #2)

Ưu tiên tuyệt đối — hiển thị **trước** mọi recommendation khác.

```
Query nhận được
    ↓
FHIR query AllergyIntolerance
    ↓
    ├── Không có dị ứng liên quan → tiếp tục bình thường
    │
    └── Phát hiện dị ứng liên quan đến câu hỏi
            ↓
    ┌───────────────────────────────────────┐
    │    CẢNH BÁO DỊ ỨNG                 │
    │  Bệnh nhân DỊ ỨNG: Penicillin        │
    │  Mức độ: Cao (anaphylaxis)            │
    │  KHÔNG kê đơn nhóm beta-lactam       │
    │  → Nguồn: FHIR AllergyIntolerance    │
    │           (ghi nhận: 12/03/2024)     │
    └───────────────────────────────────────┘
            ↓
    Tiếp tục recommendation với ràng buộc tránh nhóm thuốc trên
```

> **Allergy detection rate target = 100%** (NFR) — không chấp nhận false negative.

---

## 4. NEWS2 Calculation Flow (F-RAG-06, US-03)

Kích hoạt khi bác sĩ hỏi về tình trạng tổng thể / mức độ nguy hiểm.

```
Trigger: "Bệnh nhân hiện tại tình trạng thế nào?" / "NEWS2 bao nhiêu điểm?"
    ↓
Query FHIR Observation — giá trị mới nhất (_sort=-date&_count=1) của:
    ├── SpO₂              (LOINC 59408-5)
    ├── Nhịp thở          (LOINC 9279-1)
    ├── Huyết áp tâm thu  (LOINC 8480-6)
    ├── Mạch              (LOINC 8867-4)
    ├── Nhiệt độ          (LOINC 8310-5)
    ├── Ý thức            (Glasgow / AVPU)
    └── Đang thở O₂?      (từ MedicationRequest hoặc Procedure)
    ↓
Check: Bệnh nhân có hypercapnic respiratory failure không?
    ├── Có   → NEWS2 Scale 2 (SpO₂ target 88-92%)
    └── Không → NEWS2 Scale 1
    ↓
Tính tổng điểm NEWS2
    ↓
┌─────────────────────────────────────────────────────┐
│    NEWS2 Score: 8 điểm →  MỨC ĐỘ CAO           │
│                                                      │
│  Chi tiết:                                           │
│  SpO₂ 88%      → 3 điểm                             │
│  Nhịp thở 22   → 2 điểm                             │
│  Mạch 118      → 2 điểm                             │
│  Nhiệt độ 38.9 → 1 điểm                             │
│  Huyết áp 105  → 0 điểm                             │
│  Ý thức Alert  → 0 điểm                             │
│  Thở O₂        → 2 điểm  (nếu có)                  │
│                                                      │
│  Phản ứng cần thiết: Đánh giá khẩn cấp,             │
│  cân nhắc chuyển ICU.                               │
│  → Nguồn: NEWS2, Royal College of Physicians 2017   │
└─────────────────────────────────────────────────────┘
```

---

## 5. qSOFA Calculation Flow (F-RAG-07)

Kích hoạt khi nghi ngờ nhiễm khuẩn huyết / sepsis screening.

```
Trigger: "Bệnh nhân này có dấu hiệu sepsis không?" / "Tính qSOFA"
    ↓
Query FHIR Observation — giá trị mới nhất của:
    ├── Glasgow Coma Scale (ý thức)
    ├── Nhịp thở            (LOINC 9279-1)
    └── Huyết áp tâm thu    (LOINC 8480-6)
    ↓
Tính qSOFA (mỗi tiêu chí 1 điểm):
    ├── Nhịp thở ≥ 22/phút        → 1 điểm
    ├── Thay đổi ý thức (GCS < 15) → 1 điểm
    └── Huyết áp tâm thu ≤ 100 mmHg → 1 điểm
    ↓
┌─────────────────────────────────────────────────────┐
│    qSOFA Score: 2/3 →  Nguy cơ cao                  │
│  Nhịp thở 24 ✓ · Ý thức GCS 13 ✓ · HA 110 ✗        │
│                                                     │
│  qSOFA ≥ 2 → nghi ngờ sepsis, cân nhắc đánh giá     │
│  SOFA đầy đủ + lactate + cấy máu.                   │
│  → Nguồn: Surviving Sepsis Campaign 2021            │
└─────────────────────────────────────────────────────┘
```

---

## 6. Uncertainty / Fallback Flow (F-RAG-09, US-05, Safety Req #3)

Khi RAG không đủ thông tin để trả lời an toàn.

```
RAG pipeline chạy xong
    ↓
Kiểm tra confidence score của retrieved chunks
    ├── Confidence ≥ threshold → hiển thị response bình thường
    │
    └── Confidence < threshold HOẶC không tìm được source phù hợp
            ↓
    ┌───────────────────────────────────────────────────┐
    │   Không đủ thông tin để khuyến nghị               │
    │                                                   │
    │  Câu hỏi của bạn chưa được tìm thấy trong         │
    │  cơ sở tri thức hiện tại.                         │
    │                                                   │
    │  Gợi ý:                                           │
    │  • Tham khảo trực tiếp BNF hoặc UpToDate          │
    │  • Liên hệ dược sĩ lâm sàng                       │
    │  • [ Số hotline tư vấn thuốc của bệnh viện ]      │
    └───────────────────────────────────────────────────┘
```

**Nguyên tắc:** Hệ thống KHÔNG ĐƯỢC đưa ra recommendation khi không có source. **Hallucination rate target = 0%** — đây là safety requirement không thể thương lượng.

---

## 7. Secondary User Flow — Điều dưỡng ICU (US-06)

Điều dưỡng dùng hệ thống để xác nhận nhanh trước khi cho bệnh nhân dùng thuốc.

```
Điều dưỡng push-to-talk: "Bệnh nhân này dị ứng gì?"
    ↓
ASR → transcript → confirm
    ↓
FHIR query AllergyIntolerance + MedicationRequest (read-only, quyền cơ bản)
    ↓
┌───────────────────────────────────────┐
│    Dị ứng: Penicillin (anaphylaxis)   │
│    Thuốc đang dùng:                   │
│      Vancomycin 1g q12h               │
│      Norepinephrine 0.1 mcg/kg/min    │
│  → Nguồn: FHIR AllergyIntolerance,    │
│           MedicationRequest           │
└───────────────────────────────────────┘
```

> Điều dưỡng **không** thấy recommendation điều trị đầy đủ.

---

## 8. Error Flow

### 8.1 ASR Error — Không nhận dạng được
```
Thu âm xong → ASR không transcribe được (quá ồn, nói quá nhanh)
    ↓
"Không nhận dạng được giọng nói. Vui lòng thử lại hoặc nhập bằng bàn phím."
    ↓
Bác sĩ chọn: [Thử lại]  hoặc  [Nhập tay]
```

### 8.2 FHIR Error — Không lấy được data
```
FHIR query timeout hoặc trả về lỗi
    ↓
"Không thể lấy dữ liệu bệnh nhân hiện tại."
"Kết quả chỉ dựa trên knowledge base — không bao gồm thông tin cụ thể của bệnh nhân này."
    ↓
Hệ thống vẫn trả lời nhưng hiển thị rõ limitation
```

### 8.3 FHIR Partial Data — Thiếu một số chỉ số (Rủi ro: NEWS2 sai do thiếu data)
```
FHIR trả về nhưng thiếu một số Observation (ví dụ: không có Creatinine mới)
    ↓
Tính NEWS2/eGFR với các chỉ số có sẵn, đánh dấu rõ chỉ số nào bị thiếu:
"SpO₂: 88%  |  Nhịp thở: 22  |  Creatinine: Chưa có (cần lab mới)"
    ↓
KHÔNG suy đoán giá trị thiếu — hiển thị rõ data points nào bị thiếu (Mitigation Rủi ro #3)
```

### 8.4 Drug Interaction API Error
```
DrugBank API timeout / lỗi
    ↓
Fallback sang OpenFDA Drug Label hoặc RxNav (NIH)
    ↓
Nếu tất cả fail: "Chưa kiểm tra được tương tác thuốc tự động — vui lòng tra cứu thủ công."
```

---

## 9. Role-Based Access

| Vai trò | Quyền truy cập |
|---------|----------------|
| Bác sĩ ICU | Query đầy đủ, xem recommendation, xem source, follow-up |
| Điều dưỡng | Query cơ bản, xem allergy + medication list (read-only) |
| Admin | Xem audit log, quản lý knowledge base version |
| Intern/Training | Demo mode chỉ với synthetic patient data |

---

## 10. Sequence Diagram — Happy Path

```
Bác sĩ      ASR        RAG Pipeline      FHIR Server   Vector DB   DrugBank API
  │          │              │                 │            │            │
  │─push─────▶│              │                 │            │            │
  │          │─transcribe── │                 │            │            │
  │          │◀─text────────│                 │            │            │
  │◀─confirm─│              │                 │            │            │
  │─approve──▶              │                 │            │            │
  │          │────query─────▶                 │            │            │
  │          │              │─intent extract  │            │            │
  │          │              │─FHIR GET─────────▶            │            │
  │          │              │◀─patient data────│            │            │
  │          │              │─vector search─────────────────▶            │
  │          │              │◀─relevant chunks──────────────│            │
  │          │              │─interaction check──────────────────────────▶
  │          │              │◀─interaction result────────────────────────│
  │          │              │─calc NEWS2/qSOFA/eGFR         │            │
  │          │              │─allergy check                 │            │
  │          │              │─generate (grounded + citation)│            │
  │◀──────────────response──│                 │            │            │
  │          │              │─[session end] purge PHI       │            │
```

---

## 11. Truy vết User Stories → Flow

| User Story | Mô tả | Flow tương ứng |
|-----------|-------|----------------|
| US-01 | Hỏi bằng giọng nói, nhận câu trả lời có nguồn | Mục 2 Golden Path |
| US-02 | Tự động check drug interaction | Mục 2 Bước 3–4 |
| US-03 | Xem NEWS2 hiện tại | Mục 4 NEWS2 Flow |
| US-04 | Cảnh báo rõ ràng khi có dị ứng | Mục 3 Allergy Alert Flow |
| US-05 | "Không đủ thông tin" thay vì đoán | Mục 6 Fallback Flow |
| US-06 | Điều dưỡng xem nhanh dị ứng bằng giọng nói | Mục 7 Secondary User Flow |

---
