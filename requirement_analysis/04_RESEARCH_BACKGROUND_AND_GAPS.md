# 04 · Research Background & Gaps
# ASR + RAG Clinical Assistant — ICU Decision Support
**Phiên bản:** 1.1 · **Trạng thái:** Draft
**Ngày lập:** 01/06/2026 · **Tác giả:** Tạ Bảo Ngọc
**Tài liệu nguồn:** `01_PRD_V1_ASR_RAG_CLINICAL_ASSISTANT.md`
---

## 1. Tại sao bài toán này quan trọng

### 1.1 Số liệu thực tế

- WHO ước tính **1/10 bệnh nhân** bị tổn hại do medical error trong bệnh viện
- Medication error là nguyên nhân hàng đầu trong adverse events tại ICU
- Bệnh nhân ICU dùng trung bình **10–20 thuốc đồng thời** → hàng chục cặp tương tác cần kiểm tra
- **40%** medication error xảy ra ở giai đoạn kê đơn — đây chính là điểm hệ thống RAG có thể can thiệp
- Ca trực ICU kéo dài 24–36 tiếng → cognitive fatigue là yếu tố rủi ro có thể đo lường được

### 1.2 Vì sao ICU đặc biệt khó

| Yếu tố | Tác động | Liên hệ PRD |
|--------|---------|-------------|
| Noise level 60–80 dB liên tục (máy thở, monitor alarm) | ASR truyền thống thất bại | Out of scope: noise cancellation phức tạp |
| Tay bác sĩ luôn bận (thủ thuật, thăm khám) | Không thể dùng keyboard/touchscreen | F-ASR-01 push-to-talk |
| Code-switching Việt-Anh trong clinical speech | Model ASR đơn ngữ không đủ | F-ASR-02 |
| Quyết định trong vài phút, không phải vài giờ | Latency là yêu cầu cứng | NFR latency < 5s |
| Hậu quả error không thể đảo ngược | Accuracy là yêu cầu cứng hơn latency | PRD §5: "Accuracy > Latency" |

> Hệ thống nhắm vào tình huống **bán khẩn cấp (semi-urgent)** — bệnh nhân ổn định nhưng có dấu hiệu xấu đi (PRD 2.2), là window đủ để bác sĩ query.

---

## 2. Landscape hiện tại

### 2.1 Clinical Decision Support Systems (CDSS)

**Các hệ thống thương mại hiện có:**
- **Epic CDS Hooks** — tích hợp sâu vào Epic EMR, alert-based, không có conversational interface
- **IBM Watson for Oncology** — bị ngừng 2022 do accuracy thấp hơn bác sĩ trong nhiều trường hợp
- **Isabel DDx** — differential diagnosis, không có voice interface
- **UpToDate** — reference tool phổ biến nhất, không có AI synthesis, cần tìm tay

**Gap chung của các hệ thống hiện có:**
- Không có voice interface → bác sĩ phải dùng tay
- Alert-based (passive) thay vì conversational (active query)
- Không tổng hợp context bệnh nhân cụ thể với guidelines
- Không phù hợp với workflow ICU thực tế ở Việt Nam (Western-centric, không dùng phác đồ BYT VN)

### 2.2 Medical ASR — State of the Art

Đây là các **candidate model** của project (PRD §7.1) — **chưa chốt**, quyết định cuối điền sau khi test thực tế.

| Model | Năm | WER (medical EN) | Tiếng Việt | Code-switching | Ghi chú |
|-------|-----|-----------------|-----------|----------------|---------|
| Whisper large-v3 (OpenAI) | 2023 | ~25–33% | Multilingual | Chưa test | OOV với medical terms |
| MedASR (Google) | 2025 | **4.6–9.3%** | No | Chưa test | English-only medical |
| MultiMed-ST whisper-vi (VietMed) | 2025 | TBD | Medical VI | Chưa test | Vietnamese medical |
| MultiMed-ST whisper-multilingual | 2025 | TBD | Yes | Chưa test | Candidate gần nhất cho code-switching |

**Key finding:** Chưa có model nào giải quyết hoàn hảo cả ba yêu cầu: tiếng Việt + medical terminology + code-switching. MultiMed-ST multilingual là candidate gần nhất nhưng **cần test thực tế** mới chốt.

**Fallback strategy (PRD §7.1):** Nếu các candidate không đạt, fine-tune **PhoWhisper-small** trên dataset **VietMed**.

**Paper tham khảo:**
- Le-Duc, K. (2024). *VietMed: A Dataset and Benchmark for ASR of Vietnamese in the Medical Domain*. LREC-COLING 2024.
- Radford, A. et al. (2022). *Robust Speech Recognition via Large-Scale Weak Supervision* (Whisper paper). arXiv:2212.04356.
- Le-Duc, K. et al. (2025). *MultiMed-ST: Large-scale Many-to-many Multilingual Medical Speech Translation*. EMNLP 2025.

### 2.3 Medical RAG — State of the Art

**Vấn đề với LLM thuần túy trong medical context:**
- GPT-4 hallucinate thông tin y tế với confidence cao — nguy hiểm hơn không biết gì
- Không có khả năng cite nguồn cụ thể theo mặc định
- Knowledge cutoff → guidelines cũ không được update

**RAG giải quyết như thế nào:**
```
LLM thuần túy:
  Input → LLM → Output (từ tham số đã train)
  → Có thể hallucinate, không có source

RAG:
  Input → Retrieve relevant chunks → LLM + context → Output + citation
  → Grounded, có source, có thể kiểm chứng
```

Đây là cơ sở cho Safety Req của PRD §4.3: **citation rate 100%, hallucination rate 0%** — từ chối trả lời nếu không có source.

**Thách thức của medical RAG và cách project xử lý:**

1. **Chunking strategy** — guideline y tế có cấu trúc phức tạp. Project chunk tài liệu `quy_trinh_icu_vn.pdf` (892 trang) **theo từng quy trình** → 232 chunks, mỗi chunk giữ nguyên Khái niệm / Chỉ định–Chống chỉ định / Các bước / Theo dõi tai biến (PRD §6.3). Tránh chunk cứng theo số ký tự làm mất context "Chống chỉ định".
2. **Query-document mismatch** — bác sĩ hỏi "liều Vancomycin cho suy thận", document viết "renal dose adjustment" → cần multilingual embedding tốt (BGE-M3 / Qwen3 / Gemini — cần test).
3. **Multi-hop reasoning** — "septic shock + suy thận → kháng sinh nào?" cần combine nhiều chunks + FHIR patient context.
4. **Structured vs unstructured split** — LOINC/ICD-10 là exact lookup (SQLite), **không** chunk vào Vector DB; chỉ guideline văn bản mới embed (PRD §6.2–6.3).

**Paper tham khảo:**
- Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020.
- Zakka, C. et al. (2024). *Almanac — Retrieval-Augmented Language Models for Clinical Medicine*. NEJM AI.

### 2.4 Kiến trúc dữ liệu của project

Theo PRD 6, dữ liệu chia làm 4 lớp riêng biệt — quan trọng cho việc hiểu vì sao mỗi loại được xử lý khác nhau:

**(a) FHIR R4 — patient context (dynamic)**

FHIR R4 được chọn vì là chuẩn quốc tế, có sandbox công khai (SMART Health IT — `launch.smarthealthit.org`), và structured data dễ extract cho tính NEWS2/qSOFA/eGFR.

| Resource | Dùng để | LOINC / mã quan trọng |
|----------|---------|----------------------|
| Patient | Demographics (tuổi, cân nặng, giới tính) cho eGFR | — |
| Encounter | Lần nhập viện hiện tại (không query trực tiếp) | — |
| Observation | Vital signs, lab results cho NEWS2/qSOFA | 59408-5 (SpO₂), 8867-4 (HR), 9279-1 (RR), 8480-6 (SBP), 8310-5 (Temp), 2160-0 (Creatinine), 32693-4 (Lactate) |
| MedicationRequest | Drug interaction check + thuốc đang dùng | — |
| MedicationAdministration | Thuốc đã dùng thực tế | — |
| AllergyIntolerance | Safety check — query đầu tiên | — |
| Condition | Chẩn đoán ICD-10 → chọn đúng guideline + check hypercapnic RF | A41 (Sepsis), J18.9 (Pneumonia), N17 (AKI) |
| DiagnosticReport | Lab panels tổng hợp cho sepsis workup | — |
| Procedure | Thủ thuật đã làm (vd: đang thở O₂) | — |

> Lưu ý kỹ thuật (PRD 6.1): mỗi Observation lấy giá trị mới nhất bằng `_sort=-date&_count=1`, tránh lấy data của lần nhập viện cũ.

**(b) Lookup tables — SQLite (`clinical_db.sqlite`, static, exact lookup)**
- `loinc_icu` — **28 codes** ICU-relevant (filter từ LOINC full DB) → dịch LOINC code sang tên đưa vào LLM context.
- `icd10` — **10.663 codes** bilingual (parse từ ICD-10 Việt Nam, QĐ Bộ Y tế) → dịch mã chẩn đoán sang tên tiếng Việt.

**(c) Knowledge Base — Vector DB (semantic search)**
- Tài liệu: **Quy trình kỹ thuật hồi sức cấp cứu — 232 quy trình** (Bộ Y tế VN, QĐ 1904/QĐ-BYT, 2014, PDF 892 trang). Đây là tài liệu **đã có sẵn** — không còn là "nếu available".
- Vector DB: **ChromaDB**; embedding multilingual (đang cân nhắc BGE-M3 / Qwen3-Embedding-8B / text-embedding-3-small / Gemini Embedding).

**(d) Drug Interaction — API ngoài (không tự build DB)**
- **DrugBank** (đề xuất dùng — free cho academic, 15.000+ drugs, có cả contraindication).
- Dự phòng: **OpenFDA Drug Label**, **RxNav (NIH)**.

---

## 3. Clinical Domain Knowledge

### 3.1 NEWS2 — Vai trò trong hệ thống (F-RAG-06)

NEWS2 (National Early Warning Score 2) đánh giá mức độ nguy hiểm của bệnh nhân dựa trên 7 chỉ số sinh tồn.

**Tại sao tự động hóa NEWS2 có giá trị:**
- Hiện tại điều dưỡng tính tay → mất 2–3 phút và có thể sai
- NEWS2 ≥ 7 yêu cầu phản ứng cấp cứu ngay — delay là nguy hiểm
- Hệ thống tính tự động từ FHIR Observation data

**Lưu ý kỹ thuật quan trọng:**
- NEWS2 có **2 thang SpO₂** — Scale 1 (mặc định) và Scale 2 (cho hypercapnic respiratory failure)
- Phải check `Condition` resource để biết bệnh nhân có hypercapnic RF không
- Nếu dùng sai thang → score sai → clinical decision sai

### 3.2 Sepsis-3 & qSOFA (F-RAG-07)

```
Sepsis-3 chẩn đoán dựa trên:
  SOFA score tăng ≥ 2 điểm
    → Cần: PaO₂/FiO₂ (ABG), Bilirubin, Creatinine, Tiểu cầu, Glasgow, MAP, vasopressor

  qSOFA (screening nhanh, chỉ cần 3 chỉ số):
    → Glasgow < 15 OR Nhịp thở ≥ 22 OR SBP ≤ 100

  Septic Shock thêm:
    → Cần vasopressor để MAP ≥ 65 AND Lactate > 2 mmol/L
```

**Hệ quả cho RAG:**
- **qSOFA**: tính được real-time từ FHIR Observation (Glasgow, RR, SBP) → dùng làm screening. Đây là phần PoC triển khai.
- **SOFA đầy đủ**: cần lab results (delay 2–4h) → ngoài scope tính tự động, chỉ gợi ý khi qSOFA ≥ 2.
- Hệ thống phải hiển thị rõ đang dùng qSOFA hay full SOFA và tại sao.

### 3.3 Medication Safety trong ICU (F-RAG-05, F-RAG-08)

**Drug-Drug Interaction:**
- Bệnh nhân ICU dùng nhiều thuốc IV cùng lúc
- Interaction nguy hiểm: Vancomycin + Gentamicin (nephrotoxicity), Heparin + NSAIDs (bleeding)
- Project check qua **DrugBank API** dựa trên danh sách từ `MedicationRequest`.

**Renal/Hepatic Dose Adjustment:**
- 40–60% bệnh nhân ICU có suy thận hoặc suy gan
- Nhiều thuốc thải qua thận/gan → tích lũy → độc tính
- PRD F-RAG-08 yêu cầu điều chỉnh liều theo **eGFR**, tính từ Creatinine + tuổi + cân nặng + giới tính (lấy từ `Patient`).

```python
# CKD-EPI 2021 (không dùng race):
# eGFR = 142 × min(Scr/κ, 1)^α × max(Scr/κ, 1)^(-1.200) × 0.9938^Age × (1.012 nếu nữ)
# κ = 0.7 (nữ), 0.9 (nam);  α = -0.241 (nữ), -0.302 (nam)
#
# Lưu ý: CKD-EPI không dùng cân nặng. Nếu cần liều theo cân nặng tuyệt đối
# (vd Cockcroft-Gault, dosing mg/kg), cân nặng từ Patient mới cần đến.
```

---

## 4. Research Gaps mà hệ thống này giải quyết

| Gap | Hiện trạng | Cách hệ thống xử lý |
|-----|-----------|-------------------|
| **Voice interface cho CDSS** | Hầu hết CDSS dùng keyboard/click | ASR với push-to-talk (F-ASR-01) |
| **Vietnamese medical ASR** | Chưa có production-ready solution | Candidate MultiMed-ST + fallback PhoWhisper/VietMed |
| **Grounded medical Q&A** | LLM thuần túy hallucinate | RAG với citation 100%, hallucination 0% (Safety Req) |
| **Patient-specific recommendations** | Generic guidelines, không tích hợp data bệnh nhân | FHIR query + RAG synthesis |
| **ICU-specific workflow** | Tools design cho outpatient/general ward | Push-to-talk, ngắn gọn, latency < 5s |
| **Vietnamese clinical context** | Hầu hết tools Western-centric | Knowledge base = Quy trình hồi sức cấp cứu BYT VN 2014 (đã có file) + ICD-10 bilingual |

---

## 5. Giới hạn của approach hiện tại

### 5.1 Technical Gaps (Out of scope trong PoC 6 tuần)

| Vấn đề | Lý do chưa giải quyết | Plan tương lai |
|--------|----------------------|----------------|
| Ambient noise trong ICU | Push-to-talk là workaround, không phải giải pháp lâu dài | Beamforming mic + noise cancellation |
| Multi-speaker diarization | Quá phức tạp cho 6 tuần | Phase 2 |
| Trend analysis (time series) | Chỉ dùng point-in-time data | Phase 3 với time series model |
| Real-time monitor integration | Cần hospital IT infrastructure | Phase 2 |
| HIS/EMR integration thật | PoC dùng synthetic data (SMART sandbox) | Production với FHIR server bệnh viện đối tác |
| SOFA đầy đủ tự động | Cần lab delay 2–4h | Khi có đủ DiagnosticReport |

### 5.2 Clinical Validation Gap

**Rủi ro lớn nhất của project:** Không có clinician trong team để validate RAG output.

```
Scenario bị miss:
  RAG trả lời đúng theo guideline quốc tế
  NHƯNG sai theo phác đồ của bệnh viện cụ thể đó
  → Bác sĩ có thể trust nhưng outcome sai
```

**Mitigation:** Ghi rõ limitation + disclaimer "Cần bác sĩ xác nhận trước khi thực hiện" trong mọi output (Safety Req #4). Cần ít nhất 1 buổi clinical workflow testing với clinician trong **Tuần 4**.

---

## 6. Nguồn tham khảo

1. Le-Duc, K. (2024). VietMed: A Dataset and Benchmark for ASR of Vietnamese in the Medical Domain. *LREC-COLING 2024*.
2. Le-Duc, K. et al. (2025). MultiMed-ST: Large-scale Many-to-many Multilingual Medical Speech Translation. *EMNLP 2025*. arXiv:2504.03546.
3. Singer, M. et al. (2016). The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). *JAMA*, 315(8), 801–810.
4. Lewis, P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020*.
5. Zakka, C. et al. (2024). Almanac — Retrieval-Augmented Language Models for Clinical Medicine. *NEJM AI*.
6. Evans, L. et al. (2021). Surviving Sepsis Campaign: International Guidelines for Management of Sepsis and Septic Shock 2021. *Critical Care Medicine*.
7. Royal College of Physicians (2017). National Early Warning Score (NEWS) 2. London: RCP.
8. Bộ Y tế Việt Nam (2014). Quy trình kỹ thuật hồi sức cấp cứu (232 quy trình). QĐ 1904/QĐ-BYT.
9. Inker, L.A. et al. (2021). New Creatinine- and Cystatin C–Based Equations to Estimate GFR without Race (CKD-EPI 2021). *NEJM*, 385, 1737–1749.
10. HL7 International (2019). FHIR R4 Specification. https://hl7.org/fhir/R4/
11. WHO (1992). ICD-10: International Statistical Classification of Diseases. Geneva: WHO.
12. DrugBank. https://go.drugbank.com/ · OpenFDA Drug Label API · RxNav (NIH).

---
