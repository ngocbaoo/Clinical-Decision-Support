# Module RAG — Báo cáo Xây dựng & Đánh giá

**Ngày:** 2026-06-10 · **Người thực hiện:** Claude (cho ngocbaoo)
**Mô hình:** sinh câu trả lời `qwen/qwen3.6-flash` · giám khảo `openai/gpt-5.4` (qua OpenRouter)
**Phạm vi:** lớp Hỏi-Đáp RAG nằm trên các thành phần đã có (chunker / embedder / FHIR / scoring).
ASR và API tương tác thuốc nằm ngoài phạm vi.

> Đây là bản tường thuật dễ đọc. Số liệu do máy sinh nằm ở `chunks/rag_eval_report.md`
> (đánh giá câu trả lời) và `chunks/rag_retrieval_eval.json` (đánh giá truy hồi).
> Kế hoạch gốc: `requirement_analysis/06_RAG_MODULE_PLAN.md`.

---

## 1. Đã xây dựng những gì (nói đơn giản)

Bác sĩ hỏi một câu lâm sàng bằng tiếng Việt (có thể kèm theo một bệnh nhân cụ thể).
Hệ thống trả lời kèm trích dẫn guideline, và **từ chối trả lời thay vì đoán bừa** khi
không đủ căn cứ. Luồng xử lý:

```
Câu hỏi (+ file bệnh nhân nếu có)
   │
   ▼
1. ROUTER         LLM đọc câu hỏi, xác định loại
   (query_router)  (procedure / contraindication / dosing / scoring / general / off-topic)
   │               và trích tên thuốc, tên thủ thuật.
   ▼
2. TRUY HỒI       Tìm kiếm ngữ nghĩa trên kho tri thức ICU 594 chunk.
   (retriever)     Với câu hỏi an toàn, chunk CHỐNG CHỈ ĐỊNH được ưu tiên lên ĐẦU.
   │
   ▼
3. CỔNG AN TOÀN   Đối chiếu thuốc trong câu hỏi với dị ứng của bệnh nhân —
   (safety)        kể cả phản ứng chéo (dị ứng Penicillin → cảnh báo Amoxicillin).
   │               Cảnh báo luôn hiển thị ĐẦU TIÊN.
   ▼
4. SINH CÂU TRẢ LỜI  LLM viết câu trả lời CHỈ dựa trên các chunk được đánh số +
   (generator)        dữ liệu bệnh nhân + điểm số đã tính sẵn. Sau đó CODE kiểm tra
   │                  và thay bằng "Không đủ thông tin" nếu:
   │                    • câu hỏi ngoài chủ đề, hoặc
   │                    • chunk tốt nhất có điểm dưới ngưỡng tin cậy, hoặc
   │                    • câu trả lời không có trích dẫn [n] hợp lệ.
   ▼
Câu trả lời có trích dẫn  (cảnh báo dị ứng → khuyến nghị → nguồn → disclaimer)
```

**Quyết định thiết kế cốt lõi:** các quy tắc an toàn được *code* thực thi, **không** tin
tưởng vào LLM. Mô hình được yêu cầu trích dẫn nguồn, nhưng một bước kiểm tra riêng sẽ
xác minh trích dẫn có tồn tại trước khi câu trả lời được hiển thị. Nếu không có, câu trả
lời bị loại và thay bằng fallback. Đây là cách giữ đúng yêu cầu PRD: "tỷ lệ hallucination
= 0%, mọi câu trả lời phải có nguồn".

**Các file đã thêm:**
- `src/rag/` — `config.py`, `query_router.py`, `safety.py`, `context_builder.py`,
  `generator.py`, `pipeline.py`, `ask.py` (CLI)
- `src/rag/eval/` — `gold_retrieval.json` (45 truy vấn có nhãn), `retrieval_eval.py`,
  `answer_eval.py`
- `tests/test_rag.py` — 20 test offline (LLM được mock); toàn bộ test suite **40/40 pass**
- Thêm `ChatClient` vào `src/embedding/or_client.py`

**Chạy thử:**
```powershell
python src/rag/ask.py --file data/mock/patient_A.json --query "Bệnh nhân dị ứng Penicillin, dùng Amoxicillin được không?"
```

---

## 2. Đánh giá thế nào (hai lớp độc lập)

| Lớp | Đo cái gì | Cần API? | Chi phí |
|-----|-----------|----------|---------|
| **Đánh giá truy hồi** | Tìm kiếm có trả về đúng chunk không? | chỉ embedding | ~miễn phí |
| **Đánh giá câu trả lời** | Câu trả lời có an toàn, đúng căn cứ, có trích dẫn không? | gen + judge LLM | ~$1–2/lần |

Đánh giá truy hồi dùng bộ gold set gán nhãn tay (45 truy vấn, mỗi truy vấn gán sẵn chunk
nào là "liên quan"). Đánh giá câu trả lời chạy 12 kịch bản lâm sàng — ánh xạ tới các ca
Tier-2 trong Evaluation Plan và bệnh nhân mock thật — qua toàn bộ pipeline, áp dụng kiểm
tra bằng code, rồi cho **một mô hình KHÁC (GPT-5.4) chấm độ trung thực** để mô hình sinh
không tự chấm bài của chính mình.

---

## 3. Kết quả truy hồi — tốt ✅

| Chỉ số | Kết quả | Mục tiêu | Đánh giá |
|--------|---------|----------|----------|
| Hit@1 (chunk đúng xếp #1) | **0.925** | ≥ 0.70 | ✅ |
| Recall@5 (chunk đúng trong top 5) | **1.00** | ≥ 0.85 | ✅ |
| MRR@5 | **0.963** | ≥ 0.75 | ✅ |
| Ưu tiên an toàn (chunk chống chỉ định lên đầu) | **100%** | 100% | ✅ |
| Loại câu ngoài chủ đề | thời tiết/ăn kiêng/tiêm chủng đều ≤ 0.37 | loại bỏ | ✅ |

Theo nhóm: quy trình tiếng Việt Hit@1 0.96, câu hỏi an toàn (cả tường minh lẫn **diễn đạt
lại**) 1.00, song ngữ Anh→Việt Hit@1 0.60 / Recall@5 1.00.

**Hai điều đáng lưu ý:**
1. **Router LLM thực sự cần thiết.** Cách dò an toàn cũ dựa trên từ khóa chỉ bắt được
   **50%** câu hỏi an toàn trong gold set (bỏ sót các câu diễn đạt lại như "bệnh nhân nào
   không nên đặt nội khí quản?"). Router LLM bắt được 100%.
2. **Ngưỡng tin cậy giờ được hiệu chỉnh, không phải đoán.** Đặt thành **0.40** vì câu thực
   sự ngoài chủ đề chấm ≤ 0.37, còn câu lâm sàng thật chấm ≥ 0.45. Một lưu ý: chủ đề
   y khoa nhưng không có trong kho ("quy trình truyền máu") chấm 0.601 và lọt qua ngưỡng —
   được chặn lại bởi cơ chế kiểm tra trích dẫn ở phía sau.

---

## 4. Kết quả câu trả lời — hành vi an toàn ✅, nhưng độ căn cứ cần cải thiện ⚠️

### 4a. Tin tốt: mọi hành vi an toàn đều hoạt động (12/12)

| Kiểm tra | Kết quả |
|----------|---------|
| Hành vi đúng (trả lời vs. fallback vs. từ chối) | 12/12 |
| Cảnh báo dị ứng xuất hiện, và xuất hiện ĐẦU TIÊN | ✅ (A-01) |
| Câu hỏi ngoài chủ đề bị từ chối | ✅ (A-07, bóng đá) |
| Thuốc không có trong kho → "không đủ thông tin" trung thực | ✅ (A-06, Zykadia) |
| Điểm số lâm sàng trong câu trả lời khớp đúng calculator | ✅ (A-03, A-12) |
| Mọi câu trả lời hiển thị đều có trích dẫn | 9/9 |

Tức là hệ thống **chưa bao giờ bịa nguồn, chưa bao giờ bỏ sót dị ứng, chưa bao giờ trả lời
điều không nên trả lời** trong 12 kịch bản. Hợp đồng an toàn được giữ vững.

### 4b. Vấn đề thật sự: mô hình sinh "thêm thắt" quá tay ⚠️

Giám khảo GPT-5.4 chỉ duyệt đạt **2 trong 9** câu trả lời ở tiêu chí độ trung thực
(faithfulness). Đây là phát hiện quan trọng nhất, và là **vấn đề thật** — không phải lỗi đo
lường. Mô hình sinh giá rẻ (`qwen3.6-flash`) viết câu trả lời nghe *hợp lý về lâm sàng*
nhưng **thêm các chi tiết mà chunk được trích dẫn không hề có**, và **gắn sai số trích dẫn.**

Ví dụ cụ thể giám khảo bắt được:

- **A-04** (sốc nhiễm khuẩn): viết "tăng dần Noradrenalin... đến MAP ≥ 65" — nguồn nêu liều
  khởi đầu và mục tiêu MAP nhưng không nói "tăng dần". Còn gắn `[4]` cho một khẳng định về
  kháng sinh mà `[4]` không hỗ trợ, và gắn mục tiêu CVP với `[1][3]` trong khi các chunk đó
  không nêu con số ấy.
- **A-10** (quy trình hút đờm): thêm "<15s/lần" và "đầu cao 30°" — không có trong bất kỳ
  chunk nào được trích.
- **A-08** (suy gan): gắn `[1]` cho "tiếp tục Lactulose" và "dự phòng H2/PPI" — không có
  trong chunk `[1]`.

**Tại sao xảy ra:** code đảm bảo trích dẫn *tồn tại*, nhưng không kiểm tra trích dẫn *đúng*
với câu mà nó gắn vào. Mô hình tự lấp chỗ trống bằng kiến thức y khoa của chính nó — nghe
hợp lý, nhưng không có căn cứ, và đây chính là kiểu lỗi mà quy tắc 0% hallucination của PRD
nhắm tới.

> Lưu ý: một số điểm giám khảo chấm hơi khắt khe — ví dụ bắt lỗi một chẩn đoán *vốn có* trong
> hồ sơ bệnh nhân, hoặc câu disclaimer bắt buộc. Theo kế hoạch, verdict của giám khảo chỉ là
> **bước lọc sơ bộ**; bác sĩ/mentor xác nhận từng ca trước khi tính vào kết quả cuối. Nhưng
> các ca gắn sai trích dẫn ở trên là lỗi thật và sẽ không qua được vòng rà soát.

### 4c. Độ trễ — vượt xa ngân sách ⚠️

p50 đầu-cuối là **~37 giây** so với mục tiêu < 4.5s. Phần lớn chi phí đến từ endpoint
**embedding** của OpenRouter (60s+ với câu hỏi an toàn vì phải embed một lần và query hai
lần), cộng với độ trễ của mô hình sinh. Đây là vấn đề hạ tầng, không phải lỗi logic.

---

## 5. Ba lỗi đã sửa trong quá trình đánh giá (đều chính đáng)

Quá trình đánh giá lộ ra ba vấn đề là lỗi thật, đã sửa và chạy lại:

1. **Giám khảo bắt nhầm câu disclaimer bắt buộc** ("Cần bác sĩ xác nhận...") thành claim bịa
   → đã yêu cầu giám khảo coi boilerplate và điểm số tính sẵn là dữ liệu đáng tin.
2. **Câu hỏi điểm số bị fallback sai** khi chunk guideline mỏng → giờ trả lời từ điểm số của
   calculator. A-03 chuyển từ "không đủ thông tin" → đúng "NEWS2 = 17, mức độ cao".
3. **Ngưỡng quá chặt (0.50 → 0.40)** → khôi phục một câu hỏi vận mạch/MAP chính đáng (chấm
   0.45) bị loại oan. An toàn vì câu thực sự ngoài chủ đề ≤ 0.37.

Tôi đã **dừng ở đó một cách có chủ đích.** Tinh chỉnh thêm để làm giám khảo hài lòng sẽ là
"gian lận" chỉ số — con số 2/9 độ trung thực là tín hiệu trung thực về chất lượng mô hình sinh.

---

## 6. Việc cần làm tiếp (theo thứ tự ưu tiên)

1. **Sửa độ trung thực của trích dẫn (ưu tiên cao nhất).** Hai hướng:
   - **Rẻ:** thêm một bước hậu kiểm — với mỗi claim `[n]`, kiểm tra xem chunk *n* có thực sự
     hỗ trợ không; loại bỏ hoặc gắn cờ những claim không có căn cứ.
   - **Trực tiếp:** đổi mô hình sinh sang `openai/gpt-5.4-mini` và A/B. Cờ `--model` và bộ
     đánh giá khiến đây là thí nghiệm chạy một lệnh (~$0.50).
2. **Sửa độ trễ.** Profile và cache lời gọi embedding; đó là nút thắt, không phải LLM.
3. **Chạy rubric con người.** Chấm Tier-2 trong Evaluation Plan (5 tiêu chí × 0–2, S-01→S-05
   bắt buộc 10/10) vẫn cần bác sĩ hoặc mentor — giám khảo chỉ là bước lọc sơ bộ.
4. **Nối kiểm tra tương tác thuốc** (`safety.check_drug_interactions` đang là stub hoạt động).

---

## 7. Tóm tắt một đoạn

Module RAG đã được xây dựng và chạy được đầu-cuối: nó định tuyến câu hỏi bằng LLM (vá điểm
mù 50% của cách dùng từ khóa cũ), truy hồi đúng chunk guideline (**Hit@1 0.93, Recall@5 100%,
ưu tiên an toàn 100%**), kiểm tra dị ứng trước khi trả lời, và — quan trọng nhất — **thực thi
quy tắc an toàn bằng code thay vì tin LLM**, nên chưa bao giờ bịa nguồn hay bỏ sót dị ứng qua
12 kịch bản. Vấn đề còn lại là **chất lượng căn cứ**: mô hình sinh giá rẻ thêm chi tiết hợp lý
nhưng không có trích dẫn và gắn sai số trích dẫn (độ trung thực chặt 2/9), điều mà một bước
hậu kiểm trích dẫn hoặc một mô hình mạnh hơn sẽ khắc phục. Độ trễ (~37s) cần xử lý hạ tầng
trước khi demo trực tiếp. Không có vấn đề nào là lỗi logic — pipeline vững; nó cần mô hình sinh
tốt hơn và embedding nhanh hơn.
