"""
R2 — Query router: intent classification + clinical entity extraction.

One LLM call returns {intent, drugs, procedures}. Intent drives retrieval
(safety-priority for contraindication) and the off-topic refusal. A keyword
fallback keeps the pipeline alive if the LLM call fails or returns garbage.
"""

import json
import re
import sys
from pathlib import Path

# Windows consoles default to cp1252 and choke on Vietnamese — force UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ on sys.path
from embedding.or_client import ChatClient  # noqa: E402

INTENTS = ("procedure", "contraindication", "dosing", "scoring", "general", "off_topic")

# Keywords for the no-LLM fallback path (mirrors retriever.SAFETY_KEYWORDS).
_SAFETY_KW = ("chống chỉ định", "không được dùng", "không nên", "không được",
              "contraindication", "nguy hiểm", "tránh dùng", "có được", "được không")
_DOSING_KW = ("liều", "dose", "dosing", "mg/kg", "điều chỉnh liều", "titrate")
_SCORING_KW = ("news2", "qsofa", "sofa", "egfr", "map", "điểm", "score")

_ROUTER_PROMPT = """\
Bạn là bộ phân loại câu hỏi cho trợ lý lâm sàng ICU. Phân loại câu hỏi vào MỘT intent:
- "procedure": hỏi quy trình / cách tiến hành / điều trị / xử trí
- "contraindication": hỏi chống chỉ định, khi nào KHÔNG được làm / không nên dùng, rủi ro an toàn
- "dosing": hỏi liều thuốc / điều chỉnh liều
- "scoring": hỏi điểm số lâm sàng (NEWS2, qSOFA, SOFA, eGFR, MAP) của bệnh nhân
- "general": câu hỏi y khoa ICU khác
- "off_topic": KHÔNG liên quan y khoa

Đồng thời liệt kê tên thuốc và tên thủ thuật được nhắc đến (giữ nguyên cách viết).

Chỉ trả về JSON, không giải thích:
{"intent": "...", "drugs": ["..."], "procedures": ["..."]}

Ví dụ:
Câu hỏi: "Bệnh nhân dị ứng Penicillin, dùng Amoxicillin được không?"
{"intent": "contraindication", "drugs": ["Amoxicillin", "Penicillin"], "procedures": []}
Câu hỏi: "Các bước tiến hành lọc máu liên tục?"
{"intent": "procedure", "drugs": [], "procedures": ["lọc máu liên tục"]}
Câu hỏi: "NEWS2 của bệnh nhân này bao nhiêu?"
{"intent": "scoring", "drugs": [], "procedures": []}
Câu hỏi: "Kết quả bóng đá tối qua thế nào?"
{"intent": "off_topic", "drugs": [], "procedures": []}
"""


def parse_json_loose(text: str) -> dict:
    """Parse JSON out of an LLM reply that may carry fences or prose."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in: {text[:200]}")
    return json.loads(text[start:end + 1])


def keyword_route(query: str) -> dict:
    """LLM-free fallback router (keyword heuristics)."""
    q = query.lower()
    if any(kw in q for kw in _SAFETY_KW):
        intent = "contraindication"
    elif any(kw in q for kw in _DOSING_KW):
        intent = "dosing"
    elif any(kw in q for kw in _SCORING_KW):
        intent = "scoring"
    else:
        intent = "general"
    return {"intent": intent, "drugs": [], "procedures": [], "via": "keyword"}


def route(query: str, chat: ChatClient) -> dict:
    """Classify a query; never raises (falls back to keyword routing)."""
    try:
        reply = chat.chat(
            [{"role": "system", "content": _ROUTER_PROMPT},
             {"role": "user", "content": f"Câu hỏi: \"{query}\""}],
            temperature=0.0, max_tokens=200,
        )
        data = parse_json_loose(reply)
        intent = data.get("intent")
        if intent not in INTENTS:
            return keyword_route(query)
        return {
            "intent": intent,
            "drugs": [d for d in data.get("drugs", []) if isinstance(d, str)],
            "procedures": [p for p in data.get("procedures", []) if isinstance(p, str)],
            "via": "llm",
        }
    except Exception as err:
        print(f"  [router fallback] {err}", file=sys.stderr)
        return keyword_route(query)
