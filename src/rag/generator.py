"""
R4 — Generator: grounded answer with enforced citations + fallback.

The LLM is asked for {answer, citations, confidence, insufficient}; the code
(not the LLM) then enforces the safety contract:
  - off-topic intent           -> refusal, no LLM call
  - retrieval top score < thr  -> fallback (F-RAG-09)
  - LLM says insufficient      -> fallback
  - no valid [n] citations     -> fallback (hallucination guard)
Alerts from the safety gate are prepended to whatever is returned.
"""

import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ on sys.path
from embedding.or_client import ChatClient  # noqa: E402
from rag.config import CONF_THRESHOLD, DISCLAIMER  # noqa: E402
from rag.context_builder import build_messages, summarize_patient  # noqa: E402
from rag.query_router import parse_json_loose  # noqa: E402
from rag.safety import format_alerts  # noqa: E402

FALLBACK_TEXT = ("Không đủ thông tin trong cơ sở tri thức để đưa ra khuyến nghị "
                 "đáng tin cậy cho câu hỏi này. Vui lòng tra cứu guideline gốc "
                 "hoặc hỏi dược sĩ lâm sàng.")
OFF_TOPIC_TEXT = ("Câu hỏi nằm ngoài phạm vi hỗ trợ lâm sàng ICU — "
                  "tôi chỉ trả lời các câu hỏi y khoa dựa trên guideline.")


def _response(answer: str, citations: list[int], chunks: list[dict],
              alerts: list[dict], fallback: bool, reason: str | None,
              confidence: float | None) -> dict:
    cited_sources = [
        {"n": n, "source": chunks[n - 1]["source"], "title": chunks[n - 1]["title"]}
        for n in citations if 1 <= n <= len(chunks)
    ]
    alert_text = format_alerts(alerts)
    full = (alert_text + "\n\n" if alert_text else "") + answer
    if not fallback and DISCLAIMER not in full:
        full = f"{full}\n\n{DISCLAIMER}"
    return {
        "answer": full,
        "citations": citations,
        "cited_sources": cited_sources,
        "alerts": alerts,
        "fallback": fallback,
        "fallback_reason": reason,
        "confidence": confidence,
    }


def _extract_citations(data: dict, answer_text: str, n_chunks: int) -> list[int]:
    """Citations from the JSON field, plus any inline [n] markers; validated."""
    cites = set()
    for c in data.get("citations", []) or []:
        if isinstance(c, (int, float)):
            cites.add(int(c))
    for m in re.finditer(r"\[(\d{1,2})\]", answer_text):
        cites.add(int(m.group(1)))
    return sorted(c for c in cites if 1 <= c <= n_chunks)


def generate(query: str, intent: str, chunks: list[dict], patient_context: dict,
             calc: dict, alerts: list[dict], chat: ChatClient,
             threshold: float = CONF_THRESHOLD) -> dict:
    """Produce the final response dict (never raises on LLM/parse failure)."""
    if intent == "off_topic":
        return _response(OFF_TOPIC_TEXT, [], [], alerts, True, "off_topic", None)

    # Scoring questions are grounded in calculate_all() values (code-verified),
    # not guideline chunks — exempt from the retrieval threshold and citation gate.
    is_scoring = intent == "scoring"

    top_score = max((c.get("score", 0.0) for c in chunks), default=0.0)
    if not is_scoring and (not chunks or top_score < threshold):
        return _response(FALLBACK_TEXT, [], chunks, alerts, True,
                         f"retrieval_below_threshold (top={top_score:.2f} < {threshold})",
                         top_score)

    summary = summarize_patient(patient_context, calc)
    messages = build_messages(query, chunks, summary, format_alerts(alerts),
                              intent=intent)
    try:
        reply = chat.chat(messages, temperature=0.1, max_tokens=900)
        data = parse_json_loose(reply)
    except Exception as err:
        return _response(FALLBACK_TEXT, [], chunks, alerts, True,
                         f"generation_error: {err}", top_score)

    answer = (data.get("answer") or "").strip()
    # Scoring answers are grounded in calculate_all() — the "insufficient" flag
    # (about guideline coverage) doesn't apply as long as the model produced text.
    insufficient = (data.get("insufficient") and not is_scoring)
    if insufficient or not answer:
        return _response(FALLBACK_TEXT, [], chunks, alerts, True,
                         "llm_insufficient", top_score)

    citations = _extract_citations(data, answer, len(chunks))
    if not citations and not is_scoring:
        # Hallucination guard: an uncited clinical answer is never shown.
        return _response(FALLBACK_TEXT, [], chunks, alerts, True,
                         "no_valid_citations", top_score)

    conf = data.get("confidence")
    conf = float(conf) if isinstance(conf, (int, float)) else None
    return _response(answer, citations, chunks, alerts, False, None, conf)
