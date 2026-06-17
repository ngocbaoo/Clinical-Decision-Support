"""
R4 — Generator: grounded answer with enforced citations + verifier + fallback.

The LLM now returns T1-structured sentences {text, evidence, citation}; the code (not the
LLM) enforces the safety contract:
  - off-topic intent           -> refusal, no LLM call
  - retrieval top score < thr  -> fallback (F-RAG-09)
  - LLM says insufficient      -> fallback
  - claim-level verifier       -> strip / fallback per the safety decision tree (R4b)
  - no valid [n] citations     -> fallback (hallucination guard)
Alerts from the safety gate are prepended to whatever is returned.

A legacy {answer, citations} reply (no "sentences") is still accepted and skips the
verifier — keeps older callers / offline tests working.
"""

import re
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ on sys.path
from embedding.or_client import ChatClient  # noqa: E402
from rag.config import CONF_THRESHOLD, DISCLAIMER, VERIFIER_BACKEND  # noqa: E402
from rag.context_builder import build_messages, summarize_patient  # noqa: E402
from rag.query_router import parse_json_loose  # noqa: E402
from rag.safety import format_alerts  # noqa: E402
from rag.verifier import _is_effective_safety, verify_answer  # noqa: E402

FALLBACK_TEXT = ("Không đủ thông tin trong cơ sở tri thức để đưa ra khuyến nghị "
                 "đáng tin cậy cho câu hỏi này. Vui lòng tra cứu guideline gốc "
                 "hoặc hỏi dược sĩ lâm sàng.")
OFF_TOPIC_TEXT = ("Câu hỏi nằm ngoài phạm vi hỗ trợ lâm sàng ICU — "
                  "tôi chỉ trả lời các câu hỏi y khoa dựa trên guideline.")
UNVERIFIED_BANNER = "⚠️ Câu trả lời CHƯA được xác minh tự động (verifier không khả dụng)."


def _response(answer: str, citations: list[int], chunks: list[dict],
              alerts: list[dict], fallback: bool, reason: str | None,
              confidence: float | None, verify: dict | None = None,
              prefix: str | None = None) -> dict:
    cited_sources = [
        {"n": n, "source": chunks[n - 1]["source"], "title": chunks[n - 1]["title"]}
        for n in citations if 1 <= n <= len(chunks)
    ]
    alert_text = format_alerts(alerts)
    full = (alert_text + "\n\n" if alert_text else "") + answer
    if prefix:
        full = f"{prefix}\n\n{full}"
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
        "verify": verify,
    }


def _parse_claims(data: dict) -> tuple[list[dict] | None, str]:
    """Return (claims, mode). mode is "t1" (structured) or "legacy" (answer+citations)."""
    sentences = data.get("sentences")
    if isinstance(sentences, list) and sentences:
        claims = []
        for s in sentences:
            if not isinstance(s, dict):
                continue
            text = (s.get("text") or "").strip()
            if not text:
                continue
            cit = s.get("citation")
            cit = int(cit) if isinstance(cit, (int, float)) else None
            claims.append({"text": text, "evidence": (s.get("evidence") or "").strip(),
                           "citation": cit})
        return claims, "t1"
    return None, "legacy"


def _assemble(claims: list[dict], n_chunks: int) -> tuple[str, list[int]]:
    """Render kept claims as text with [n] markers; collect valid citations."""
    parts, cites = [], set()
    for c in claims:
        cit = c.get("citation")
        text = c["text"]
        if isinstance(cit, int) and 1 <= cit <= n_chunks:
            cites.add(cit)
            if f"[{cit}]" not in text:
                text = f"{text} [{cit}]"
        parts.append(text)
    return " ".join(parts), sorted(cites)


def _extract_citations(data: dict, answer_text: str, n_chunks: int) -> list[int]:
    """Legacy: citations from the JSON field + inline [n] markers; validated."""
    cites = set()
    for c in data.get("citations", []) or []:
        if isinstance(c, (int, float)):
            cites.add(int(c))
    for m in re.finditer(r"\[(\d{1,2})\]", answer_text):
        cites.add(int(m.group(1)))
    return sorted(c for c in cites if 1 <= c <= n_chunks)


def generate(query: str, intent: str, chunks: list[dict], patient_context: dict,
             calc: dict, alerts: list[dict], chat: ChatClient,
             threshold: float = CONF_THRESHOLD, *, verifier_chat: ChatClient | None = None,
             backend: str = VERIFIER_BACKEND, nli=None) -> dict:
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
    # The model occasionally emits malformed JSON (more so with reasoning disabled). One
    # retry is now cheap (~3s without reasoning vs ~24s with) and recovers these instead of
    # falling back. max_tokens is generous so a long procedure's JSON can't truncate.
    data, last_err = None, None
    for _attempt in range(2):
        try:
            reply = chat.chat(messages, temperature=0.1, max_tokens=1500)
            data = parse_json_loose(reply)
            break
        except Exception as err:
            last_err = err
    if data is None:
        return _response(FALLBACK_TEXT, [], chunks, alerts, True,
                         f"generation_error: {last_err}", top_score)

    insufficient = (data.get("insufficient") and not is_scoring)
    claims, mode = _parse_claims(data)

    # ---- legacy {answer, citations} path (no per-claim evidence -> no verifier) ----
    if mode == "legacy":
        answer = (data.get("answer") or "").strip()
        if insufficient or not answer:
            return _response(FALLBACK_TEXT, [], chunks, alerts, True,
                             "llm_insufficient", top_score)
        citations = _extract_citations(data, answer, len(chunks))
        if not citations and not is_scoring:
            return _response(FALLBACK_TEXT, [], chunks, alerts, True,
                             "no_valid_citations", top_score)
        conf = data.get("confidence")
        conf = float(conf) if isinstance(conf, (int, float)) else None
        return _response(answer, citations, chunks, alerts, False, None, conf)

    # ---- T1 structured path ----
    if insufficient or not claims:
        return _response(FALLBACK_TEXT, [], chunks, alerts, True,
                         "llm_insufficient", top_score)

    verify_meta = None
    run_verify = (verifier_chat is not None or backend == "local_nli") and not is_scoring
    if run_verify:
        _t = time.perf_counter()
        result = verify_answer(claims, chunks, summary, intent, backend=backend,
                               verifier_chat=verifier_chat, nli=nli)
        verify_meta = {k: result.get(k) for k in
                       ("status", "branch", "unsupported_ratio", "contradiction_count",
                        "is_ordered_procedure", "fallback_reason")}
        verify_meta["elapsed_s"] = round(time.perf_counter() - _t, 2)
        verify_meta["backend"] = backend
        # Per-claim audit trail (text/citation/verdict/safety from decide(), joined with the
        # model's self-declared evidence quote) — needed to human-review fallbacks.
        verify_meta["verdicts"] = [
            {**v, "evidence": (claims[i].get("evidence") or "")[:200]}
            for i, v in enumerate(result.get("verdicts") or [])
        ]
        if result["status"] == "error":
            # Fail-CLOSED for safety-critical answers; fail-OPEN (banner) otherwise.
            safety_critical = intent == "contraindication" or bool(alerts) or any(
                _is_effective_safety(c["text"], False, intent) for c in claims)
            if safety_critical:
                verify_meta["status"] = "error_failclosed"
                return _response(FALLBACK_TEXT, [], chunks, alerts, True,
                                 "verifier_unavailable_safety", top_score,
                                 verify=verify_meta)
            verify_meta["status"] = "unverified"
            answer, citations = _assemble(claims, len(chunks))
            return _response(answer, citations, chunks, alerts, False, None, None,
                             verify=verify_meta, prefix=UNVERIFIED_BANNER)
        if result["action"] == "fallback":
            return _response(FALLBACK_TEXT, [], chunks, alerts, True,
                             result["fallback_reason"], top_score, verify=verify_meta)
        claims = [{"text": c["text"], "citation": c["citation"]}
                  for c in result["kept_claims"]]

    answer, citations = _assemble(claims, len(chunks))
    if not answer:
        return _response(FALLBACK_TEXT, [], chunks, alerts, True,
                         "verifier_unsupported", top_score, verify=verify_meta)
    if not citations and not is_scoring:
        return _response(FALLBACK_TEXT, [], chunks, alerts, True,
                         "no_valid_citations", top_score, verify=verify_meta)
    conf = data.get("confidence")
    conf = float(conf) if isinstance(conf, (int, float)) else None
    return _response(answer, citations, chunks, alerts, False, None, conf,
                     verify=verify_meta)
