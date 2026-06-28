"""Offline unit tests for the RAG module (no network, LLM mocked)."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.generator import generate, FALLBACK_TEXT, OFF_TOPIC_TEXT  # noqa: E402
from rag.query_router import keyword_route, parse_json_loose  # noqa: E402
from rag.safety import (check_allergies, check_contraindications,  # noqa: E402
                        check_drug_interactions, format_alerts)
from rag import safety as safety_mod  # noqa: E402
from rag.openfda import find_mention  # noqa: E402
from rag.context_builder import build_messages, summarize_patient  # noqa: E402
from rag.config import DISCLAIMER  # noqa: E402
from rag.verifier import (decide, verify_answer, _is_effective_safety,  # noqa: E402
                          _evidence_matches, _evidence_grounded)
from rag.logging_utils import log_request  # noqa: E402
from rag.fusion import (rrf_fuse, comorbidity_fuse,  # noqa: E402
                        comorbidity_names, comorbidity_queries)
from rag.reranker import order_by_rerank  # noqa: E402  (pure helper; torch only in the class)
from rag.comorbidity_gate import (check_comorbidity_conflicts,  # noqa: E402
                                  apply_comorbidity_gate)


def _chunks(scores=(0.7, 0.6)):
    return [{"text": f"chunk {i}", "title": f"T{i}", "source": "S",
             "chunk_type": "procedure", "score": s}
            for i, s in enumerate(scores, start=1)]


def _patient_with_allergy(allergen="Penicillin"):
    return {
        "patient": {"name": "X", "gender": "male", "age": 60},
        "allergies": [{"allergen": allergen, "criticality": "high",
                       "reaction": "anaphylaxis", "category": "medication"}],
        "observations": {}, "medications": [], "conditions": [],
        "medication_administrations": [], "procedures": [],
        "diagnostic_reports": [], "missing_resources": [],
    }


# ---- safety gate ----------------------------------------------------------
def test_allergy_direct_match():
    alerts = check_allergies(["Penicillin"], _patient_with_allergy())
    assert len(alerts) == 1 and alerts[0]["match"] == "direct"


def test_allergy_cross_reactivity_penicillin_amoxicillin():
    alerts = check_allergies(["Amoxicillin"], _patient_with_allergy("Penicillin"))
    assert len(alerts) == 1
    assert "cross-reactivity" in alerts[0]["match"]


def test_allergy_no_false_positive():
    alerts = check_allergies(["Vancomycin"], _patient_with_allergy("Penicillin"))
    assert alerts == []


def test_allergy_empty_context():
    assert check_allergies(["Amoxicillin"], {}) == []


def test_format_alerts_leads_with_warning():
    alerts = check_allergies(["Amoxicillin"], _patient_with_allergy())
    text = format_alerts(alerts)
    assert text.startswith("⚠️")
    assert "Amoxicillin" in text and "Penicillin" in text


# ---- drug interactions (OpenFDA, network mocked) --------------------------
_WARFARIN_LABEL = ("Concomitant use of Warfarin and Aspirin increases the risk "
                   "of bleeding. Avoid coadministration with NSAIDs.")


def _patient_on(*meds):
    return {"medications": [{"name": m, "dose": ""} for m in meds]}


def test_interaction_query_drug_vs_patient_med(monkeypatch):
    # Query asks about Aspirin; patient is on Warfarin whose label names Aspirin.
    monkeypatch.setattr(safety_mod, "get_interaction_text",
                        lambda d: [_WARFARIN_LABEL] if "aspirin" in d.lower() else [])
    alerts = check_drug_interactions(["Aspirin"], _patient_on("Warfarin"))
    assert len(alerts) == 1
    a = alerts[0]
    assert a["type"] == "interaction" and a["drug_b"] == "Warfarin"
    assert "bleeding" in a["snippet"].lower()


def test_interaction_deduplicates_pairs(monkeypatch):
    # Both drugs queried; each label names the other -> still one alert.
    monkeypatch.setattr(safety_mod, "get_interaction_text",
                        lambda d: [_WARFARIN_LABEL])
    alerts = check_drug_interactions(["Warfarin", "Aspirin"], {})
    assert len(alerts) == 1


def test_interaction_none_when_no_drugs(monkeypatch):
    monkeypatch.setattr(safety_mod, "get_interaction_text",
                        lambda d: [_WARFARIN_LABEL])
    assert check_drug_interactions([], _patient_on("Warfarin")) == []


def test_interaction_degrades_on_empty_label(monkeypatch):
    monkeypatch.setattr(safety_mod, "get_interaction_text", lambda d: [])
    assert check_drug_interactions(["Aspirin"], _patient_on("Warfarin")) == []


def test_find_mention_whole_word_only():
    assert find_mention("Avoid use with Aspirin.", "Aspirin") is not None
    # 'pin' must not match inside 'Aspirin'
    assert find_mention("Avoid use with Aspirin.", "pin") is None


def test_format_alerts_renders_interaction_block():
    alerts = [{"type": "interaction", "drug_a": "Aspirin", "drug_b": "Warfarin",
               "other_source": "thuốc kê đơn", "snippet": "bleeding risk"}]
    text = format_alerts(alerts)
    assert "TƯƠNG TÁC THUỐC" in text and "Aspirin ⇄ Warfarin" in text


def test_format_alerts_allergy_first_then_interaction():
    alerts = check_allergies(["Amoxicillin"], _patient_with_allergy())
    alerts += [{"type": "interaction", "drug_a": "A", "drug_b": "B",
                "other_source": "câu hỏi", "snippet": "x"}]
    text = format_alerts(alerts)
    assert text.index("DỊ ỨNG") < text.index("TƯƠNG TÁC")


# ---- contraindications (OpenFDA, network mocked) --------------------------
_WARFARIN_CONTRA = ("Warfarin sodium is contraindicated in: Pregnancy. "
                    "Warfarin can cause fetal harm. Also contraindicated in "
                    "patients with recent surgery or active bleeding.")


def _patient_with_conditions(*name_en):
    return {"conditions": [{"name_vi": n, "name_en": n, "display": n}
                           for n in name_en]}


def test_contra_matches_condition(monkeypatch):
    monkeypatch.setattr(safety_mod, "get_contraindication_text",
                        lambda d: [_WARFARIN_CONTRA])
    patient = _patient_with_conditions("Active bleeding")
    alerts = check_contraindications(["Warfarin"], patient)
    assert len(alerts) == 1
    assert alerts[0]["type"] == "contraindication"
    assert "bleeding" in alerts[0]["snippet"].lower()


def test_contra_pregnancy_stem_match(monkeypatch):
    monkeypatch.setattr(safety_mod, "get_contraindication_text",
                        lambda d: [_WARFARIN_CONTRA])
    # Condition recorded in Vietnamese ("thai") -> pregnancy stem trigger,
    # matches FDA "Pregnancy".
    patient = {"conditions": [{"name_vi": "Mang thai", "name_en": ""}]}
    alerts = check_contraindications(["Warfarin"], patient)
    assert any(a["condition"] == "Thai kỳ" for a in alerts)


def test_contra_no_match_unrelated_condition(monkeypatch):
    monkeypatch.setattr(safety_mod, "get_contraindication_text",
                        lambda d: [_WARFARIN_CONTRA])
    patient = _patient_with_conditions("Pneumonia")
    assert check_contraindications(["Warfarin"], patient) == []


def test_contra_none_when_no_conditions(monkeypatch):
    monkeypatch.setattr(safety_mod, "get_contraindication_text",
                        lambda d: [_WARFARIN_CONTRA])
    assert check_contraindications(["Warfarin"], {}) == []


def test_contra_degrades_on_empty_label(monkeypatch):
    monkeypatch.setattr(safety_mod, "get_contraindication_text", lambda d: [])
    patient = _patient_with_conditions("Active bleeding")
    assert check_contraindications(["Warfarin"], patient) == []


def test_format_alerts_orders_allergy_contra_interaction():
    alerts = (check_allergies(["Amoxicillin"], _patient_with_allergy())
              + [{"type": "contraindication", "drug": "W", "condition": "Thai kỳ",
                  "matched": "pregnan", "snippet": "x"}]
              + [{"type": "interaction", "drug_a": "A", "drug_b": "B",
                  "other_source": "câu hỏi", "snippet": "y"}])
    text = format_alerts(alerts)
    assert text.index("DỊ ỨNG") < text.index("CHỐNG CHỈ ĐỊNH") < text.index("TƯƠNG TÁC")


# ---- router fallback / JSON parsing ---------------------------------------
def test_keyword_route_safety():
    assert keyword_route("chống chỉ định lọc máu")["intent"] == "contraindication"
    assert keyword_route("liều vancomycin bao nhiêu")["intent"] == "dosing"
    assert keyword_route("NEWS2 bao nhiêu điểm")["intent"] == "scoring"
    assert keyword_route("quy trình đặt catheter")["intent"] == "general"


def test_parse_json_loose_fenced():
    out = parse_json_loose('Here:\n```json\n{"a": 1}\n```\ndone')
    assert out == {"a": 1}


def test_parse_json_loose_prose_wrapped():
    out = parse_json_loose('blah {"intent": "procedure", "drugs": []} blah')
    assert out["intent"] == "procedure"


# ---- generator contract ----------------------------------------------------
def _mock_chat(reply: str):
    chat = MagicMock()
    chat.chat.return_value = reply
    return chat


def test_generate_off_topic_no_llm_call():
    chat = _mock_chat("")
    resp = generate("q", "off_topic", [], {}, {}, [], chat)
    assert resp["fallback"] and OFF_TOPIC_TEXT in resp["answer"]
    chat.chat.assert_not_called()


def test_generate_below_threshold_falls_back():
    chat = _mock_chat("")
    resp = generate("q", "general", _chunks(scores=(0.2,)), {}, {}, [], chat,
                    threshold=0.45)
    assert resp["fallback"] and FALLBACK_TEXT in resp["answer"]
    chat.chat.assert_not_called()
    # Regression: the below-threshold branch must put the score in `confidence`, not
    # `verify` — a float in `verify` crashes pipeline.ask (response["verify"].get(...)).
    assert resp["verify"] is None
    assert resp["confidence"] == 0.2


def test_generate_valid_answer_with_citations():
    chat = _mock_chat('{"answer": "Khuyến nghị X [1]", "citations": [1], '
                      '"confidence": 0.8, "insufficient": false}')
    resp = generate("q", "procedure", _chunks(), {}, {}, [], chat)
    assert not resp["fallback"]
    assert resp["citations"] == [1]
    assert DISCLAIMER in resp["answer"]


def test_generate_uncited_answer_is_blocked():
    chat = _mock_chat('{"answer": "Khuyến nghị không nguồn", "citations": [], '
                      '"insufficient": false}')
    resp = generate("q", "procedure", _chunks(), {}, {}, [], chat)
    assert resp["fallback"] and resp["fallback_reason"] == "no_valid_citations"


def test_generate_out_of_range_citation_is_blocked():
    chat = _mock_chat('{"answer": "Xem [9]", "citations": [9], '
                      '"insufficient": false}')
    resp = generate("q", "procedure", _chunks(), {}, {}, [], chat)
    assert resp["fallback"] and resp["fallback_reason"] == "no_valid_citations"


def test_generate_llm_insufficient_falls_back():
    chat = _mock_chat('{"answer": "", "citations": [], "insufficient": true}')
    resp = generate("q", "procedure", _chunks(), {}, {}, [], chat)
    assert resp["fallback"] and resp["fallback_reason"] == "llm_insufficient"


def test_generate_scoring_intent_allows_uncited():
    chat = _mock_chat('{"answer": "NEWS2 = 9, nguy cơ cao", "citations": [], '
                      '"insufficient": false}')
    resp = generate("q", "scoring", [], {}, {}, [], chat)
    assert not resp["fallback"]


def test_generate_alerts_prepended():
    alerts = check_allergies(["Amoxicillin"], _patient_with_allergy())
    chat = _mock_chat('{"answer": "Không dùng [1]", "citations": [1], '
                      '"insufficient": false}')
    resp = generate("q", "contraindication", _chunks(), _patient_with_allergy(),
                    {}, alerts, chat)
    assert resp["answer"].lstrip().startswith("⚠️")


def test_generate_garbage_reply_falls_back():
    chat = _mock_chat("I cannot answer that.")
    resp = generate("q", "procedure", _chunks(), {}, {}, [], chat)
    assert resp["fallback"] and "generation_error" in resp["fallback_reason"]


# ---- verifier: decision tree (pure) ---------------------------------------
def _cl(text, citation=1):
    return {"text": text, "citation": citation, "evidence": ""}


def _v(verdict, is_safety=False):
    return {"verdict": verdict, "is_safety": is_safety}


def test_decide_supported_kept():
    d = decide([_cl("Bù dịch tinh thể")], [_v("supported")], "procedure")
    assert d["action"] == "keep" and len(d["kept_claims"]) == 1


def test_decide_neutral_ordinary_stripped():
    d = decide([_cl("Bù dịch tinh thể"), _cl("Thêm vitamin C")],
               [_v("supported"), _v("neutral")], "procedure")
    assert d["action"] == "keep" and len(d["kept_claims"]) == 1
    assert d["unsupported_ratio"] == 0.5


def test_decide_all_neutral_falls_back():
    d = decide([_cl("Thêm vitamin C")], [_v("neutral")], "procedure")
    assert d["action"] == "fallback" and d["fallback_reason"] == "verifier_unsupported"


def test_decide_contradicted_falls_back_whole_answer():
    # one contradiction taints the whole answer even though another is supported
    d = decide([_cl("Bù dịch [1]"), _cl("Không dùng noradrenalin [1]")],
               [_v("supported"), _v("contradicted")], "procedure")
    assert d["action"] == "fallback" and d["branch"] == "contradicted"
    assert d["fallback_reason"] == "verifier_contradicted"


def test_decide_unsupported_safety_falls_back():
    d = decide([_cl("Bù dịch [1]"), _cl("Chống chỉ định dùng thuốc X [1]")],
               [_v("supported"), _v("neutral")], "procedure")
    assert d["action"] == "fallback" and d["fallback_reason"] == "verifier_unsupported_safety"


def test_decide_safety_via_verifier_flag():
    d = decide([_cl("Theo dõi sát bệnh nhân")], [_v("neutral", is_safety=True)], "procedure")
    assert d["fallback_reason"] == "verifier_unsupported_safety"


def test_decide_intent_contraindication_requires_all_supported():
    d = decide([_cl("Có thể dùng thuốc")], [_v("neutral")], "contraindication")
    assert d["action"] == "fallback" and d["fallback_reason"] == "verifier_unsupported_safety"


def test_decide_ordered_procedure_integrity_break():
    d = decide([_cl("Bước 1 đặt tư thế"), _cl("Bước 2 hút đờm")],
               [_v("supported"), _v("neutral")], "procedure")
    assert d["action"] == "fallback" and d["fallback_reason"] == "verifier_integrity_break"


def test_decide_ordered_procedure_all_supported_kept():
    d = decide([_cl("Bước 1 đặt tư thế"), _cl("Bước 2 hút đờm")],
               [_v("supported"), _v("supported")], "procedure")
    assert d["action"] == "keep" and len(d["kept_claims"]) == 2


def test_effective_safety_keyword_backstop():
    assert _is_effective_safety("Chống chỉ định dùng X", False, "procedure")
    assert _is_effective_safety("bất kỳ câu nào", False, "contraindication")
    assert not _is_effective_safety("Bù dịch tinh thể", False, "procedure")


# ---- verifier: fast-path + backend ----------------------------------------
def test_evidence_matches_fold():
    assert _evidence_matches("phải đặt nội khí quản", "... PHẢI  đặt nội khí quản ngay ...")
    assert not _evidence_matches("xyz", "abc def")


def test_verify_fast_path_skips_backend():
    chat = MagicMock()  # must NOT be called — evidence is literally in the chunk
    claims = [{"text": "Đặt nội khí quản [1]", "evidence": "phải đặt nội khí quản",
               "citation": 1}]
    chunks = [{"text": "Bệnh nhân phải đặt nội khí quản ngay", "title": "T", "source": "S",
               "chunk_type": "procedure", "score": 0.8}]
    res = verify_answer(claims, chunks, "", "procedure", backend="llm", verifier_chat=chat)
    assert res["status"] == "ok" and res["action"] == "keep"
    chat.chat.assert_not_called()


def test_evidence_grounded_fuzzy():
    chunk = "Khuyến cáo bù dịch tinh thể 30ml/kg trong giờ đầu, MAP mục tiêu >= 65 mmHg."
    assert _evidence_grounded("bù dịch tinh thể 30ml/kg", chunk)
    assert _evidence_grounded("MAP mục tiêu ≥ 65 mmHg", chunk)        # unicode >= , diacritics
    assert not _evidence_grounded("dự phòng xuất huyết bằng kháng H2", chunk)  # fabricated quote


def test_verify_ungrounded_claim_dropped_by_code():
    # Lever 1: evidence NOT in the cited chunk -> "neutral" by code, NEVER sent to the backend.
    chat = MagicMock()  # must NOT be called
    claims = [{"text": "Khuyến nghị X [1]", "evidence": "không có trong chunk này", "citation": 1}]
    chunks = [{"text": "nội dung hoàn toàn khác", "title": "T", "source": "S",
               "chunk_type": "procedure", "score": 0.8}]
    res = verify_answer(claims, chunks, "", "procedure", backend="llm", verifier_chat=chat)
    assert res["action"] == "fallback"  # only claim dropped -> nothing kept
    chat.chat.assert_not_called()


def test_verify_backend_error_returns_status_error():
    chat = MagicMock()
    chat.chat.side_effect = RuntimeError("api down")
    # a no-citation (patient-data) claim is the path that still reaches the LLM backend
    claims = [{"text": "MAP 60 mmHg", "evidence": "", "citation": None}]
    chunks = [{"text": "khác", "title": "T", "source": "S", "chunk_type": "procedure",
               "score": 0.8}]
    res = verify_answer(claims, chunks, "MAP = 60", "procedure", backend="llm", verifier_chat=chat)
    assert res["status"] == "error"


# ---- generator T1 + verifier integration ----------------------------------
# evidence IS in the chunk (Lever 1 passes); the verdict then comes from the Lever-2 NLI gate.
_T1_REPLY = ('{"sentences": [{"text": "Bù dịch tinh thể 30ml/kg", '
             '"evidence": "bù dịch tinh thể 30ml/kg", "citation": 1}], '
             '"confidence": 0.8, "insufficient": false}')
# a no-citation (patient-data) claim -> the only path that still reaches the LLM/NLI backend.
_T1_REPLY_NOCITE = ('{"sentences": [{"text": "MAP hiện tại là 60 mmHg", '
                    '"evidence": "", "citation": null}], '
                    '"confidence": 0.8, "insufficient": false}')


def _t1_chunks():
    return [{"text": "Khuyến cáo bù dịch tinh thể 30ml/kg trong giờ đầu", "title": "T1",
             "source": "S", "chunk_type": "procedure", "score": 0.7}]


def test_generate_t1_verified_answer():
    # grounded evidence + no NLI passed -> Lever-1 keeps it (supported); backend not needed
    gen = _mock_chat(_T1_REPLY)
    ver = MagicMock()
    resp = generate("q", "procedure", _t1_chunks(), {}, {}, [], gen, verifier_chat=ver, backend="llm")
    assert not resp["fallback"]
    assert "[1]" in resp["answer"] and resp["citations"] == [1]
    assert resp["verify"]["status"] == "ok"


def test_generate_t1_fast_path_supported_without_verifier_call():
    # grounded evidence -> kept by code, the LLM verifier is never called
    gen = _mock_chat(_T1_REPLY)
    ver = MagicMock()
    resp = generate("q", "procedure", _t1_chunks(), {}, {}, [], gen, verifier_chat=ver, backend="llm")
    assert not resp["fallback"] and resp["citations"] == [1]
    ver.chat.assert_not_called()


def test_generate_t1_contradiction_falls_back():
    # Lever 2: the NLI judges the grounded claim against its evidence span -> contradiction
    gen = _mock_chat(_T1_REPLY)
    ver = MagicMock()
    resp = generate("q", "procedure", _t1_chunks(), {}, {}, [], gen, verifier_chat=ver,
                    backend="llm", nli=lambda p, h: ("contradiction", 0.95))
    assert resp["fallback"] and resp["fallback_reason"] == "verifier_contradicted"


def test_generate_lever2_evidence_span_is_premise():
    # Lever 2 feeds (premise = the claim's EVIDENCE span, hypothesis = claim) to the nli callable —
    # NOT the whole chunk (that tight premise is what makes the entailment check crisp).
    gen = _mock_chat(_T1_REPLY)
    seen = []

    def fake_nli(premise, hypothesis):
        seen.append((premise, hypothesis))
        return "entailment", 0.95  # -> supported

    resp = generate("q", "procedure", _t1_chunks(), {}, {}, [], gen,
                    backend="local_nli", nli=fake_nli)
    assert seen, "nli was never called"
    assert seen[0][0] == "bù dịch tinh thể 30ml/kg"  # premise is the evidence span, not the chunk
    assert not resp["fallback"] and resp["citations"] == [1]


def test_generate_lever2_low_confidence_neutral_kept():
    # confidence gate: a LOW-confidence non-entailment keeps the claim (mDeBERTa over-rejects VN)
    gen = _mock_chat(_T1_REPLY)
    resp = generate("q", "procedure", _t1_chunks(), {}, {}, [], gen,
                    backend="local_nli", nli=lambda p, h: ("neutral", 0.5))  # below NLI_REJECT_CONF
    assert not resp["fallback"] and resp["citations"] == [1]


def test_generate_lever2_high_confidence_neutral_dropped():
    # confidence gate: a HIGH-confidence neutral (over-claim) is dropped -> fallback
    gen = _mock_chat(_T1_REPLY)
    resp = generate("q", "procedure", _t1_chunks(), {}, {}, [], gen,
                    backend="local_nli", nli=lambda p, h: ("neutral", 0.95))  # >= NLI_REJECT_CONF
    assert resp["fallback"]


def test_generate_local_nli_contradiction_falls_back():
    gen = _mock_chat(_T1_REPLY)
    resp = generate("q", "procedure", _t1_chunks(), {}, {}, [], gen,
                    backend="local_nli", nli=lambda p, h: ("contradiction", 0.9))
    assert resp["fallback"] and resp["fallback_reason"] == "verifier_contradicted"


def test_generate_t1_fail_closed_for_contraindication():
    # no-citation claim reaches the backend; backend down + safety intent -> fail CLOSED
    gen = _mock_chat(_T1_REPLY_NOCITE)
    ver = MagicMock()
    ver.chat.side_effect = RuntimeError("verifier down")
    resp = generate("q", "contraindication", _t1_chunks(), {}, {}, [], gen,
                    verifier_chat=ver, backend="llm")
    assert resp["fallback"] and resp["fallback_reason"] == "verifier_unavailable_safety"


def test_generate_t1_fail_open_with_banner():
    # no-citation claim reaches the backend; backend down + non-safety intent -> fail OPEN (banner)
    gen = _mock_chat(_T1_REPLY_NOCITE)
    ver = MagicMock()
    ver.chat.side_effect = RuntimeError("verifier down")
    resp = generate("q", "procedure", _t1_chunks(), {}, {}, [], gen,
                    verifier_chat=ver, backend="llm")
    assert not resp["fallback"]
    assert "CHƯA được xác minh" in resp["answer"]
    assert resp["verify"]["status"] == "unverified"


def test_pipeline_runs_retrieval_and_safety_in_parallel(monkeypatch):
    # retrieval and safety each "take" 0.4s; run concurrently the stage wall-clock must be ~0.4s,
    # well under the 0.8s serial sum. Stubs everything LLM/network so this stays offline.
    import time as _t

    import rag.pipeline as P
    from rag.pipeline import RAGPipeline

    pipe = RAGPipeline.__new__(RAGPipeline)  # bypass __init__ (no chroma / no network)
    pipe.chat = None
    pipe.verify = False
    pipe.backend = "llm"
    pipe.gen_model = "x"
    pipe._nli = None

    monkeypatch.setattr(P, "route", lambda q, chat: {
        "intent": "general", "drugs": ["aspirin"], "procedures": [], "via": "stub"})
    monkeypatch.setattr(pipe, "retrieve_for_intent",
                        lambda q, i, comorbidities=None: (_t.sleep(0.4) or [{"text": "c",
                                      "source": "s", "title": "t", "score": 0.9,
                                      "chunk_type": "x"}]))
    monkeypatch.setattr(P, "check_allergies", lambda d, c: (_t.sleep(0.4) or []))
    monkeypatch.setattr(P, "check_contraindications", lambda d, c: [])
    monkeypatch.setattr(P, "check_drug_interactions", lambda d, c: [])
    monkeypatch.setattr(P, "generate", lambda *a, **k: {
        "answer": "x", "citations": [], "cited_sources": [], "alerts": [], "fallback": False,
        "fallback_reason": None, "confidence": None, "verify": None})
    monkeypatch.setattr(pipe, "_get_nli", lambda: None)
    monkeypatch.setattr(pipe, "_log", lambda *a, **k: None)

    res = pipe.ask("q", {}, {})
    assert res["timings_s"]["total"] < 0.7  # parallel (~0.4), not serial (~0.8)
    assert res["timings_s"]["retrieval"] >= 0.4 and res["timings_s"]["safety"] >= 0.4


# ---- logging ----------------------------------------------------------------
def test_log_request_writes_jsonl(tmp_path):
    import json as _json
    log_request({"request_id": "abc", "query": "q"}, log_dir=tmp_path)
    files = list(tmp_path.glob("rag-*.jsonl"))
    assert len(files) == 1
    rec = _json.loads(files[0].read_text(encoding="utf-8").strip())
    assert rec["request_id"] == "abc" and "ts" in rec


# ---- context builder --------------------------------------------------------
def test_build_messages_numbers_chunks_and_leads_alerts():
    msgs = build_messages("câu hỏi", _chunks(), "tóm tắt BN", "⚠️ CẢNH BÁO")
    user = msgs[1]["content"]
    assert user.index("CẢNH BÁO") < user.index("TÀI LIỆU")
    assert "[1] (S — T1)" in user and "[2] (S — T2)" in user


def test_summarize_patient_lists_missing_obs():
    ctx = _patient_with_allergy()
    ctx["observations"] = {"spo2": {"value": None}, "heart_rate": {"value": 80, "unit": "bpm"}}
    text = summarize_patient(ctx, {})
    assert "THIẾU DỮ LIỆU" in text and "spo2" in text
    assert "heart_rate=80" in text


def test_summarize_patient_no_context():
    assert "guideline chung" in summarize_patient({}, {})


# --- Comorbidity-aware retrieval / RRF (src/rag/fusion.py) --------------------

def _doc(t, score=0.0):
    return {"text": t, "chunk_type": "procedure", "score": score}


def test_rrf_single_list_is_identity_order():
    lst = [_doc("a"), _doc("b"), _doc("c")]
    assert [d["text"] for d in rrf_fuse([lst])] == ["a", "b", "c"]


def test_rrf_dedup_accumulates_across_lists():
    # 'shared' appears in both lists -> its summed RRF score beats singletons -> ranks first.
    primary = [_doc("p1"), _doc("shared")]
    aux = [_doc("shared"), _doc("x")]
    fused = [d["text"] for d in rrf_fuse([primary, aux])]
    assert fused[0] == "shared"
    # the original primary dict is kept (first occurrence), not the aux copy
    assert rrf_fuse([primary, aux])[0] is primary[1]


def test_comorbidity_fuse_keeps_full_primary_and_appends():
    # ABSOLUTE recall guarantee: the entire primary top-K is preserved; comorbidity is APPENDED.
    primary = [_doc("p1", 0.8), _doc("p2", 0.7), _doc("p3", 0.6), _doc("p4_tail", 0.5)]
    aux = [[_doc("comorbid", 0.7)]]
    out = [d["text"] for d in comorbidity_fuse(primary, aux, n_results=4, comorbidity_slots=1)]
    assert out[:4] == ["p1", "p2", "p3", "p4_tail"]   # nothing evicted, even the rank-4 tail
    assert out[4] == "comorbid"                       # comorbidity appended as the +1 chunk


def test_comorbidity_fuse_admits_best_comorbidity_by_rrf():
    # Across two comorbidity lists, the chunk ranked high in BOTH is the one appended.
    primary = [_doc("p1", 0.8), _doc("p2", 0.7)]
    aux = [[_doc("shared", 0.7), _doc("a_only", 0.6)],
           [_doc("shared", 0.7), _doc("b_only", 0.6)]]
    out = [d["text"] for d in comorbidity_fuse(primary, aux, n_results=2, comorbidity_slots=1)]
    assert out == ["p1", "p2", "shared"]


def test_comorbidity_fuse_skips_irrelevant_and_degrades_to_baseline():
    # If no aux chunk clears min_score, nothing is appended -> exactly baseline top-K.
    primary = [_doc("p1", 0.8), _doc("p2", 0.7), _doc("p3", 0.6)]
    aux = [[_doc("weak_comorbid", 0.20)]]
    out = [d["text"] for d in comorbidity_fuse(primary, aux, n_results=3,
                                               comorbidity_slots=1, min_score=0.45)]
    assert out == ["p1", "p2", "p3"]


def test_comorbidity_fuse_does_not_duplicate_primary():
    # A comorbidity chunk already in primary is not appended again.
    primary = [_doc("p1", 0.8), _doc("shared", 0.7)]
    aux = [[_doc("shared", 0.9)]]
    out = [d["text"] for d in comorbidity_fuse(primary, aux, n_results=2, comorbidity_slots=1)]
    assert out == ["p1", "shared"]


def test_comorbidity_fuse_no_aux_is_identity():
    primary = [_doc("p1"), _doc("p2"), _doc("p3")]
    assert comorbidity_fuse(primary, [], n_results=2, comorbidity_slots=1) == primary[:2]


def test_comorbidity_names_extracts_and_dedups():
    ctx = {"conditions": [
        {"name_vi": "Suy gan do rượu"},
        {"display": "Hepatic encephalopathy", "name_vi": "Bệnh não gan"},
        {"name_vi": "Suy gan do rượu"},  # duplicate -> dropped
        {"icd10_code": "N18"},           # falls back to code
    ]}
    names = comorbidity_names(ctx)
    assert names == ["Suy gan do rượu", "Bệnh não gan", "N18"]
    assert comorbidity_names({}) == [] and comorbidity_names(None) == []


def test_comorbidity_queries_templated_and_capped():
    qs = comorbidity_queries(["Xơ gan", "Suy thận", "Suy tim"], "lưu ý ở bệnh nhân {cond}", max_n=2)
    assert qs == ["lưu ý ở bệnh nhân Xơ gan", "lưu ý ở bệnh nhân Suy thận"]


# --- Cross-encoder reranker ordering (src/rag/reranker.py, torch-free helper) -

def test_order_by_rerank_sorts_desc_and_annotates():
    # The off-topic chunk leads on bi-encoder order but the reranker scores it lowest -> demoted.
    chunks = [_doc("antivenom_junk", 0.44), _doc("sepsis_protocol", 0.41), _doc("misc", 0.40)]
    rr_scores = [-5.2, 6.1, -1.0]   # cross-encoder: sepsis most relevant, junk least
    out = order_by_rerank(chunks, rr_scores)
    assert [c["text"] for c in out] == ["sepsis_protocol", "misc", "antivenom_junk"]
    assert out[0]["rerank_score"] == 6.1
    assert out[0]["score"] == 0.41   # original bi-encoder score preserved alongside


def test_order_by_rerank_is_stable_on_ties_and_truncates():
    chunks = [_doc("a"), _doc("b"), _doc("c")]
    out = order_by_rerank(chunks, [1.0, 1.0, 0.0], top_k=2)
    assert [c["text"] for c in out] == ["a", "b"]   # tie keeps original order; cut to top_k


# --- Comorbidity-conflict enforcement gate (src/rag/comorbidity_gate.py) ------

_LIVER_CTX = {"conditions": [{"name_vi": "Suy gan do rượu"}, {"name_vi": "Bệnh não gan"}]}


def test_gate_flags_fluid_bolus_in_liver_failure():
    ans = "Truyền dịch nhanh 1000-2000ml trong 1-2 giờ đầu nếu tụt huyết áp."
    conf = check_comorbidity_conflicts(ans, _LIVER_CTX)
    assert len(conf) == 1 and conf[0]["id"] == "aggressive_fluids"
    assert conf[0]["comorbidity"] == "Suy gan do rượu"   # the matched condition name


def test_gate_flags_paracetamol_in_liver_failure_diacritic_insensitive():
    # diacritic-insensitive: "PARACETAMOL" upper-cased still matches.
    conf = check_comorbidity_conflicts("Có thể hạ sốt bằng PARACETAMOL.", _LIVER_CTX)
    assert any(c["id"] == "hepatotoxic_paracetamol" for c in conf)


def test_gate_no_conflict_without_matching_comorbidity():
    ctx = {"conditions": [{"name_vi": "Viêm phổi"}]}   # no liver/renal/cardiac
    assert check_comorbidity_conflicts("Truyền dịch nhanh 1000-2000ml.", ctx) == []


def test_gate_no_conflict_when_no_flagged_recommendation():
    assert check_comorbidity_conflicts("Cấy máu và dùng kháng sinh sớm.", _LIVER_CTX) == []


def test_apply_gate_prepends_banner_and_raises_alert():
    resp = {"answer": "Truyền dịch nhanh 1000-2000ml.", "alerts": [], "fallback": False}
    out = apply_comorbidity_gate(resp, _LIVER_CTX)
    assert out["answer"].startswith("⚠️ LƯU Ý BỆNH NỀN")
    assert "Truyền dịch nhanh" in out["answer"]            # original answer kept
    assert any(a["type"] == "comorbidity_conflict" for a in out["alerts"])
    assert out["comorbidity_conflicts"][0]["id"] == "aggressive_fluids"
    assert resp["answer"] == "Truyền dịch nhanh 1000-2000ml."  # input not mutated


def test_apply_gate_noop_on_fallback():
    resp = {"answer": "Không đủ thông tin…", "alerts": [], "fallback": True}
    assert apply_comorbidity_gate(resp, _LIVER_CTX) is resp
