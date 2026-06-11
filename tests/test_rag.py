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
