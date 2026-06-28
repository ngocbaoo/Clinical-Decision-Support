"""
Offline tests for the patient-scoped web API (src/web/app.py).

These exercise the catalog + profile endpoints (fully local: FHIR file + SQLite lookups) and
the now-deterministic opening assessment — none of these build the RAG pipeline, so they need
no OPEN_ROUTER_KEY or Chroma. The OpenFDA drug-safety scan is stubbed so the tests stay offline
and fast. Only the chat endpoint hits the live LLM and is verified manually.
"""

import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import rag.safety as safety  # noqa: E402  (monkeypatched to keep the opener offline)
from web.app import (app, to_profile, _ctx_for,  # noqa: E402
                     _build_assessment, _normalize_drug_alert)

client = TestClient(app)


def test_list_patients_returns_full_catalog():
    r = client.get("/api/patients")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 18
    assert all({"id", "name", "description"} <= set(p) for p in data)
    assert any(p["id"] == "pt-001" for p in data)


def test_profile_shapes_allergy_and_scores():
    r = client.get("/api/patients/pt-001")
    assert r.status_code == 200
    prof = r.json()
    assert prof["name"]
    # pt-001 (Nguyễn Văn An): Penicillin allergy + a computed qSOFA score
    allergens = " ".join((a.get("allergen") or "") for a in prof["allergies"]).lower()
    assert "penicillin" in allergens
    assert prof["scores"]["qsofa"] is not None
    assert "summary" in prof and prof["summary"]


def test_unknown_patient_is_404():
    assert client.get("/api/patients/pt-999").status_code == 404


def test_empty_chat_query_rejected():
    # 400, not a pipeline call — guards before any LLM use.
    assert client.post("/api/patients/pt-001/chat", json={"query": "   "}).status_code == 400


def test_to_profile_is_pipeline_free():
    # Building a profile must not require the LLM pipeline (offline-safe).
    ctx, calc = _ctx_for("pt-003")
    prof = to_profile(ctx, calc)
    assert prof["id"] == "pt-003"
    assert isinstance(prof["vitals"], list)


def _stub_safety(monkeypatch, **fns):
    for name in ("check_allergies", "check_contraindications", "check_drug_interactions"):
        monkeypatch.setattr(safety, name, fns.get(name, lambda *a, **k: []))


def test_assessment_is_deterministic_and_pipeline_free(monkeypatch):
    # The opener must produce authoritative score flags + recorded allergies with NO LLM/pipeline.
    _stub_safety(monkeypatch)
    ctx, calc = _ctx_for("pt-001")
    a = _build_assessment(ctx, calc)
    assert a["kind"] == "assessment" and a["name"]
    assert isinstance(a["score_flags"], list)  # qSOFA/NEWS2/etc. strings from the scoring module
    assert any("penicillin" in (al.get("allergen") or "").lower() for al in a["allergies"])
    assert isinstance(a["drug_alerts"], list)
    assert "khuyến nghị" in a["note"]  # "not a treatment recommendation" disclaimer present


def test_assessment_safety_scan_degrades_on_error(monkeypatch):
    # An OpenFDA outage must NEVER block the opener — drug_alerts degrades to [], flags survive.
    def boom(*a, **k):
        raise RuntimeError("openfda down")
    _stub_safety(monkeypatch, check_allergies=boom,
                 check_contraindications=boom, check_drug_interactions=boom)
    a = _build_assessment(*_ctx_for("pt-001"))
    assert a["drug_alerts"] == []
    assert isinstance(a["score_flags"], list)


def test_assessment_endpoint_returns_assessment_kind(monkeypatch):
    _stub_safety(monkeypatch)
    r = client.post("/api/patients/pt-002/assessment")
    assert r.status_code == 200 and r.json()["kind"] == "assessment"


def test_normalize_drug_alert_severities():
    allergy = _normalize_drug_alert({"type": "allergy", "drug": "Amoxicillin",
                                     "allergen": "Penicillin", "reaction": "Anaphylaxis"})
    assert allergy["severity"] == "danger" and "Amoxicillin" in allergy["title"]
    contra = _normalize_drug_alert({"type": "contraindication", "drug": "Ibuprofen",
                                    "condition": "Suy thận", "snippet": "renal impairment"})
    assert contra["severity"] == "danger" and "Ibuprofen" in contra["title"]
    interact = _normalize_drug_alert({"type": "interaction", "drug_a": "Warfarin",
                                      "drug_b": "Aspirin", "snippet": "bleeding risk"})
    assert interact["severity"] == "warn" and "Warfarin" in interact["title"]
