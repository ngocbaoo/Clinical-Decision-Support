"""
Offline tests for the patient-scoped web API (src/web/app.py).

These exercise only the catalog + profile endpoints, which are fully local (FHIR file +
SQLite lookups) — they never build the RAG pipeline, so they need no OPEN_ROUTER_KEY or
Chroma. The chat/assessment endpoints hit the live LLM and are verified manually.
"""

import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from web.app import app, to_profile, _ctx_for  # noqa: E402

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
