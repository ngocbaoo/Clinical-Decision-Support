"""
Patient-scoped web API over the ICU RAG pipeline (FastAPI).

Flow the UI enforces: pick a patient FIRST, then chat scoped to that patient. Every chat
message carries the patient's full FHIR context (single-turn), so the assistant always
"understands the situation". The opening assessment is one grounded+verified pipeline call.

This module is a pure CONSUMER of the pipeline — it changes nothing in src/rag, src/fhir,
src/scoring. Run:  uvicorn web.app:app --app-dir src --reload
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ on sys.path

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from paths import MOCK_DIR, ROOT  # noqa: E402
from fhir.fhir_client import FHIRClient  # noqa: E402
from scoring.calculator import calculate_all  # noqa: E402
from rag.context_builder import summarize_patient  # noqa: E402

ASSESSMENT_QUERY = ("Tóm tắt tình trạng hiện tại của bệnh nhân và nêu các vấn đề an toàn / "
                    "cần lưu ý quan trọng nhất.")

# --- patient catalog (static mock) -------------------------------------------
_INDEX = json.loads((MOCK_DIR / "index.json").read_text(encoding="utf-8"))
_PATIENTS = _INDEX["patients"]
_ID_TO_FILE = {p["id"]: p["file"] for p in _PATIENTS}

# --- caches ------------------------------------------------------------------
# ⚠️ These caches are valid ONLY because the mock FHIR bundles are STATIC. Against a LIVE
# FHIR server, ICU ctx/calc/assessment change by the minute, so caching would silently serve
# stale clinical data — a safety bug. On live FHIR: remove these, or add a short TTL +
# invalidate on new data. Do not let this demo shortcut become a real-world safety bug.
_ctx_cache: dict[str, tuple[dict, dict]] = {}
_assessment_cache: dict[str, dict] = {}

# Pipeline is built lazily on first LLM use (chat/assessment): the catalog + profile
# endpoints are fully offline (local FHIR file + SQLite), so they — and the offline tests —
# never need OPEN_ROUTER_KEY or the Chroma store.
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from rag.pipeline import RAGPipeline
        _pipeline = RAGPipeline()  # config defaults: flash + reasoning-off, verifier on
    return _pipeline


def _ctx_for(pid: str) -> tuple[dict, dict]:
    """(patient_context, calc) for a patient id; cached (static-mock only — see caveat)."""
    if pid not in _ID_TO_FILE:
        raise HTTPException(status_code=404, detail=f"unknown patient '{pid}'")
    if pid not in _ctx_cache:
        client = FHIRClient.from_file(str(MOCK_DIR / _ID_TO_FILE[pid]))
        ctx = client.build_patient_context()
        _ctx_cache[pid] = (ctx, calculate_all(ctx))
    return _ctx_cache[pid]


def to_profile(ctx: dict, calc: dict) -> dict:
    """Shape build_patient_context() + calculate_all() into a flat profile for the UI."""
    p = ctx.get("patient", {})
    enc = ctx.get("encounter", {})
    vitals = [{"key": k, "value": o.get("value"), "unit": o.get("unit") or ""}
              for k, o in ctx.get("observations", {}).items() if o.get("value") is not None]
    scores = {
        "map": (calc.get("map") or {}).get("value"),
        "qsofa": (calc.get("qsofa") or {}).get("total"),
        "qsofa_positive": (calc.get("qsofa") or {}).get("positive"),
        "sofa": (calc.get("sofa") or {}).get("total"),
        "news2": (calc.get("news2") or {}).get("total"),
        "news2_risk": (calc.get("news2") or {}).get("risk_level"),
        "egfr": (calc.get("egfr") or {}).get("egfr"),
        "egfr_stage": (calc.get("egfr") or {}).get("stage"),
    }
    return {
        "id": ctx.get("patient_id"),
        "name": p.get("name", "?"),
        "age": p.get("age"),
        "gender": p.get("gender"),
        "encounter": {"service_type": enc.get("service_type"),
                      "class": enc.get("class"),
                      "period_start": enc.get("period_start"),
                      "reasons": enc.get("reasons", [])},
        "allergies": [{"allergen": a.get("allergen"), "criticality": a.get("criticality"),
                       "reaction": a.get("reaction")} for a in ctx.get("allergies", [])],
        "conditions": [(c.get("name_vi") or c.get("display") or c.get("icd10_code") or "?")
                       for c in ctx.get("conditions", [])],
        "medications": [{"name": m.get("name"), "dose": m.get("dose")}
                        for m in ctx.get("medications", [])],
        "vitals": vitals,
        "scores": scores,
        "alerts": (calc.get("summary") or {}).get("alerts", []),
        "summary": summarize_patient(ctx, calc),
    }


def _shape_answer(result: dict) -> dict:
    """Slim a pipeline result into the JSON the UI needs (drops chunks)."""
    r = result["response"]
    v = r.get("verify") or {}
    return {
        "answer": r["answer"],
        "cited_sources": r["cited_sources"],
        "alerts": r["alerts"],
        "fallback": r["fallback"],
        "fallback_reason": r["fallback_reason"],
        "verify": {"status": v.get("status"), "branch": v.get("branch"),
                   "unsupported_ratio": v.get("unsupported_ratio")},
        "timings_s": result["timings_s"],
        "request_id": result["request_id"],
    }


app = FastAPI(title="ICU RAG — patient-scoped assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"], allow_headers=["*"],
)


class ChatIn(BaseModel):
    query: str


_demographics_cache: dict[str, dict] = {}  # static-mock only — see caching caveat above


def _demographics(pid: str, file: str) -> dict:
    """Light demographics (gender/birthDate/age) for the selector — reads only the Patient
    resource via get_patient(), NOT the full 9-query build_patient_context()."""
    if pid not in _demographics_cache:
        p = FHIRClient.from_file(str(MOCK_DIR / file)).get_patient()
        _demographics_cache[pid] = {"gender": p.get("gender"),
                                    "birthDate": p.get("birthDate"), "age": p.get("age")}
    return _demographics_cache[pid]


@app.get("/api/patients")
def list_patients():
    return [{"id": p["id"], "name": p["name"], "description": p.get("description", ""),
             **_demographics(p["id"], p["file"])}
            for p in _PATIENTS]


@app.get("/api/patients/{pid}")
def get_patient(pid: str):
    ctx, calc = _ctx_for(pid)
    return to_profile(ctx, calc)


@app.post("/api/patients/{pid}/assessment")
def assessment(pid: str):
    ctx, calc = _ctx_for(pid)
    if pid not in _assessment_cache:  # static-mock cache — see caveat above
        result = get_pipeline().ask(ASSESSMENT_QUERY, ctx, calc)
        _assessment_cache[pid] = _shape_answer(result)
    return _assessment_cache[pid]


@app.post("/api/patients/{pid}/chat")
def chat(pid: str, body: ChatIn):
    ctx, calc = _ctx_for(pid)
    query = (body.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="empty query")
    return _shape_answer(get_pipeline().ask(query, ctx, calc))


# Serve the built SPA (prod = one server). Mounted last so /api/* wins. Dev uses Vite instead.
_DIST = ROOT / "frontend" / "dist"
if _DIST.exists():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="spa")
