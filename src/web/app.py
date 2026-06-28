"""
Patient-scoped web API over the ICU RAG pipeline (FastAPI).

Flow the UI enforces: pick a patient FIRST, then chat scoped to that patient. Every chat
message carries the patient's full FHIR context (single-turn), so the assistant always
"understands the situation". The opening assessment is a DETERMINISTIC chart summary —
authoritative score flags + allergies + an OpenFDA drug-safety scan over the patient's current
medications. No LLM, no citation gate, no fallback: it never refuses and is effectively instant
(vs the old ~35s pipeline call that fell back whenever the summary cited no guideline chunk).

This module is a pure CONSUMER of the pipeline — it changes nothing in src/rag, src/fhir,
src/scoring. Run:  uvicorn web.app:app --app-dir src --reload
"""

import io
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ on sys.path

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from paths import MOCK_DIR, ROOT  # noqa: E402
from fhir.fhir_client import FHIRClient  # noqa: E402
from scoring.calculator import calculate_all  # noqa: E402
from rag.context_builder import summarize_patient  # noqa: E402

_GENDER_VI = {"male": "Nam", "female": "Nữ", "other": "Khác", "unknown": "Không rõ"}

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


# ASR is loaded just as lazily as the pipeline: the CTranslate2 model and faster-whisper import
# never touch the offline catalog/profile path (or the offline tests). It is built on the first
# /api/asr/transcribe call. The serve-time runtime is torch-free (ctranslate2 only); the int8 model
# loads in ~2s vs the old ~8s transformers cold-start (docs/ASR_CT2_MIGRATION.md).
_transcriber = None


def get_transcriber():
    global _transcriber
    if _transcriber is None:
        from asr.config import ASR_MODEL
        from asr.transcriber import WhisperTranscriber
        _transcriber = WhisperTranscriber(ASR_MODEL)
    return _transcriber


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


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("warmup")


def warmup_models() -> None:
    """Eagerly load EVERY heavy model at app boot (not lazily on first request): the RAG pipeline
    (chat + embedder + Chroma), the cross-encoder reranker, the local NLI verifier, and the ASR
    transcriber — each warmed with a tiny inference so the first real request pays no cold-start /
    CUDA-kernel cost. Per-model timing is logged; each stage is guarded so one failure (e.g. GPU OOM
    on the 4GB card) degrades that model but still boots the app. The previous lazy loads made the
    first chat block for tens of seconds while a 2.2GB reranker loaded mid-request → 'no response'."""
    log.info("Warming up models on startup…")

    t = time.perf_counter()
    try:
        pipe = get_pipeline()
        log.info("✓ RAG pipeline (chat + embedder + Chroma) in %.1fs", time.perf_counter() - t)
    except Exception:
        log.exception("✗ RAG pipeline FAILED to load — chat/assessment will error")
        return  # nothing downstream works without the pipeline

    t = time.perf_counter()
    try:
        rr = pipe._get_reranker()
        if rr is not None:
            rr.rerank("khởi động", [{"text": "khởi động mô hình xếp hạng"}])
            log.info("✓ cross-encoder reranker in %.1fs", time.perf_counter() - t)
        else:
            log.warning("• reranker disabled / model not cached → bi-encoder retrieval only")
    except Exception:
        log.exception("✗ reranker FAILED → bi-encoder retrieval only")

    t = time.perf_counter()
    try:
        nli = pipe._get_nli()
        if nli is not None:
            nli("khởi động", "khởi động")
            log.info("✓ NLI verifier in %.1fs", time.perf_counter() - t)
        else:
            log.info("• NLI verifier not required by current config")
    except Exception:
        log.exception("✗ NLI verifier FAILED to load")

    t = time.perf_counter()
    try:
        import numpy as np
        get_transcriber().transcribe(np.zeros(16000, dtype="float32"), 16000)
        log.info("✓ ASR transcriber in %.1fs", time.perf_counter() - t)
    except Exception:
        log.exception("✗ ASR transcriber FAILED to load")

    log.info("Warmup complete — all models resident; first request will be fast.")


@asynccontextmanager
async def lifespan(_app: "FastAPI"):
    # Skip eager warmup under pytest (offline tests must stay torch-free) or when explicitly
    # disabled (WARMUP_MODELS=0). Otherwise load everything before the server accepts requests.
    if "pytest" not in sys.modules and os.getenv("WARMUP_MODELS", "1") != "0":
        warmup_models()
    yield


app = FastAPI(title="ICU RAG — patient-scoped assistant", lifespan=lifespan)
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


def _normalize_drug_alert(a: dict) -> dict:
    """Flatten a safety-scan alert into {type, severity, title, detail} for the UI."""
    t = a.get("type")
    if t == "allergy":
        detail = f"Đang dùng thuốc trùng dị ứng đã ghi nhận ({a.get('allergen')})"
        if a.get("reaction"):
            detail += f" — phản ứng: {a['reaction']}"
        return {"type": t, "severity": "danger",
                "title": f"Dị ứng × thuốc: {a.get('drug')}", "detail": detail}
    if t == "contraindication":
        return {"type": t, "severity": "danger",
                "title": f"Chống chỉ định: {a.get('drug')} ↔ {a.get('condition')}",
                "detail": (a.get("snippet") or "").strip()[:240]}
    if t == "interaction":
        return {"type": t, "severity": "warn",
                "title": f"Tương tác thuốc: {a.get('drug_a')} ↔ {a.get('drug_b')}",
                "detail": (a.get("snippet") or "").strip()[:240]}
    return {"type": t or "alert", "severity": "warn",
            "title": a.get("drug") or "Cảnh báo an toàn", "detail": ""}


def _build_assessment(ctx: dict, calc: dict) -> dict:
    """Deterministic opening assessment: no LLM. Pulls the authoritative score flags the
    scoring module already computed, the recorded allergies, and runs the OpenFDA drug-safety
    scan over the patient's CURRENT medications (allergy-on-med, med↔condition contraindication,
    med↔med interaction). The OpenFDA scan is best-effort — a network outage degrades to no drug
    alerts, never blocks or refuses the opener."""
    profile = to_profile(ctx, calc)
    meds = [m["name"] for m in profile["medications"] if m.get("name")]
    drug_alerts: list[dict] = []
    try:
        # Lazy import keeps the offline catalog/profile path free of rag.safety/openfda.
        from rag.safety import (check_allergies, check_contraindications,
                                 check_drug_interactions)
        raw = (check_allergies(meds, ctx) + check_contraindications(meds, ctx)
               + check_drug_interactions(meds, ctx))
        drug_alerts = [_normalize_drug_alert(a) for a in raw]
    except Exception as exc:  # OpenFDA outage etc. — degrade, never block the opener
        print(f"  [assessment safety scan failed] {exc}", file=sys.stderr)

    subtitle = " · ".join(filter(None, [
        _GENDER_VI.get(profile["gender"], profile["gender"]),
        f"{profile['age']} tuổi" if profile.get("age") is not None else None,
        (profile.get("encounter") or {}).get("service_type"),
    ]))
    return {
        "kind": "assessment",
        "name": profile["name"],
        "subtitle": subtitle,
        "score_flags": (calc.get("summary") or {}).get("alerts", []),
        "allergies": profile["allergies"],
        "drug_alerts": drug_alerts,
        "conditions": profile["conditions"],
        "note": ("Tóm tắt tự động từ hồ sơ bệnh nhân — không phải khuyến nghị điều trị. "
                 "Đặt câu hỏi bên dưới để nhận tư vấn có trích dẫn guideline."),
    }


@app.post("/api/patients/{pid}/assessment")
def assessment(pid: str):
    ctx, calc = _ctx_for(pid)
    if pid not in _assessment_cache:  # static-mock cache — see caveat above
        _assessment_cache[pid] = _build_assessment(ctx, calc)
    return _assessment_cache[pid]


@app.post("/api/patients/{pid}/chat")
def chat(pid: str, body: ChatIn):
    ctx, calc = _ctx_for(pid)
    query = (body.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="empty query")
    return _shape_answer(get_pipeline().ask(query, ctx, calc))


@app.post("/api/asr/transcribe")
async def asr_transcribe(request: Request):
    """Push-to-talk → transcript. The client posts a raw mono WAV blob (Content-Type audio/wav);
    we read it with soundfile (no python-multipart / ffmpeg needed), transcribe with the drug-name
    initial_prompt bias, and run the SUGGEST-only drug matcher. Returns {text, latency_s,
    suggestions} — the UI puts `text` in an EDITABLE box and shows suggestions as hints. It NEVER
    auto-sends and NEVER rewrites the transcript: the doctor confirms (F-ASR-04/05, Risk #1)."""
    import numpy as np
    import soundfile as sf

    from asr.drug_lexicon import PROMPT_DRUGS
    from asr.drug_match import suggest_drugs

    raw = await request.body()
    if not raw:
        raise HTTPException(status_code=400, detail="empty audio")
    try:
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32")
    except Exception as exc:  # unreadable / unsupported container
        raise HTTPException(status_code=400, detail=f"unreadable audio: {exc}")
    if getattr(audio, "ndim", 1) > 1:  # downmix stereo → mono
        audio = audio.mean(axis=1)
    if audio.size == 0:
        raise HTTPException(status_code=400, detail="silent / zero-length audio")

    result = get_transcriber().transcribe(np.ascontiguousarray(audio), sr, prompt=PROMPT_DRUGS)
    return {"text": result["text"], "latency_s": result["latency_s"],
            "suggestions": suggest_drugs(result["text"])}


# Serve the built SPA (prod = one server). Mounted last so /api/* wins. Dev uses Vite instead.
_DIST = ROOT / "frontend" / "dist"
if _DIST.exists():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="spa")
