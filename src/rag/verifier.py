"""
R4b — Faithfulness verifier: claim-level entailment + safety decision tree.

Each generated claim arrives as (text, evidence, citation) from the T1 structured
generator (src/rag/context_builder.SYSTEM_PROMPT). The verifier assigns each claim a
verdict {supported, neutral, contradicted} against its cited chunk, then a
CODE-ENFORCED decision tree (decide()) decides keep / strip / fallback. Safety is
privileged: a contradiction, an unsupported safety claim, or a broken sequential
procedure each fall back the WHOLE answer rather than silently editing it — because
excising one line from an answer that just proved it is unreliable keeps the output of
a model that is post-rationalizing.

Backends:
  - "llm":       openai/gpt-5.4-mini batched entailment (different family from the qwen
                 generator -> avoids correlated errors).
  - "local_nli": offline mDeBERTa-XNLI via ONNX (no torch) — used once the Phase-1 spike
                 (src/rag/eval/nli_validation.py) passes the safety bar.
  - "hybrid":    local NLI for easy claims, escalate low-confidence/safety to the LLM.

The evidence-quote substring match is a POSITIVE FAST-PATH ONLY ($0): a hit short-circuits
to `supported`; a miss is NOT a penalty (VN normalization is brittle) — the claim simply
falls through to the backend, checked directly against the cited chunk text.

Fail-CLOSED for safety-critical answers, fail-OPEN (with a visible banner, set by the
generator) otherwise — never a silent revert to the unverified baseline.
"""

import json
import re
import sys
import unicodedata
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ on sys.path
from rag.query_router import parse_json_loose  # noqa: E402
from prompts import load_prompt  # noqa: E402

# Folded (diacritic-free) safety markers — code backstop so a verifier miss can't
# silently downgrade a safety claim.
_SAFETY_KW = ("chong chi dinh", "khong duoc dung", "khong nen dung", "chong dung",
              "tai bien", "bien chung", "nguy hiem", "thong than trong")
# Markers that an answer is a sequential / referential procedure.
_ORDERED_KW = ("buoc 1", "buoc 2", "sau do", "tiep theo", "tiep tuc", "neu that bai",
               "neu khong", "truoc tien", "dau tien", "cuoi cung")


def _fold(s: str) -> str:
    """Lowercase + strip Vietnamese diacritics + collapse whitespace.

    NFD does not decompose đ/Đ (they are not base+combining), so map them explicitly —
    otherwise "chống chỉ định" folds to "...đinh" and the keyword backstop misses.
    """
    s = (s or "").lower().replace("đ", "d")
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", s).strip()


def _evidence_matches(evidence: str, chunk_text: str) -> bool:
    """Positive fast-path: is the model's quoted evidence really in the chunk?"""
    e = _fold(evidence)
    return len(e) >= 12 and e in _fold(chunk_text)


def _norm_tokens(s: str) -> list[str]:
    """Alphanumeric tokens after compatibility + diacritic normalization, so VN clinical text
    matches despite formatting noise: NFKC folds subscripts/units (SpO₂ -> SpO2, full-width),
    then diacritics are stripped and dash/space variants vanish under tokenization
    ("5–10 phút" / "5-10 phút" -> ["5","10","phut"]; "hô hấp" / "ho hap" -> ["ho","hap"])."""
    s = unicodedata.normalize("NFKC", s or "").lower().replace("đ", "d")
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.findall(r"[a-z0-9]+", s)


def _evidence_grounded(evidence: str, chunk_text: str, min_coverage: float = None) -> bool:
    """Lever 1 — is the model's quoted evidence REALLY in the cited chunk? Fuzzy (token-coverage)
    instead of exact substring, because exact match dies on VN diacritic/dash/unit noise. Returns
    True iff >= min_coverage of the evidence's content tokens appear in the chunk. A fabricated
    quote (tokens absent from the chunk) scores low and fails -> the claim is dropped, not trusted.
    """
    from rag.config import EVIDENCE_MIN_COVERAGE
    cov = EVIDENCE_MIN_COVERAGE if min_coverage is None else min_coverage
    ev = _norm_tokens(evidence)
    if len(ev) < 3:           # too short to be a meaningful quote -> not grounding
        return False
    chunk = set(_norm_tokens(chunk_text))
    hits = sum(1 for t in ev if t in chunk)
    return hits / len(ev) >= cov


def _is_effective_safety(text: str, model_is_safety: bool, intent: str) -> bool:
    """A claim is safety if the verifier flagged it, OR a keyword backstop fires, OR
    the whole question is a contraindication query (every claim treated as safety)."""
    if model_is_safety or intent == "contraindication":
        return True
    folded = _fold(text)
    return any(kw in folded for kw in _SAFETY_KW)


def _looks_ordered(claims: list[dict]) -> bool:
    blob = _fold(" ".join(c.get("text", "") for c in claims))
    if any(kw in blob for kw in _ORDERED_KW):
        return True
    return bool(re.search(r"\b[1-9]\)", blob)) or bool(re.search(r"\b[1-9]\.", blob))


# Hedge prefix for an unverifiable patient-derived safety caution: kept (not silently
# dropped, not allowed to fallback the whole answer) but stripped of false certainty.
HEDGE_PREFIX = ("⚠️ Lưu ý (dựa trên dữ liệu bệnh nhân, chưa đối chiếu được "
                "guideline — cần dược sĩ xác nhận): ")


def _alert_confirms(text: str, alerts: list[dict] | None) -> bool:
    """True if a deterministic safety gate already produced an alert matching this claim.

    The alerts come from check_allergies / check_contraindications / check_drug_interactions —
    code/FDA-derived facts about THIS patient, not LLM assertions. A citation=null safety claim
    that restates such an alert is trustworthy by virtue of the gate, not the model. Match = the
    claim text mentions one of the alert's key terms (drug / allergen / condition / paired drug).
    """
    folded = _fold(text)
    for a in alerts or []:
        for key in ("drug", "allergen", "condition", "matched", "drug_a", "drug_b"):
            term = _fold(a.get(key) or "")
            if len(term) >= 4 and term in folded:
                return True
    return False


# --- backends ----------------------------------------------------------------
# Prompt lives in src/prompts/verifier.xml.
_LLM_PROMPT = load_prompt("verifier")


def classify_with_llm(claims: list[dict], chunks: list[dict], patient_summary: str,
                      verifier_chat) -> list[dict]:
    """One batched entailment call. Returns a verdict dict per input claim (by index)."""
    docs = "\n\n".join(f"[{i + 1}] {c.get('text', '')[:1500]}"
                       for i, c in enumerate(chunks))
    claim_lines = "\n".join(
        f'{i}. (cites [{c.get("citation")}]) {c.get("text", "")}'
        for i, c in enumerate(claims))
    user = (f"SOURCE CHUNKS:\n{docs or '(none)'}\n\n"
            f"PATIENT DATA:\n{patient_summary}\n\n"
            f"CLAIMS:\n{claim_lines}")
    reply = verifier_chat.chat(
        [{"role": "system", "content": _LLM_PROMPT},
         {"role": "user", "content": user}],
        temperature=0.0, max_tokens=700)
    data = parse_json_loose(reply)
    by_i = {item.get("i"): item for item in (data.get("claims") or [])}
    out = []
    for i in range(len(claims)):
        item = by_i.get(i, {})
        verdict = item.get("verdict")
        if verdict not in ("supported", "neutral", "contradicted"):
            verdict = "neutral"  # unparseable -> treat as unsupported (conservative)
        out.append({"verdict": verdict, "is_safety": bool(item.get("is_safety"))})
    return out


# --- decision tree (pure, code-enforced) -------------------------------------
def decide(claims: list[dict], verdicts: list[dict], intent: str) -> dict:
    """Apply the §3a safety decision tree. `claims` and `verdicts` are index-aligned.

    Each verdict: {verdict, is_safety}. Returns the action + metrics; never raises.
    """
    n = len(claims) or 1
    enriched = []
    for c, v in zip(claims, verdicts):
        enriched.append({
            "text": c.get("text", ""),
            "citation": c.get("citation"),
            "verdict": v["verdict"],
            "safety": _is_effective_safety(c.get("text", ""), v.get("is_safety", False),
                                           intent),
        })
    contradicted = sum(1 for c in enriched if c["verdict"] == "contradicted")
    neutral = sum(1 for c in enriched if c["verdict"] == "neutral")
    unsupported_ratio = round((contradicted + neutral) / n, 3)
    ordered = _looks_ordered(claims)

    base = {"unsupported_ratio": unsupported_ratio, "contradiction_count": contradicted,
            "is_ordered_procedure": ordered, "verdicts": enriched}

    unsupported_safety = [c for c in enriched if c["safety"] and c["verdict"] != "supported"]
    base["hedged_count"] = 0

    # 1. any contradiction -> trust broken, drop the whole answer
    if contradicted:
        return {**base, "action": "fallback", "branch": "contradicted",
                "fallback_reason": "verifier_contradicted", "kept_claims": []}
    # 2. unsupported safety claim that CITES a guideline chunk -> fail-CLOSED: a misquoted/
    #    fabricated guideline safety claim is the dangerous case, drop the whole answer.
    #    (Patient-derived safety claims (citation=null) that a deterministic alert confirms were
    #    already marked supported upstream; the rest are hedged in step 4, not killed — killing a
    #    real patient-grounded caution suppresses a valid warning, which is LESS safe.)
    if any(isinstance(c["citation"], int) for c in unsupported_safety):
        return {**base, "action": "fallback", "branch": "safety",
                "fallback_reason": "verifier_unsupported_safety", "kept_claims": []}
    # 3. sequential procedure with any non-supported step -> can't excise a step
    if ordered and any(c["verdict"] != "supported" for c in enriched):
        return {**base, "action": "fallback", "branch": "integrity",
                "fallback_reason": "verifier_integrity_break", "kept_claims": []}
    # 4. keep supported claims; hedge (don't drop) patient-derived unverified safety cautions
    kept = [c for c in enriched if c["verdict"] == "supported"]
    hedged = [{**c, "text": HEDGE_PREFIX + c["text"]} for c in unsupported_safety]
    base["hedged_count"] = len(hedged)
    all_kept = kept + hedged
    if not all_kept:
        return {**base, "action": "fallback", "branch": "empty",
                "fallback_reason": "verifier_unsupported", "kept_claims": []}
    branch = "hedge" if hedged else "strip"
    return {**base, "action": "keep", "branch": branch, "fallback_reason": None,
            "kept_claims": all_kept}


def verify_answer(claims: list[dict], chunks: list[dict], patient_summary: str,
                  intent: str, *, backend: str = "llm", verifier_chat=None,
                  nli=None, alerts: list[dict] | None = None) -> dict:
    """Classify claims (with the $0 evidence fast-path) then apply decide().

    Returns the decide() dict plus "status" ("ok" | "error"). On any backend failure
    returns status="error" and lets the caller choose fail-open vs fail-closed.
    """
    if not claims:
        return {"status": "ok", "action": "keep", "branch": "empty_input",
                "kept_claims": [], "unsupported_ratio": 0.0, "contradiction_count": 0,
                "is_ordered_procedure": False, "fallback_reason": None, "verdicts": []}

    # Lever 1 — evidence binding, CODE-ENFORCED ("grounded by enforcement", not by convention):
    # a claim that CITES a chunk must quote evidence that actually appears in that chunk. Grounded
    # -> supported ($0 fast-path). NOT grounded -> "neutral" by code (the model fabricated/misquoted
    # the evidence) -> dropped by decide(), NEVER handed to the lenient LLM backend that used to
    # rubber-stamp it. Claims citing NO chunk (patient-data / pre-computed scores) still go to the
    # backend, which checks them against the patient summary.
    from rag.config import VERIFY_EVIDENCE_NLI, NLI_REJECT_CONF
    _NLI_MAP = {"entailment": "supported", "contradiction": "contradicted", "neutral": "neutral"}
    fast = [None] * len(claims)
    pending_idx = []
    for i, c in enumerate(claims):
        cit = c.get("citation")
        chunk = chunks[cit - 1] if isinstance(cit, int) and 1 <= cit <= len(chunks) else None
        if chunk is not None:
            evidence = c.get("evidence", "")
            if not _evidence_grounded(evidence, chunk.get("text", "")):
                fast[i] = {"verdict": "neutral", "is_safety": False}  # Lever 1: fabricated/misquoted
            elif VERIFY_EVIDENCE_NLI and nli is not None:
                # Lever 2: grounded is necessary but NOT sufficient — the claim must be entailed by
                # its OWN evidence span (catches "real quote, over-reaching claim"). Tight premise.
                # Confidence-gated: drop only when NLI is CONFIDENT it is not entailed; an entailed
                # OR low-confidence verdict keeps the claim (mDeBERTa over-rejects VN paraphrases).
                label, conf = nli(evidence, c.get("text", ""))
                if label != "entailment" and conf >= NLI_REJECT_CONF:
                    fast[i] = {"verdict": _NLI_MAP.get(label, "neutral"), "is_safety": False}
                else:
                    fast[i] = {"verdict": "supported", "is_safety": False}
            else:
                fast[i] = {"verdict": "supported", "is_safety": False}  # Lever-1-only fallback
        elif _alert_confirms(c.get("text", ""), alerts):
            # Lever 3 — citation=null claim restating a deterministic safety alert (allergy /
            # contraindication / interaction). The alert is code/FDA-derived for THIS patient, so
            # the claim is trustworthy WITHOUT the lenient backend (which marks it neutral because
            # the terse patient_summary states the fact but not the recommendation).
            fast[i] = {"verdict": "supported", "is_safety": True}
        else:
            pending_idx.append(i)

    try:
        if pending_idx:
            pending = [claims[i] for i in pending_idx]
            if backend == "local_nli":
                got = classify_with_nli(pending, chunks, patient_summary, nli)
            elif backend == "hybrid":
                got = classify_hybrid(pending, chunks, patient_summary, nli, verifier_chat)
            else:
                got = classify_with_llm(pending, chunks, patient_summary, verifier_chat)
            for j, i in enumerate(pending_idx):
                fast[i] = got[j]
    except Exception as err:  # noqa: BLE001 — caller decides fail-open/closed
        return {"status": "error", "error": str(err), "action": "error",
                "kept_claims": [], "unsupported_ratio": None, "contradiction_count": None,
                "is_ordered_procedure": False, "fallback_reason": None, "verdicts": []}

    decision = decide(claims, fast, intent)
    decision["status"] = "ok"
    return decision


# --- local NLI backends (wired after the Phase-1 ONNX spike passes) ----------
def classify_with_nli(claims, chunks, patient_summary, nli):
    """Map (premise=cited chunk, hypothesis=claim) through an entailment model.

    `nli(premise, hypothesis) -> (label, confidence)` with label in
    {entailment, neutral, contradiction}. Raises if no nli callable is provided so the
    pipeline can fall back / escalate rather than silently pass.
    """
    if nli is None:
        raise RuntimeError("local_nli backend requested but no NLI model loaded "
                           "(run the Phase-1 spike + export the ONNX model first)")
    _MAP = {"entailment": "supported", "contradiction": "contradicted",
            "neutral": "neutral"}
    out = []
    for c in claims:
        cit = c.get("citation")
        chunk = chunks[cit - 1] if isinstance(cit, int) and 1 <= cit <= len(chunks) else None
        premise = (chunk or {}).get("text", "") or patient_summary
        label, _conf = nli(premise, c.get("text", ""))
        out.append({"verdict": _MAP.get(label, "neutral"), "is_safety": False})
    return out


def classify_hybrid(claims, chunks, patient_summary, nli, verifier_chat):
    """Local NLI for confident calls; escalate low-confidence / contradiction / safety
    to the LLM (where light NLI is weakest on VN negation)."""
    from rag.config import VERIFY_NLI_CONF
    out = [None] * len(claims)
    escalate = []
    for i, c in enumerate(claims):
        cit = c.get("citation")
        chunk = chunks[cit - 1] if isinstance(cit, int) and 1 <= cit <= len(chunks) else None
        premise = (chunk or {}).get("text", "") or patient_summary
        label, conf = nli(premise, c.get("text", ""))
        safety = _is_effective_safety(c.get("text", ""), False, "")
        if conf < VERIFY_NLI_CONF or label == "contradiction" or safety:
            escalate.append(i)
        else:
            out[i] = {"verdict": {"entailment": "supported", "neutral": "neutral",
                                  "contradiction": "contradicted"}.get(label, "neutral"),
                      "is_safety": safety}
    if escalate:
        got = classify_with_llm([claims[i] for i in escalate], chunks, patient_summary,
                                verifier_chat)
        for j, i in enumerate(escalate):
            out[i] = got[j]
    return out
