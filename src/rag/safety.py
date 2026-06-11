"""
R3 — Safety gate: allergy conflict check (Safety Req #2 — alert renders FIRST).

Matches drugs mentioned in the query (router entities) against the patient's
AllergyIntolerance list, including basic cross-reactivity groups (a Penicillin
allergy must flag Amoxicillin). `check_allergies` is pure / no network.

Drug-drug interaction checking (F-RAG-05) is backed by OpenFDA drug labels
(see rag/openfda.py); it makes network calls but degrades to [] on any error.
"""

import sys
import unicodedata
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ on sys.path
import re  # noqa: E402
from rag.openfda import (find_mention, get_contraindication_text,  # noqa: E402
                         get_interaction_text)

# ICD-description filler words: too generic to be a reliable contraindication
# trigger, so we drop them when extracting condition terms to match.
_CONDITION_STOPWORDS = {
    "unspecified", "organism", "disease", "disorder", "syndrome", "other",
    "without", "with", "acute", "chronic", "type", "stage", "primary",
    "secondary", "complication", "complications", "finding", "status",
}

# Cross-reactivity groups: an allergy to any member flags every member.
# Keys are canonical group names; members are lowercase substrings.
ALLERGY_GROUPS = {
    "penicillin": ("penicillin", "amoxicillin", "ampicillin", "piperacillin",
                   "oxacillin", "augmentin", "amoxicillin-clavulanate"),
    "cephalosporin": ("cephalosporin", "cefazolin", "ceftriaxone", "cefepime",
                      "cefotaxime", "ceftazidime", "cefuroxime", "cephalexin"),
    "sulfonamide": ("sulfonamide", "sulfa", "sulfamethoxazole", "cotrimoxazole",
                    "co-trimoxazole", "bactrim", "trimethoprim-sulfamethoxazole"),
    "nsaid": ("aspirin", "ibuprofen", "diclofenac", "ketorolac", "naproxen",
              "nsaid"),
    "aminoglycoside": ("gentamicin", "amikacin", "tobramycin", "aminoglycoside"),
    "quinolone": ("ciprofloxacin", "levofloxacin", "moxifloxacin", "quinolone"),
}


def _norm(s: str) -> str:
    """Lowercase + strip Vietnamese diacritics for tolerant matching."""
    s = unicodedata.normalize("NFD", (s or "").lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def _group_of(drug: str) -> str | None:
    d = _norm(drug)
    for group, members in ALLERGY_GROUPS.items():
        if any(m in d for m in members):
            return group
    return None


def check_allergies(drugs: list[str], patient_context: dict) -> list[dict]:
    """Return alerts for query drugs that conflict with recorded allergies.

    A conflict is a direct name match (substring either way) or shared
    cross-reactivity group. Returns [] when no patient context / no allergies.
    """
    alerts = []
    allergies = (patient_context or {}).get("allergies", [])
    for drug in drugs:
        d = _norm(drug)
        if not d:
            continue
        d_group = _group_of(drug)
        for a in allergies:
            allergen = a.get("allergen") or ""
            al = _norm(allergen)
            if not al:
                continue
            direct = al in d or d in al
            grouped = d_group is not None and _group_of(allergen) == d_group
            if direct or grouped:
                alerts.append({
                    "type": "allergy",
                    "drug": drug,
                    "allergen": allergen,
                    "criticality": a.get("criticality"),
                    "reaction": a.get("reaction"),
                    "match": "direct" if direct else f"cross-reactivity ({d_group})",
                })
    return alerts


def check_drug_interactions(drugs: list[str], patient_context: dict) -> list[dict]:
    """F-RAG-05 — screen query drugs against each other + the patient's current
    medications using OpenFDA drug-label interaction sections.

    For each query drug we fetch its label's interaction text and check whether
    any *other* drug (another queried drug or a recorded medication) is named in
    it. Pairs are de-duplicated so A↔B is reported once. Network failures /
    unknown drugs degrade silently to no alert (see rag/openfda.py).
    """
    if not drugs:
        return []

    meds = [(m.get("name") or "").strip()
            for m in (patient_context or {}).get("medications", [])]
    meds = [m for m in meds if m]
    query_drugs = [d.strip() for d in drugs if d and d.strip()]

    alerts: list[dict] = []
    seen_pairs: set[frozenset] = set()
    for drug in query_drugs:
        paras = get_interaction_text(drug)
        if not paras:
            continue
        text = "\n".join(paras)
        candidates = [d for d in query_drugs if _norm(d) != _norm(drug)] + meds
        for other in candidates:
            pair = frozenset((_norm(drug), _norm(other)))
            if len(pair) < 2 or pair in seen_pairs:
                continue
            snippet = find_mention(text, other)
            if snippet:
                seen_pairs.add(pair)
                source = "thuốc kê đơn" if other in meds else "câu hỏi"
                alerts.append({
                    "type": "interaction",
                    "drug_a": drug,
                    "drug_b": other,
                    "other_source": source,
                    "snippet": snippet,
                })
    return alerts


def _condition_terms(patient_context: dict) -> list[tuple[str, str, bool]]:
    """Patient triggers to look for in a drug's contraindication text.

    Returns (search_term, display_label, whole_word). English condition tokens
    (from `name_en`, len >= 5, non-filler) are matched whole-word; a pregnancy
    trigger is added as the stem 'pregnan' (catches pregnant/pregnancy) when the
    patient record signals it.
    """
    terms: list[tuple[str, str, bool]] = []
    ctx = patient_context or {}
    for c in ctx.get("conditions", []):
        label = (c.get("name_vi") or c.get("name_en") or c.get("display")
                 or c.get("icd10_code") or "?")
        english = c.get("name_en") or c.get("display") or ""
        for tok in re.findall(r"[a-zA-Z]+", english):
            if len(tok) >= 5 and tok.lower() not in _CONDITION_STOPWORDS:
                terms.append((tok, label, True))

    # Pregnancy is a high-value, frequently-listed contraindication trigger.
    # Only add the stem when the English name doesn't already cover it (else the
    # whole-word token match above fires too, double-reporting the same fact).
    conds = ctx.get("conditions", [])
    english_blob = _norm(" ".join(c.get("name_en") or "" for c in conds))
    vi_blob = _norm(" ".join(c.get("name_vi") or "" for c in conds))
    if "pregnan" not in english_blob and ("thai" in vi_blob or "pregnan" in vi_blob):
        terms.append(("pregnan", "Thai kỳ", False))
    return terms


def check_contraindications(drugs: list[str], patient_context: dict) -> list[dict]:
    """F-RAG-05 — screen query drugs against the patient's conditions using the
    OpenFDA `contraindications` label section.

    For each query drug we fetch its contraindication text and check whether any
    of the patient's condition terms (or pregnancy) is named in it. De-duplicated
    per (drug, condition). Network failures / unknown drugs degrade silently to
    no alert (see rag/openfda.py). Cross-lingual caveat: matching relies on the
    condition's English name, so Vietnamese-only conditions may be missed.
    """
    if not drugs:
        return []
    terms = _condition_terms(patient_context)
    if not terms:
        return []

    alerts: list[dict] = []
    seen: set[tuple] = set()
    for drug in (d.strip() for d in drugs if d and d.strip()):
        paras = get_contraindication_text(drug)
        if not paras:
            continue
        text = "\n".join(paras)
        for term, label, whole in terms:
            key = (_norm(drug), label)
            if key in seen:
                continue
            snippet = find_mention(text, term, whole_word=whole)
            if snippet:
                seen.add(key)
                alerts.append({
                    "type": "contraindication",
                    "drug": drug,
                    "condition": label,
                    "matched": term,
                    "snippet": snippet,
                })
    return alerts


def format_alerts(alerts: list[dict]) -> str:
    """Render alerts as the leading block of a response (allergy first)."""
    if not alerts:
        return ""
    allergy = [a for a in alerts if a.get("type", "allergy") == "allergy"]
    contra = [a for a in alerts if a.get("type") == "contraindication"]
    interaction = [a for a in alerts if a.get("type") == "interaction"]

    blocks = []
    if allergy:
        lines = ["⚠️ CẢNH BÁO DỊ ỨNG:"]
        for al in allergy:
            extra = (f" — phản ứng đã ghi nhận: {al['reaction']}"
                     if al.get("reaction") else "")
            crit = f" [{al['criticality']}]" if al.get("criticality") else ""
            lines.append(f"  • {al['drug']} xung đột với dị ứng {al['allergen']}"
                         f" ({al['match']}){crit}{extra}")
        blocks.append("\n".join(lines))
    if contra:
        lines = ["⚠️ CẢNH BÁO CHỐNG CHỈ ĐỊNH (nguồn: nhãn thuốc FDA/OpenFDA):"]
        for ct in contra:
            lines.append(f"  • {ct['drug']} chống chỉ định liên quan"
                         f" {ct['condition']}: {ct['snippet']}")
        blocks.append("\n".join(lines))
    if interaction:
        lines = ["⚠️ CẢNH BÁO TƯƠNG TÁC THUỐC (nguồn: nhãn thuốc FDA/OpenFDA):"]
        for it in interaction:
            lines.append(f"  • {it['drug_a']} ⇄ {it['drug_b']}"
                         f" ({it['other_source']}): {it['snippet']}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
