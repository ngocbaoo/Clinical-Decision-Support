"""
R3 — Safety gate: allergy conflict check (Safety Req #2 — alert renders FIRST).

Matches drugs mentioned in the query (router entities) against the patient's
AllergyIntolerance list, including basic cross-reactivity groups (a Penicillin
allergy must flag Amoxicillin). Pure functions, no network.

Drug-interaction checking (F-RAG-05) is a stub hook — wire DrugBank/RxNav later.
"""

import sys
import unicodedata
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ on sys.path

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
                    "drug": drug,
                    "allergen": allergen,
                    "criticality": a.get("criticality"),
                    "reaction": a.get("reaction"),
                    "match": "direct" if direct else f"cross-reactivity ({d_group})",
                })
    return alerts


def check_drug_interactions(drugs: list[str], patient_context: dict) -> list[dict]:
    """F-RAG-05 stub — returns [] until DrugBank/RxNav is wired in."""
    return []


def format_alerts(alerts: list[dict]) -> str:
    """Render alerts as the leading block of a response."""
    if not alerts:
        return ""
    lines = ["⚠️ CẢNH BÁO DỊ ỨNG:"]
    for al in alerts:
        extra = f" — phản ứng đã ghi nhận: {al['reaction']}" if al.get("reaction") else ""
        crit = f" [{al['criticality']}]" if al.get("criticality") else ""
        lines.append(f"  • {al['drug']} xung đột với dị ứng {al['allergen']}"
                     f" ({al['match']}){crit}{extra}")
    return "\n".join(lines)
