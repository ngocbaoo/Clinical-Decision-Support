"""
Canonical ICU drug lexicon — single source of truth for the ASR drug matcher and prompt biasing.

Seeded from src/rag/safety.py ALLERGY_GROUPS (cross-reactivity members) plus the drug names used in
the curated test set (src/asr/eval/drug_test_cases.py). Kept as canonical English INN spellings;
the matcher de-diacritics + fuzzy-matches against these, so morphological/phonetic garbles from ASR
still resolve here. Extend freely — this is just a name list, no network.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ on sys.path
from rag.safety import ALLERGY_GROUPS  # noqa: E402

# Common ICU drugs beyond the allergy cross-reactivity groups (vasopressors, sedation, antibiotics,
# anticoagulants, etc.) — covers the curated test set and typical bedside questions.
_ICU_DRUGS = [
    "Vancomycin", "Gentamicin", "Amikacin", "Tobramycin",
    "Norepinephrine", "Adrenalin", "Epinephrine", "Dopamine", "Dobutamine", "Vasopressin",
    "Meropenem", "Imipenem", "Ceftriaxone", "Cefepime", "Ceftazidime", "Cefotaxime",
    "Piperacillin", "Tazobactam", "Amoxicillin", "Ampicillin", "Penicillin",
    "Levofloxacin", "Ciprofloxacin", "Moxifloxacin", "Metronidazole", "Azithromycin", "Linezolid",
    "Colistin", "Sulfamethoxazole", "Trimethoprim",
    "Propofol", "Midazolam", "Fentanyl", "Morphine", "Ketamine", "Dexmedetomidine",
    "Heparin", "Enoxaparin", "Warfarin",
    "Insulin", "Furosemide", "Dexamethasone", "Hydrocortisone", "Amiodarone", "Digoxin",
    "Phenytoin", "Levetiracetam", "Fluconazole", "Amphotericin", "Paracetamol", "Ketorolac",
    "Noradrenaline",
]


# Group-name placeholders inside ALLERGY_GROUPS that are drug *classes*, not single agents — they
# would only create false fuzzy matches, so they are excluded from the matcher lexicon.
_CLASS_PLACEHOLDERS = {"nsaid", "aminoglycoside", "quinolone", "sulfonamide", "sulfa",
                       "cephalosporin"}


def _build_lexicon() -> list[str]:
    seen: dict[str, str] = {}  # lowercase -> canonical, de-duplicated, order-stable
    for name in _ICU_DRUGS:
        seen.setdefault(name.lower(), name)
    for members in ALLERGY_GROUPS.values():
        for m in members:  # lowercase substrings; keep real agents, drop class placeholders
            if m not in _CLASS_PLACEHOLDERS:
                seen.setdefault(m, m.capitalize())
    return list(seen.values())


DRUG_LEXICON: list[str] = _build_lexicon()

# Whisper initial_prompt must stay short: the full lexicon (~371 tokens) overflows the model's
# 448-token decoder budget, and long prompts dilute biasing anyway. Use a focused list of the most
# common ICU drugs (vasopressors, sedation, key antibiotics, anticoagulants).
_PROMPT_LIST = [
    "Vancomycin", "Gentamicin", "Meropenem", "Ceftriaxone", "Piperacillin", "Levofloxacin",
    "Norepinephrine", "Adrenalin", "Dopamine", "Dobutamine",
    "Propofol", "Midazolam", "Fentanyl", "Morphine",
    "Heparin", "Warfarin", "Insulin", "Furosemide", "Amiodarone", "Dexamethasone",
]
PROMPT_DRUGS: str = ", ".join(_PROMPT_LIST)
