"""
Scoring helpers for the ASR bake-off — pure functions, NO torch (so tests stay offline/fast).

- normalize_vi: shared text normalizer for fair WER/CER (NFC, lowercase, strip punctuation).
- wer / cer: word- and character-error-rate via jiwer.
- drug_hits: the F-ASR-03 metric — did the transcript contain each expected drug name?
  Matching is diacritic-insensitive substring (same spirit as src/rag/safety._norm), because a
  clinician cares that "Vancomycin" was heard, not that surrounding words matched exactly.
"""

import re
import unicodedata

import jiwer


def normalize_vi(s: str) -> str:
    """NFC + lowercase + collapse punctuation/whitespace. Diacritics PRESERVED (WER is on VN text)."""
    s = unicodedata.normalize("NFC", (s or "").lower())
    s = re.sub(r"[^\w\sàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]",
               " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _strip_diacritics(s: str) -> str:
    """Lowercase + drop diacritics — tolerant matching for drug names (mirror of safety._norm)."""
    s = unicodedata.normalize("NFD", (s or "").lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def wer(reference: str, hypothesis: str) -> float:
    return jiwer.wer(normalize_vi(reference), normalize_vi(hypothesis))


def cer(reference: str, hypothesis: str) -> float:
    return jiwer.cer(normalize_vi(reference), normalize_vi(hypothesis))


def drug_hits(hypothesis: str, drugs: list[str]) -> list[dict]:
    """For each expected drug, whether its name appears in the transcript (diacritic-insensitive)."""
    h = _strip_diacritics(hypothesis)
    return [{"drug": d, "hit": _strip_diacritics(d) in h} for d in drugs]


def recoverable_hits(hypothesis: str, drugs: list[str]) -> list[dict]:
    """The gate KPI: would the matcher OFFER each expected drug to the doctor? A drug is
    'recoverable' if it appears verbatim OR the matcher suggests it (as top pick or an alternative)
    for some span. Import is local to keep this module torch-free and avoid an import cycle."""
    from asr.drug_match import suggest_drugs

    raw = {h["drug"]: h["hit"] for h in drug_hits(hypothesis, drugs)}
    offered = set()
    for s in suggest_drugs(hypothesis):
        offered.add(_strip_diacritics(s["suggestion"]))
        offered.update(_strip_diacritics(a) for a in s["alternatives"])
    return [{"drug": d, "verbatim": raw[d],
             "recoverable": raw[d] or _strip_diacritics(d) in offered} for d in drugs]
