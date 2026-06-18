"""
Post-ASR drug-name recovery — SUGGEST-only.

ASR errors on drug names are near-miss phonetic garbles, not omissions ("vancomy sin"→Vancomycin,
"profile"→Propofol). This module fuzzy-matches transcript spans against the ICU lexicon and proposes
the intended drug, so the confirm-box UI can offer the doctor the right name to accept. It NEVER
rewrites the transcript: auto-correcting a drug name is exactly the Risk #1 we guard against — the
doctor confirms. rapidfuzz only (no new dependency).
"""

import re
import sys
from pathlib import Path

from rapidfuzz import fuzz, process

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ on sys.path
from asr.drug_lexicon import DRUG_LEXICON  # noqa: E402
from asr.eval.metrics import _strip_diacritics  # noqa: E402

# Tuning (calibrate on the real-voice probe). WRatio scores observed garbles 67-88; the lone false
# match ("morphine" for "maripinum") scored 59, so a floor of 65 plus a top1-top2 margin separates
# true recoveries from junk.
SCORE_FLOOR = 65.0
MARGIN = 6.0          # top1 must beat top2 by this to be unambiguous; else alternatives are returned
MAX_ALTERNATIVES = 3

_LEX_NORM = [_strip_diacritics(d) for d in DRUG_LEXICON]
_NORM_TO_CANON = dict(zip(_LEX_NORM, DRUG_LEXICON))


def _clean(tok: str) -> str:
    """Drop leading/trailing punctuation from a token so 'fenteno,' matches like 'fenteno'."""
    return re.sub(r"^\W+|\W+$", "", tok)


def _spans(transcript: str):
    """Candidate spans: each token and each adjacent token-pair (joined, to catch split garbles
    like 'vancomy sin'). Yields (surface, start_tok, end_tok, normalized_query)."""
    raw = transcript.split()
    toks = [_clean(t) for t in _strip_diacritics(transcript).split()]
    for i, t in enumerate(toks):
        yield raw[i], i, i, t
    for i in range(len(toks) - 1):
        yield f"{raw[i]} {raw[i+1]}", i, i + 1, toks[i] + toks[i + 1]


def suggest_drugs(transcript: str, score_floor: float = SCORE_FLOOR, margin: float = MARGIN):
    """Propose drug names for fuzzy-matching spans. Returns a list of
    {span, start, end, suggestion, score, alternatives} — highest score first, one per drug.
    Pure: does not modify the transcript."""
    best_per_drug: dict[str, dict] = {}
    for surface, start, end, query in _spans(transcript):
        if len(query) < 4:  # too short to be a reliable drug match
            continue
        ranked = process.extract(query, _LEX_NORM, scorer=fuzz.WRatio, limit=MAX_ALTERNATIVES + 1)
        if not ranked or ranked[0][1] < score_floor:
            continue
        top_norm, top_score, _ = ranked[0]
        suggestion = _NORM_TO_CANON[top_norm]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        alts = ([_NORM_TO_CANON[n] for n, s, _ in ranked[1:] if s >= score_floor]
                if (top_score - second) < margin else [])
        cand = {"span": surface, "start": start, "end": end, "suggestion": suggestion,
                "score": round(top_score, 1), "alternatives": alts}
        # keep the strongest span per suggested drug
        prev = best_per_drug.get(suggestion)
        if prev is None or cand["score"] > prev["score"]:
            best_per_drug[suggestion] = cand
    return sorted(best_per_drug.values(), key=lambda c: -c["score"])
