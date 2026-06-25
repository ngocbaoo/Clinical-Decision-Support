"""
Offline ASR tests — metrics + the curated test set only. NO torch / no model download, so these
run in CI and keep the no-torch core honest (importing the transcriber is what pulls in torch).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asr.config import DECODE  # noqa: E402
from asr.drug_match import suggest_drugs  # noqa: E402
from asr.eval.drug_test_cases import DRUG_TEST_CASES  # noqa: E402
from asr.eval.metrics import cer, drug_hits, normalize_vi, recoverable_hits, wer  # noqa: E402


def _suggested(transcript):
    """All drugs the matcher would offer (top picks + alternatives), lowercased."""
    out = set()
    for s in suggest_drugs(transcript):
        out.add(s["suggestion"].lower())
        out.update(a.lower() for a in s["alternatives"])
    return out


def test_curated_set_has_20_cases_each_with_a_drug():
    assert len(DRUG_TEST_CASES) >= 20
    ids = [c["id"] for c in DRUG_TEST_CASES]
    assert len(ids) == len(set(ids)), "duplicate case ids"
    for c in DRUG_TEST_CASES:
        assert c["drugs"], f"{c['id']} has no expected drug"
        # ground-truth sanity: every expected drug name actually appears in its own sentence
        assert all(h["hit"] for h in drug_hits(c["text"], c["drugs"])), c["id"]


def test_normalize_vi_lowercases_and_strips_punctuation():
    assert normalize_vi("Vancomycin, 1g!") == "vancomycin 1g"
    assert normalize_vi("  Bệnh   nhân.  ") == "bệnh nhân"


def test_wer_cer_perfect_and_imperfect():
    assert wer("bệnh nhân dùng vancomycin", "bệnh nhân dùng vancomycin") == 0.0
    assert cer("vancomycin", "vancomycin") == 0.0
    assert wer("dùng vancomycin", "dùng gentamicin") == 0.5  # 1 of 2 words wrong


def test_drug_hits_diacritic_insensitive_and_detects_miss():
    hits = drug_hits("bệnh nhân dùng VANCOMYCIN nhưng không có thuốc kia", ["Vancomycin", "Gentamicin"])
    by = {h["drug"]: h["hit"] for h in hits}
    assert by["Vancomycin"] is True   # case-insensitive substring
    assert by["Gentamicin"] is False  # genuinely absent


# --- matcher (the post-ASR drug-recovery layer) ---------------------------------------------

def test_matcher_recovers_observed_garbles():
    # garbles actually produced by whisper-small on the TTS set -> expected drug
    cases = {
        "bệnh nhân đang dùng vancomy sin một gam": "vancomycin",
        "đang chuyển profile và fenteno": "propofol",
        "điều chỉnh liều merupinum": "meropenem",
        "liều nói rapanephrin hiện tại": "norepinephrine",
    }
    for transcript, expected in cases.items():
        assert expected in _suggested(transcript), (transcript, expected)


def test_matcher_is_suggest_only_does_not_mutate():
    t = "đang chuyển profile và fenteno"
    suggest_drugs(t)
    assert t == "đang chuyển profile và fenteno"  # input untouched


def test_matcher_rejects_non_drug_words():
    # ordinary clinical words must not be proposed as drugs
    sugg = _suggested("bệnh nhân tỉnh táo huyết áp ổn định nhịp thở đều")
    assert not (sugg & {"morphine", "dopamine", "heparin", "insulin"})


def test_recoverable_recall_counts_verbatim_and_matched():
    rec = recoverable_hits("đang dùng vancomy sin và profile", ["Vancomycin", "Propofol"])
    by = {r["drug"]: r for r in rec}
    assert by["Vancomycin"]["verbatim"] is False      # garbled, not verbatim
    assert by["Vancomycin"]["recoverable"] is True     # but the matcher offers it
    assert by["Propofol"]["recoverable"] is True


# --- decoder config (the experiment-chosen production decoding; no torch) --------------------

def test_decode_config_is_well_formed():
    # gate the production decoder shape so a bad edit fails offline, before any model loads.
    # These are faster-whisper (CTranslate2) params now — no torch/attn backend.
    assert isinstance(DECODE, dict)
    assert DECODE["compute_type"] in {"int8", "int8_float16", "int8_bfloat16", "float16", "float32"}
    assert isinstance(DECODE["beam_size"], int) and DECODE["beam_size"] >= 1
    temp = DECODE.get("temperature", 0.0)  # scalar (greedy) or fallback list/tuple
    assert isinstance(temp, (int, float, list, tuple)), "temperature: scalar or fallback sequence"
    if isinstance(temp, (list, tuple)):
        assert temp[0] == 0.0 and all(t >= 0 for t in temp)  # fallback starts greedy
