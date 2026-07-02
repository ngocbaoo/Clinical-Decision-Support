"""
OpenFDA drug-label client for drug-drug interaction screening (F-RAG-05).

OpenFDA serves structured FDA drug labels at /drug/label.json. Each label may
carry a free-text `drug_interactions` section. We fetch the label for a drug
and scan that section for mentions of the patient's *other* drugs; a match is
surfaced as an interaction alert.

Caveats (documented on purpose):
  - Labels are unstructured prose and coverage is uneven, so this is a
    screening aid, not a curated pairwise interaction database. A missing
    alert is NOT a guarantee of safety.
  - OpenFDA indexes US labels by English generic/brand names; Vietnamese or
    locally-branded drug names will often miss.
  - Every failure (timeout, rate-limit, unknown drug, network down) degrades
    to "no interaction text" so the pipeline never crashes or blocks.

No new dependency: uses urllib from the stdlib. An optional OPENFDA_API_KEY in
the environment raises the rate limit (240/min vs. 40/min) but is not required.
"""

import json
import os
import re
import sys
import threading
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ on sys.path
from paths import DATA_DIR  # noqa: E402

OPENFDA_LABEL_URL = "https://api.fda.gov/drug/label.json"
_TIMEOUT_S = 6.0

# Disk-persisted cache so the OpenFDA lookups survive restarts — the safety stage's biggest tail
# was a cold api.fda.gov round-trip (p95 ~6.8s) on first-seen drugs. Pre-warm with prewarm() /
# `python src/rag/openfda.py`. Gitignored (derived data).
_CACHE_FILE = DATA_DIR / "openfda_cache.json"
_lock = threading.Lock()  # guards the on-disk write (safety runs in its own pipeline thread now)

# Label sections that carry interaction prose. The structured prescription
# field is `drug_interactions`; OTC monographs often lack it, so we also accept
# the combined lab-test variant. (We deliberately skip the huge, noisy
# `warnings` section to avoid spurious name matches.)
_INTERACTION_FIELDS = ("drug_interactions",
                       "drug_and_or_laboratory_test_interactions")

# Structured contraindications section of the label.
_CONTRAINDICATION_FIELDS = ("contraindications",)

# Caches: normalized drug name -> section paragraphs. Loaded from disk at import and re-saved on
# every new fetch, so a warmed cache survives process restarts (not just one session).
_interaction_cache: dict[str, list[str]] = {}
_contraindication_cache: dict[str, list[str]] = {}


def _load_cache() -> None:
    """Populate the in-memory caches from the persisted file (best-effort)."""
    try:
        if _CACHE_FILE.is_file():
            data = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
            _interaction_cache.update(data.get("interaction", {}))
            _contraindication_cache.update(data.get("contraindication", {}))
    except Exception:  # noqa: BLE001 — a corrupt cache must never break startup
        pass


def _persist() -> None:
    """Atomically write both caches to disk (best-effort; never breaks a request)."""
    try:
        with _lock:
            _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp = _CACHE_FILE.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({"interaction": _interaction_cache,
                                       "contraindication": _contraindication_cache},
                                      ensure_ascii=False), encoding="utf-8")
            tmp.replace(_CACHE_FILE)
    except Exception:  # noqa: BLE001
        pass


def _norm(s: str) -> str:
    """Lowercase + strip Vietnamese diacritics for tolerant matching."""
    s = unicodedata.normalize("NFD", (s or "").lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn").strip()


# British/INN/locally-used drug spellings -> US FDA generic names, so OpenFDA's
# US-indexed labels (keyed on `openfda.generic_name`) actually resolve. Without
# this, "paracetamol" returns 0 results (FDA indexes it as "acetaminophen") and
# the whole contraindication/interaction screen silently no-ops. Keys and values
# are _norm()-folded (lowercase, no diacritics); unknown drugs fall through
# unchanged. Extend freely — pure data, no network.
_FDA_GENERIC_ALIASES = {
    "paracetamol": "acetaminophen",
    "adrenaline": "epinephrine",
    "adrenalin": "epinephrine",
    "noradrenaline": "norepinephrine",
    "noradrenalin": "norepinephrine",
    "salbutamol": "albuterol",
    "frusemide": "furosemide",
    "lignocaine": "lidocaine",
    "pethidine": "meperidine",
    "rifampicin": "rifampin",
    "thiopentone": "thiopental",
    "phenobarbitone": "phenobarbital",
    "isoprenaline": "isoproterenol",
    "glyceryl trinitrate": "nitroglycerin",
    "amphotericin": "amphotericin b",
    "cotrimoxazole": "sulfamethoxazole and trimethoprim",
    "co-trimoxazole": "sulfamethoxazole and trimethoprim",
}


def to_fda_generic(drug: str) -> str:
    """Map a drug name to its US FDA generic spelling for OpenFDA lookup/matching
    (_norm-folded). A no-op for names already in US form or unknown."""
    return _FDA_GENERIC_ALIASES.get(_norm(drug), _norm(drug))


def _fetch(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "vsf-rag/1.0"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_label_sections(drug: str, fields: tuple[str, ...],
                        cache: dict[str, list[str]]) -> list[str]:
    """Return the paragraphs of `fields` for `drug` ([] if none/unknown).

    Requires a label that actually carries one of `fields` (_exists_), else the
    API returns e.g. an OTC monograph without it. Tries the generic-name index
    first, then brand name. Any error (HTTP 404 for unknown drug, timeout,
    rate-limit) yields [] and is cached so we don't hammer the API.
    """
    key = to_fda_generic(drug)  # paracetamol -> acetaminophen, etc.
    if not key:
        return []
    if key in cache:
        return cache[key]

    api_key = os.getenv("OPENFDA_API_KEY")
    exists = " OR ".join(f"_exists_:{f}" for f in fields)
    paras: list[str] = []
    for name_field in ("openfda.generic_name", "openfda.brand_name"):
        params = {"search": f'{name_field}:"{key}" AND ({exists})', "limit": "1"}
        if api_key:
            params["api_key"] = api_key
        url = f"{OPENFDA_LABEL_URL}?{urllib.parse.urlencode(params)}"
        try:
            data = _fetch(url)
        except Exception:
            continue  # unknown drug (404) / transient error -> try next field
        results = data.get("results") or []
        if not results:
            continue
        for fld in fields:
            val = results[0].get(fld) or []
            if isinstance(val, str):
                val = [val]
            paras.extend(p for p in val if p)
        if paras:
            break

    cache[key] = paras
    _persist()  # keep the warmed entry across restarts
    return paras


def get_interaction_text(drug: str) -> list[str]:
    """`drug_interactions` paragraphs for `drug` ([] if none/unknown)."""
    return _get_label_sections(drug, _INTERACTION_FIELDS, _interaction_cache)


def get_contraindication_text(drug: str) -> list[str]:
    """`contraindications` paragraphs for `drug` ([] if none/unknown)."""
    return _get_label_sections(drug, _CONTRAINDICATION_FIELDS,
                               _contraindication_cache)


def prewarm(drugs) -> int:
    """Fetch + cache labels for `drugs` (e.g. the ICU lexicon) so first live requests are warm.
    Returns the number of newly-fetched drugs. Safe to re-run (skips already-cached)."""
    fetched = 0
    for d in drugs:
        key = _norm(d)
        if not key or (key in _interaction_cache and key in _contraindication_cache):
            continue
        get_interaction_text(d)
        get_contraindication_text(d)
        fetched += 1
    return fetched


_load_cache()  # warm the in-memory caches from disk at import


def find_mention(text: str, term: str, whole_word: bool = True) -> str | None:
    """If `term` appears in `text`, return a trimmed snippet around it.

    Matches on the normalized term (whole word by default, to avoid spurious
    substring hits like 'pin' in 'Aspirin'); pass whole_word=False for stems
    (e.g. 'pregnan' to catch pregnant/pregnancy). Returns the sentence
    containing the first match, capped to 240 chars.
    """
    name = _norm(term)
    if len(name) < 4:
        return None
    pattern = rf"\b{re.escape(name)}\b" if whole_word else re.escape(name)
    if not re.search(pattern, _norm(text)):
        return None
    for sentence in re.split(r"(?<=[.;])\s+", text):
        if re.search(pattern, _norm(sentence)):
            snippet = sentence.strip()
            return snippet[:240] + ("…" if len(snippet) > 240 else "")
    return None


if __name__ == "__main__":  # one-off: warm the on-disk cache for the ICU drug lexicon
    from asr.drug_lexicon import DRUG_LEXICON  # plain string list, no torch
    n = prewarm(DRUG_LEXICON)
    print(f"[openfda] prewarmed {n} new drug(s); cache -> {_CACHE_FILE} "
          f"({len(_interaction_cache)} interaction / {len(_contraindication_cache)} contraindication)")
