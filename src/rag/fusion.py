"""
Reciprocal Rank Fusion (RRF) + comorbidity-aware query expansion.

Why this exists: retrieval is driven by the user's QUERY, so it is blind to the patient's
background pathology. A "sốt cao → sốc nhiễm khuẩn" query pulls the generic sepsis fluid-bolus
protocol but never the "bù dịch thận trọng ở bệnh nhân xơ gan" caveat — even though that caveat
IS in the corpus. The generator then can't reconcile the protocol against a comorbidity it never
saw. We fix this at retrieval: run the primary query AND an auxiliary query per active comorbidity,
then fuse the ranked lists with RRF so comorbidity-relevant chunks can enter the top-K WITHOUT
displacing the primary intent's chunks.

RRF (Cormack et al. 2009) fuses by RANK, not by raw score, so lists from different queries (whose
score scales differ) combine fairly: score(d) = Σ_lists w_list / (k + rank_d_in_list). A document
near the top of any list gets a large contribution; appearing in several lists compounds. We weight
the PRIMARY list higher than the auxiliary (comorbidity) lists so the original intent's recall is
protected — the auxiliary lists can only add context, not evict a strongly-primary chunk.

Pure functions, no I/O — unit-testable offline (no Chroma / embedding API / torch).
"""

from __future__ import annotations

from typing import Callable, Iterable


def _default_key(item: dict) -> str:
    """Identity for de-duplication across lists: the chunk text prefix (chunks have no stable id
    in the retrieval payload). 120 chars is enough to distinguish distinct guideline chunks while
    tolerating identical leading boilerplate being trimmed differently."""
    return (item.get("text") or "")[:120]


def rrf_fuse(ranked_lists: list[list[dict]], *, k: int = 60,
             weights: list[float] | None = None, n_results: int | None = None,
             key: Callable[[dict], str] = _default_key) -> list[dict]:
    """Weighted Reciprocal Rank Fusion of several ranked result lists.

    - `k` damps the rank curve (standard default 60): larger k flattens the contribution of top
      ranks, so deeper hits still matter.
    - `weights[i]` scales list i's contribution (default 1.0 each). Give the primary list a larger
      weight than auxiliary lists to protect the primary intent's recall.
    - De-dups by `key`; a chunk appearing in multiple lists accumulates each list's contribution.
    - Returns the fused list (highest score first), truncated to `n_results` if given. Each output
      item is the ORIGINAL dict from whichever list it first appeared in (score/chunk_type intact).
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    scores: dict[str, float] = {}
    rep: dict[str, dict] = {}
    for lst, w in zip(ranked_lists, weights):
        for rank, item in enumerate(lst, start=1):
            kk = key(item)
            scores[kk] = scores.get(kk, 0.0) + w / (k + rank)
            rep.setdefault(kk, item)
    order = sorted(scores, key=lambda kk: scores[kk], reverse=True)
    fused = [rep[kk] for kk in order]
    return fused[:n_results] if n_results is not None else fused


def comorbidity_fuse(primary: list[dict], aux_lists: list[list[dict]], *, n_results: int,
                     comorbidity_slots: int, k: int = 60, min_score: float = 0.0,
                     key: Callable[[dict], str] = _default_key) -> list[dict]:
    """Add comorbidity context to the primary intent list under an ABSOLUTE recall guarantee.

    The primary intent KEEPS its full top-`n_results` budget — comorbidity chunks are APPENDED, not
    substituted, so the primary list returned is byte-identical to baseline and recall cannot drop
    by construction. (An earlier "reserve a slot within n_results" design displaced the primary
    rank-`n_results` chunk; for a query whose key chunk sat exactly there — e.g. the sepsis protocol
    at rank 4 for a fever query — that silently deleted the answer's core. Never take from the
    primary budget.)

    RRF is used where it shines: ranking the comorbidity candidates across the several
    per-comorbidity lists, so the single best comorbidity chunk (high in multiple lists) is chosen.
    A candidate must clear `min_score` and not duplicate a primary chunk; up to `comorbidity_slots`
    are appended. Result length is n_results .. n_results + comorbidity_slots (bounded prefill cost,
    only for patients who actually have comorbidities).
    """
    base = primary[:n_results]
    if not aux_lists or comorbidity_slots <= 0:
        return base
    result = list(base)
    seen = {key(c) for c in result}
    added = 0
    for c in rrf_fuse(aux_lists, k=k, key=key):     # best comorbidity chunks by RRF across lists
        if added >= comorbidity_slots:
            break
        if c.get("score", 0.0) >= min_score and key(c) not in seen:
            seen.add(key(c))
            result.append(c)
            added += 1
    return result


def comorbidity_names(patient_context: dict | None) -> list[str]:
    """Active comorbidity names from a patient context, de-duplicated, order-preserving.

    General over patients: reads `conditions[*]` and prefers the Vietnamese name, falling back to
    display / English / ICD-10. Returns [] when there is no context (e.g. eval queries with no
    patient), which makes the caller short-circuit to plain retrieval — no behaviour change.
    """
    names: list[str] = []
    seen: set[str] = set()
    for c in (patient_context or {}).get("conditions", []) or []:
        nm = (c.get("name_vi") or c.get("display") or c.get("name_en")
              or c.get("icd10_code") or "").strip()
        kk = nm.lower()
        if nm and kk not in seen:
            seen.add(kk)
            names.append(nm)
    return names


def comorbidity_queries(names: Iterable[str], template: str, max_n: int) -> list[str]:
    """Turn comorbidity names into auxiliary retrieval queries via `template` (with a `{cond}`
    field), capped at `max_n` to bound the extra embedding/search cost. General — no condition is
    special-cased; the template biases each query toward management/caution content for that
    comorbidity (the embedding of the resulting phrase pulls the relevant guideline chunks)."""
    out: list[str] = []
    for nm in names:
        nm = (nm or "").strip()
        if not nm:
            continue
        out.append(template.format(cond=nm))
        if len(out) >= max_n:
            break
    return out
