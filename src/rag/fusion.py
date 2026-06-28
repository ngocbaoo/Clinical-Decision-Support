"""Reciprocal Rank Fusion (RRF) + comorbidity-aware query expansion."""

from __future__ import annotations

from typing import Callable, Iterable


def _default_key(item: dict) -> str:
    return (item.get("text") or "")[:120]


def rrf_fuse(ranked_lists: list[list[dict]], *, k: int = 60,
             weights: list[float] | None = None, n_results: int | None = None,
             key: Callable[[dict], str] = _default_key) -> list[dict]:
    """Weighted Reciprocal Rank Fusion of several ranked result lists."""
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
    """Append up to `comorbidity_slots` best comorbidity chunks (RRF across aux lists) to the full
    primary top-`n_results`, under an absolute recall guarantee (primary budget untouched)."""
    base = primary[:n_results]
    if not aux_lists or comorbidity_slots <= 0:
        return base
    result = list(base)
    seen = {key(c) for c in result}
    added = 0
    for c in rrf_fuse(aux_lists, k=k, key=key):
        if added >= comorbidity_slots:
            break
        if c.get("score", 0.0) >= min_score and key(c) not in seen:
            seen.add(key(c))
            result.append(c)
            added += 1
    return result


def comorbidity_names(patient_context: dict | None) -> list[str]:
    """Active comorbidity names from a patient context, de-duplicated, order-preserving."""
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
    """Turn comorbidity names into auxiliary retrieval queries via `template` ({cond}), capped at
    `max_n`."""
    out: list[str] = []
    for nm in names:
        nm = (nm or "").strip()
        if not nm:
            continue
        out.append(template.format(cond=nm))
        if len(out) >= max_n:
            break
    return out
