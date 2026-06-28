"""
Deterministic comorbidity-conflict enforcement gate (Risk #1 backstop).

The retrieval (#1) + generator-reconcile (#3) levers make the answer MENTION a comorbidity, but they
can only ground a caution on text that was retrieved — so a dangerous recommendation can still slip
through grounded on a generic protocol (the "bù dịch 1-2L" sepsis bolus stays even for a cirrhotic
patient, because the corpus has no "cautious fluids in cirrhosis" sentence to cite). This gate is the
deterministic safety net: a small, high-confidence rules table of (dangerous recommendation ×
conflicting comorbidity). When the FINAL answer contains a flagged recommendation AND the patient has
the conflicting comorbidity, we attach a mandatory warning. It does NOT delete the answer (the
needed care — e.g. sepsis management — stays); it forces the modification the doctor must make. The
DISCLAIMER ("Cần bác sĩ xác nhận…") still applies.

Pure text/Context matching — no LLM, no torch, offline-testable. Curated CONSERVATIVELY: a missed
conflict is the status quo, but a false alarm erodes trust, so only clinically unambiguous pairs.
"""

import unicodedata


def _norm(s: str) -> str:
    """Lowercase, đ→d, strip diacritics — so 'Suy gan' / 'suy gan' / 'SUY GAN' all match, robust to
    the answer's formatting."""
    s = (s or "").lower().replace("đ", "d")
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


# Each rule: a flagged recommendation (any of `rec`) that is unsafe for a patient carrying any of
# `cond`. `message` is the mandatory warning ({cond} = the patient's matched condition name).
# Keywords are written with diacritics for readability; matched diacritic-insensitively.
RULES = [
    {
        "id": "aggressive_fluids",
        "rec": ["bù dịch nhanh", "truyền dịch nhanh", "1000-2000", "1000 - 2000", "30ml/kg",
                "30 ml/kg", "bolus dịch", "test truyền dịch", "hồi sức dịch", "bù dịch tích cực"],
        "cond": ["suy gan", "xơ gan", "bệnh não gan", "suy tim", "phù phổi", "esrd",
                 "suy thận", "bệnh thận mạn"],
        "severity": "critical",
        "message": ("Bù dịch tích cực cần THẬN TRỌNG ở bệnh nhân {cond}: nguy cơ quá tải dịch / phù "
                    "phổi (và vỡ giãn tĩnh mạch nếu xơ gan). Cá thể hóa thể tích và tốc độ, theo dõi "
                    "sát đáp ứng."),
    },
    {
        "id": "hepatotoxic_paracetamol",
        "rec": ["paracetamol", "acetaminophen", "efferalgan", "hạ sốt"],
        "cond": ["suy gan", "xơ gan", "bệnh gan", "bệnh não gan"],
        "severity": "critical",
        "message": ("Paracetamol có độc tính gan: ở bệnh nhân {cond} cần TRÁNH hoặc giảm liều sâu và "
                    "theo dõi men gan; hỏi dược sĩ lâm sàng trước khi dùng."),
    },
    {
        "id": "nephrotoxic_nsaid",
        "rec": ["nsaid", "ibuprofen", "diclofenac", "kháng viêm không steroid"],
        "cond": ["suy thận", "bệnh thận mạn", "esrd", "suy gan", "xơ gan"],
        "severity": "critical",
        "message": ("NSAID gây độc thận / giữ muối nước: ở bệnh nhân {cond} cần TRÁNH; cân nhắc thuốc "
                    "giảm đau thay thế."),
    },
]


def _patient_conditions(patient_context: dict) -> list[str]:
    out = []
    for c in (patient_context or {}).get("conditions", []) or []:
        nm = (c.get("name_vi") or c.get("display") or c.get("name_en") or "").strip()
        if nm:
            out.append(nm)
    return out


def check_comorbidity_conflicts(answer: str, patient_context: dict) -> list[dict]:
    """Return one conflict per fired rule: {id, severity, comorbidity, message}. Deterministic."""
    na = _norm(answer)
    if not na:
        return []
    conds = [(_norm(nm), nm) for nm in _patient_conditions(patient_context)]
    conflicts = []
    for rule in RULES:
        if not any(_norm(k) in na for k in rule["rec"]):
            continue
        hit = next((orig for ncond, orig in conds
                    if any(_norm(ck) in ncond for ck in rule["cond"])), None)
        if hit:
            conflicts.append({"id": rule["id"], "severity": rule["severity"],
                              "comorbidity": hit, "message": rule["message"].format(cond=hit)})
    return conflicts


def apply_comorbidity_gate(response: dict, patient_context: dict) -> dict:
    """If the (non-fallback) answer carries a flagged recommendation conflicting with the patient's
    comorbidity, prepend a mandatory warning banner and raise alerts. Returns a NEW dict; a no-op
    when there are no conflicts or the response is a safety fallback."""
    if response.get("fallback"):
        return response
    conflicts = check_comorbidity_conflicts(response.get("answer", ""), patient_context)
    if not conflicts:
        return response
    banner = ("⚠️ LƯU Ý BỆNH NỀN — kiểm tra trước khi thực hiện:\n"
              + "\n".join(f"• {c['message']}" for c in conflicts))
    out = dict(response)
    out["answer"] = f"{banner}\n\n{response.get('answer', '')}"
    out["alerts"] = list(response.get("alerts") or []) + [
        {"type": "comorbidity_conflict", "drug": None, **c} for c in conflicts]
    out["comorbidity_conflicts"] = conflicts
    return out
