"""
Clinical calculations for the ICU Clinical Assistant.

Input:  patient_context dict from FHIRClient.build_patient_context()
Output: calculations dict with MAP, qSOFA, SOFA, NEWS2, eGFR + a summary of
        alerts, ready to attach to the patient context for the RAG pipeline.

Core logic reads only from the input dict (no hardcoded patient data, no HTTP),
so calculate_all() is reusable independent of the FHIR layer. The CLI wraps
FHIRClient for convenience.

Run:
    python src/scoring/calculator.py --file data/mock/patient_A.json
    python src/scoring/calculator.py --file data/mock/patient_A.json --json
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Windows consoles default to cp1252 and choke on Vietnamese — force UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ on sys.path
from fhir.fhir_client import FHIRClient  # noqa: E402


# ===========================================================================
# Task 2 — Helper functions
# ===========================================================================
def get_obs_value(observations: dict, key: str):
    """Safely extract an observation value (may be number, string, or None)."""
    obs = observations.get(key, {})
    return obs.get("value") if obs else None


def get_obs_number(observations: dict, key: str) -> float | None:
    """Numeric view of an observation value for scoring math.

    Vitals can arrive as a string (e.g. GCS recorded as a FHIR
    valueCodeableConcept "10 (E2 V3 M5) - sedated"). Pull the leading number so
    comparisons like `gcs < 15` never raise TypeError. Return None if not numeric.
    """
    v = get_obs_value(observations, key)
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        m = re.match(r"\s*(-?\d+(?:\.\d+)?)", v)
        return float(m.group(1)) if m else None
    return None


def get_obs_unit(observations: dict, key: str, default: str = "") -> str:
    obs = observations.get(key, {})
    return (obs.get("unit") or default) if obs else default


def get_obs_timestamp(observations: dict, key: str) -> str:
    obs = observations.get(key, {})
    return (obs.get("timestamp") or "") if obs else ""


def convert_creatinine_to_mgdl(value, unit: str) -> float | None:
    """Creatinine -> mg/dL for CKD-EPI. µmol/L ÷ 88.4; mg/dL unchanged."""
    if value is None:
        return None
    unit = (unit or "").lower()
    if "umol" in unit or "µmol" in unit:
        return round(value / 88.4, 3)
    if "mg" in unit:
        return round(value, 3)
    return None


def convert_bilirubin_to_mgdl(value, unit: str) -> float | None:
    """Bilirubin -> mg/dL for SOFA. µmol/L ÷ 17.1; otherwise unchanged."""
    if value is None:
        return None
    unit = (unit or "").lower()
    if "umol" in unit or "µmol" in unit:
        return round(value / 17.1, 3)
    return round(value, 3)


def _agent_class(name: str) -> str | None:
    """Classify a medication name into a SOFA cardiovascular agent class.

    Order matters: norepinephrine/noradrenaline contain the "adrenaline"/"epi"
    substrings, so they must be matched before epinephrine.
    """
    n = (name or "").lower()
    if any(k in n for k in ("norepinephrine", "noradrenaline", "norepi")):
        return "norepi"
    if any(k in n for k in ("epinephrine", "adrenaline", "epi")):
        return "epi"
    if "dopamine" in n:
        return "dopamine"
    if "dobutamine" in n:
        return "dobutamine"
    if "vasopressin" in n or "phenylephrine" in n:
        return "other"
    return None


def _parse_dose(dose_str: str) -> tuple:
    """Return (value, is_rate, has_dose) from a dose string like '0.1 mcg/kg/min'.

    is_rate is True only for continuous-infusion units (contain '/min' or 'kg'),
    which is what the SOFA cardiovascular criteria are defined on; a bolus such as
    '0.5 mg' is not a SOFA infusion.
    """
    s = (dose_str or "").strip().lower()
    if not s:
        return (None, False, False)
    is_rate = ("/min" in s) or ("kg" in s)
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    return (float(m.group(1)) if m else None, is_rate, True)


def _cv_score(cls: str, value, is_rate: bool, has_dose: bool) -> int:
    """SOFA cardiovascular score for one agent (Sepsis-3 / News2 PDF table)."""
    if cls == "dobutamine":
        return 2  # any dose
    if cls == "dopamine":
        if not has_dose:
            return 3                      # undocumented infusion -> conservative
        if not is_rate:
            return 0                      # bolus, not a SOFA infusion
        if value is None:
            return 3
        if value <= 5:
            return 2
        if value <= 15:
            return 3
        return 4
    if cls in ("norepi", "epi"):
        if not has_dose:
            return 3
        if not is_rate:
            return 0
        return 3 if (value is not None and value <= 0.1) else 4
    if cls == "other":  # vasopressin / phenylephrine
        if not has_dose:
            return 3
        return 0 if not is_rate else 3
    return 0


def has_vasopressor(medications: list, administrations: list) -> dict:
    """Detect active vasopressors and derive a SOFA cardiovascular score.

    Scoring follows the SOFA cardiovascular table: Dobutamine (any) -> 2;
    Dopamine <=5 -> 2, >5 -> 3, >15 -> 4; Norepi/Epi <=0.1 -> 3, >0.1 -> 4.
    Doses are read from the FHIR dose strings; only continuous infusions
    (rate units) count toward the score, so a PRN bolus is detected but does
    not inflate SOFA. Undocumented infusion dose -> conservative 3.
    """
    agents = []
    max_score = 0
    max_dose = None
    for m in list(medications) + list(administrations):
        name = m.get("name", "") or ""
        cls = _agent_class(name)
        if not cls:
            continue
        if name not in agents:
            agents.append(name)
        value, is_rate, has_dose = _parse_dose(m.get("dose", ""))
        score = _cv_score(cls, value, is_rate, has_dose)
        if score > max_score:
            max_score = score
        if is_rate and value is not None and (max_dose is None or value > max_dose):
            max_dose = value

    return {
        "detected": bool(agents),
        "agents": agents,
        "max_dose": max_dose,
        "sofa_score": max_score,
    }


def has_hypercapnic_rf(conditions: list) -> bool:
    """True if the patient has hypercapnic (type 2) respiratory failure -> NEWS2 Scale 2."""
    hypercapnic_codes = ["J96.1", "J44"]
    hypercapnic_keywords = ["hypercapnic", "type 2 respiratory",
                            "copd", "co2 retention", "chronic respiratory"]
    for c in conditions:
        code = c.get("icd10_code", "") or ""
        name = ((c.get("name_vi", "") or "") + " "
                + (c.get("name_en", "") or "") + " "
                + (c.get("display", "") or "")).lower()
        if any(code.startswith(hc) for hc in hypercapnic_codes):
            return True
        if any(kw in name for kw in hypercapnic_keywords):
            return True
    return False


def is_on_oxygen(medications: list, administrations: list,
                 procedures: list) -> bool:
    """True if the patient is on supplemental O2 / ventilation."""
    oxygen_keywords = ["oxygen", "nasal cannula", "face mask",
                       "ventilat", "thở máy", "thở oxy", "thở ô-xy"]
    all_items = (
        [m.get("name", "").lower() for m in medications]
        + [m.get("name", "").lower() for m in administrations]
        + [p.get("name", "").lower() for p in procedures]
    )
    for item in all_items:
        if not item:
            continue
        if any(kw in item for kw in oxygen_keywords):
            return True
        # "o2" / "oxy" as a standalone-ish token (avoid matching inside words)
        if re.search(r"\bo2\b|\boxy\b", item):
            return True
    return False


# ===========================================================================
# Task 3 — MAP
# ===========================================================================
def calculate_map(observations: dict) -> dict:
    """Mean arterial pressure = (SBP + 2·DBP) / 3."""
    sbp = get_obs_number(observations, "systolic_bp")
    dbp = get_obs_number(observations, "diastolic_bp")

    if sbp is None or dbp is None:
        return {
            "value": None, "sbp": sbp, "dbp": dbp,
            "interpretation": "Không đủ dữ liệu", "missing": True,
        }

    map_val = round((sbp + 2 * dbp) / 3, 1)
    if map_val >= 70:
        interp = "Bình thường"
    elif map_val >= 65:
        interp = "Thấp — target MAP ≥ 65 (SSC 2021)"
    else:
        interp = "Nghiêm trọng — cần vasopressor"

    return {"value": map_val, "sbp": sbp, "dbp": dbp,
            "interpretation": interp, "missing": False}


# ===========================================================================
# Task 4 — qSOFA
# ===========================================================================
def calculate_qsofa(observations: dict) -> dict:
    """quick SOFA — GCS<15, RR≥22, SBP≤100 (1 pt each); positive if ≥2."""
    components = {}
    missing = []
    total = 0

    gcs = get_obs_number(observations, "gcs")
    if gcs is not None:
        score = 1 if gcs < 15 else 0
        total += score
        components["gcs"] = {"value": gcs, "score": score, "threshold": "< 15"}
    else:
        missing.append("gcs")
        components["gcs"] = {"value": None, "score": 0, "threshold": "< 15"}

    rr = get_obs_number(observations, "resp_rate")
    if rr is not None:
        score = 1 if rr >= 22 else 0
        total += score
        components["resp_rate"] = {"value": rr, "score": score, "threshold": "≥ 22"}
    else:
        missing.append("resp_rate")
        components["resp_rate"] = {"value": None, "score": 0, "threshold": "≥ 22"}

    sbp = get_obs_number(observations, "systolic_bp")
    if sbp is not None:
        score = 1 if sbp <= 100 else 0
        total += score
        components["systolic_bp"] = {"value": sbp, "score": score, "threshold": "≤ 100"}
    else:
        missing.append("systolic_bp")
        components["systolic_bp"] = {"value": None, "score": 0, "threshold": "≤ 100"}

    positive = total >= 2
    reliability = ("UNRELIABLE" if len(missing) >= 2
                   else "PARTIAL" if missing else "FULL")

    if positive:
        interp = "DƯƠNG TÍNH — Nguy cơ cao sepsis/tử vong. Đánh giá chuyên sâu ngay."
    elif total == 1:
        interp = "Nguy cơ thấp — theo dõi tiếp"
    else:
        interp = "Âm tính — không có dấu hiệu sepsis theo qSOFA"

    return {"total": total, "positive": positive, "components": components,
            "interpretation": interp, "missing_components": missing,
            "reliability": reliability}


# ===========================================================================
# Task 5 — SOFA (5/6 organs; pulmonary skipped without FiO2)
# ===========================================================================
def _score_bands(value, bands) -> int:
    """bands: list of (predicate(value) -> bool, score), first match wins."""
    for pred, score in bands:
        if pred(value):
            return score
    return 0


def calculate_sofa(observations: dict, conditions: list,
                   medications: list, administrations: list) -> dict:
    components = {}
    missing = []
    total = 0

    # ── Coagulation: platelet (10^3/µL) ──
    plt = get_obs_number(observations, "platelet")
    if plt is not None:
        sc = _score_bands(plt, [
            (lambda v: v <= 20, 4), (lambda v: v <= 50, 3),
            (lambda v: v <= 100, 2), (lambda v: v <= 150, 1)])
        components["coagulation"] = {"score": sc, "value": plt, "missing": False}
        total += sc
    else:
        missing.append("coagulation")
        components["coagulation"] = {"score": 0, "value": None, "missing": True}

    # ── Liver: bilirubin (mg/dL) ──
    bili_raw = get_obs_number(observations, "bilirubin")
    bili = convert_bilirubin_to_mgdl(bili_raw, get_obs_unit(observations, "bilirubin", "mg/dL"))
    if bili is not None:
        sc = _score_bands(bili, [
            (lambda v: v >= 12.0, 4), (lambda v: v >= 6.0, 3),
            (lambda v: v >= 2.0, 2), (lambda v: v >= 1.2, 1)])
        components["liver"] = {"score": sc, "value": bili, "missing": False}
        total += sc
    else:
        missing.append("liver")
        components["liver"] = {"score": 0, "value": None, "missing": True}

    # ── Cardiovascular: vasopressor infusion (≥2) supersedes MAP<70 (=1) ──
    vaso = has_vasopressor(medications, administrations)
    map_res = calculate_map(observations)
    if vaso["sofa_score"] > 0:
        cv_score = vaso["sofa_score"]
        cv_missing = False
    elif map_res["value"] is not None:
        cv_score = 1 if map_res["value"] < 70 else 0
        cv_missing = False
    else:
        cv_score = 0
        cv_missing = True
        missing.append("cardiovascular")
    components["cardiovascular"] = {"score": cv_score, "vasopressor": vaso,
                                    "map": map_res["value"], "missing": cv_missing}
    total += cv_score

    # ── Neurological: GCS ──
    gcs = get_obs_number(observations, "gcs")
    if gcs is not None:
        sc = _score_bands(gcs, [
            (lambda v: v < 6, 4), (lambda v: v <= 9, 3),
            (lambda v: v <= 12, 2), (lambda v: v <= 14, 1)])
        components["neurological"] = {"score": sc, "value": gcs, "missing": False}
        total += sc
    else:
        missing.append("neurological")
        components["neurological"] = {"score": 0, "value": None, "missing": True}

    # ── Renal: creatinine (mg/dL) ──
    cr_raw = get_obs_number(observations, "creatinine")
    cr = convert_creatinine_to_mgdl(cr_raw, get_obs_unit(observations, "creatinine", "umol/L"))
    if cr is not None:
        sc = _score_bands(cr, [
            (lambda v: v >= 5.0, 4), (lambda v: v >= 3.5, 3),
            (lambda v: v >= 2.0, 2), (lambda v: v >= 1.2, 1)])
        components["renal"] = {"score": sc, "value": cr, "missing": False}
        total += sc
    else:
        missing.append("renal")
        components["renal"] = {"score": 0, "value": None, "missing": True}

    # ── Pulmonary: PaO2/FiO2 — skipped (FiO2 not modeled) ──
    components["pulmonary"] = {"score": 0, "value": None, "missing": True}

    # Mortality estimate (theo bảng SOFA — News2/Sepsis-3 PDF)
    if total > 11:
        mortality = "Tử vong ước tính ≈ 95%"
    elif total >= 2:
        mortality = "Tử vong ước tính ≈ 10% (SOFA ≥ 2 → Sepsis)"
    else:
        mortality = "Nguy cơ thấp"

    scored = 5 - sum(1 for o in ("coagulation", "liver", "cardiovascular",
                                 "neurological", "renal") if o in missing)
    reliability = (f"PARTIAL({scored})" if scored >= 3
                   else "UNRELIABLE")

    return {"total": total, "components": components,
            "mortality_estimate": mortality, "missing_components": missing,
            "reliability": reliability,
            "note": "Phổi (PaO₂/FiO₂) không tính do thiếu FiO₂; SOFA dựa trên 5 cơ quan."}


# ===========================================================================
# Task 6 — NEWS2
# ===========================================================================
def _news2_spo2_scale1(spo2) -> int:
    return _score_bands(spo2, [
        (lambda v: v <= 91, 3), (lambda v: v <= 93, 2), (lambda v: v <= 95, 1)])


def _news2_spo2_scale2(spo2, on_oxygen: bool) -> int:
    """SpO2 scoring for hypercapnic respiratory failure (Scale 2)."""
    if on_oxygen:
        return _score_bands(spo2, [
            (lambda v: v >= 97, 3), (lambda v: v >= 95, 2),
            (lambda v: v >= 93, 1)])  # 93-94 ->1, 95-96 ->2, >=97 ->3; <=92 on O2 ->0
    # breathing room air
    return _score_bands(spo2, [
        (lambda v: v <= 83, 3), (lambda v: v <= 85, 2),
        (lambda v: v <= 87, 1)])  # 88-92 (or >=93 air) -> 0


def calculate_news2(observations: dict, conditions: list,
                    medications: list, administrations: list,
                    procedures: list) -> dict:
    scale = 2 if has_hypercapnic_rf(conditions) else 1
    on_oxygen = is_on_oxygen(medications, administrations, procedures)

    components = {}
    missing = []
    total = 0
    any_three = False

    def add(name, value, score, miss=False):
        nonlocal total, any_three
        components[name] = {"value": value, "score": score, "missing": miss}
        if not miss:
            total += score
            if score == 3:
                any_three = True

    # Respiratory rate
    rr = get_obs_number(observations, "resp_rate")
    if rr is not None:
        add("resp_rate", rr, _score_bands(rr, [
            (lambda v: v <= 8, 3), (lambda v: v >= 25, 3),
            (lambda v: v >= 21, 2), (lambda v: v <= 11, 1)]))
    else:
        missing.append("resp_rate"); add("resp_rate", None, 0, True)

    # SpO2 (scale-dependent)
    spo2 = get_obs_number(observations, "spo2")
    if spo2 is not None:
        sc = _news2_spo2_scale2(spo2, on_oxygen) if scale == 2 else _news2_spo2_scale1(spo2)
        add("spo2", spo2, sc)
    else:
        missing.append("spo2"); add("spo2", None, 0, True)

    # Supplemental oxygen (+2 if on O2)
    add("on_oxygen", on_oxygen, 2 if on_oxygen else 0)

    # Systolic BP
    sbp = get_obs_number(observations, "systolic_bp")
    if sbp is not None:
        add("systolic_bp", sbp, _score_bands(sbp, [
            (lambda v: v <= 90, 3), (lambda v: v >= 220, 3),
            (lambda v: v <= 100, 2), (lambda v: v <= 110, 1)]))
    else:
        missing.append("systolic_bp"); add("systolic_bp", None, 0, True)

    # Heart rate
    hr = get_obs_number(observations, "heart_rate")
    if hr is not None:
        add("heart_rate", hr, _score_bands(hr, [
            (lambda v: v <= 40, 3), (lambda v: v >= 131, 3),
            (lambda v: v >= 111, 2), (lambda v: v <= 50, 1),
            (lambda v: v >= 91, 1)]))
    else:
        missing.append("heart_rate"); add("heart_rate", None, 0, True)

    # Consciousness (GCS 15 = Alert -> 0, else 3)
    gcs = get_obs_number(observations, "gcs")
    if gcs is not None:
        add("gcs", gcs, 0 if gcs >= 15 else 3)
    else:
        missing.append("gcs"); add("gcs", None, 0, True)

    # Temperature (°C)
    temp = get_obs_number(observations, "temperature")
    if temp is not None:
        add("temperature", temp, _score_bands(temp, [
            (lambda v: v <= 35.0, 3), (lambda v: v >= 39.1, 2),
            (lambda v: v >= 38.1, 1), (lambda v: v <= 36.0, 1)]))
    else:
        missing.append("temperature"); add("temperature", None, 0, True)

    # Risk classification
    if total >= 7:
        risk, risk_vi = "HIGH", "NGUY CƠ CAO"
    elif total >= 5:
        risk, risk_vi = "MEDIUM", "Nguy cơ trung bình"
    elif any_three:
        risk, risk_vi = "MEDIUM-LOW", "Nguy cơ thấp-trung bình"
    elif total >= 1:
        risk, risk_vi = "LOW", "Nguy cơ thấp"
    else:
        risk, risk_vi = "LOW", "Nguy cơ thấp"

    reliability = ("UNRELIABLE" if len(missing) >= 3
                   else "PARTIAL" if missing else "FULL")

    return {"total": total, "risk_level": risk, "risk_vi": risk_vi,
            "scale": scale, "on_oxygen": on_oxygen,
            "components": components, "missing_components": missing,
            "reliability": reliability,
            "note": f"NEWS2 Scale {scale}" + (" (suy hô hấp tăng CO₂)" if scale == 2 else "")}


# ===========================================================================
# Task 7 — eGFR (CKD-EPI 2021)
# ===========================================================================
def calculate_egfr(observations: dict, patient: dict) -> dict:
    obs = observations.get("creatinine", {})
    cr_value = obs.get("value") if obs else None
    cr_unit = (obs.get("unit") or "umol/L") if obs else "umol/L"
    if isinstance(cr_value, str):
        cr_value = None  # creatinine should be numeric; bail to missing otherwise

    age = patient.get("age")
    if cr_value is None or age is None:
        reason = "Thiếu Creatinine" if cr_value is None else "Thiếu tuổi bệnh nhân"
        return {
            "egfr": None, "creatinine_mgdl": None, "creatinine_umol": None,
            "stage": "Unknown", "stage_vi": "Không đủ dữ liệu",
            "dose_adjustment": False, "missing": True,
            "note": f"{reason} — không tính được eGFR",
        }

    cr_mgdl = convert_creatinine_to_mgdl(cr_value, cr_unit)
    cr_umol = cr_value if ("umol" in cr_unit.lower() or "µmol" in cr_unit.lower()) \
        else cr_value * 88.4

    gender = (patient.get("gender") or "male").lower()
    female = gender in ("female", "f", "nữ")
    kappa = 0.7 if female else 0.9
    alpha = -0.241 if female else -0.302
    sex_factor = 1.012 if female else 1.0

    ratio = cr_mgdl / kappa
    egfr = round(
        142 * (min(ratio, 1) ** alpha) * (max(ratio, 1) ** -1.200)
        * (0.9938 ** age) * sex_factor, 1)

    if egfr >= 90:
        stage, stage_vi = "Stage 1", "Bình thường hoặc tăng"
    elif egfr >= 60:
        stage, stage_vi = "Stage 2", "Giảm nhẹ"
    elif egfr >= 45:
        stage, stage_vi = "Stage 3a", "Giảm nhẹ-vừa"
    elif egfr >= 30:
        stage, stage_vi = "Stage 3b", "Giảm vừa-nặng"
    elif egfr >= 15:
        stage, stage_vi = "Stage 4", "Giảm nặng"
    else:
        stage, stage_vi = "Stage 5", "Suy thận — xem xét lọc máu"

    note = ""
    if egfr < 30:
        note = ("Suy thận nặng — điều chỉnh liều nghiêm trọng với Vancomycin, "
                "Gentamicin, kháng sinh thải qua thận")
    elif egfr < 60:
        note = "Suy thận — cần điều chỉnh liều nhiều thuốc"

    return {
        "egfr": egfr, "creatinine_mgdl": cr_mgdl,
        "creatinine_umol": round(cr_umol, 1), "stage": stage, "stage_vi": stage_vi,
        "dose_adjustment": egfr < 60, "note": note, "missing": False,
    }


# ===========================================================================
# Task 8 — calculate_all + printing + CLI
# ===========================================================================
def calculate_all(patient_context: dict) -> dict:
    obs = patient_context.get("observations", {})
    conds = patient_context.get("conditions", [])
    meds = patient_context.get("medications", [])
    admins = patient_context.get("medication_administrations", [])
    procs = patient_context.get("procedures", [])
    pat = patient_context.get("patient", {})

    map_result = calculate_map(obs)
    qsofa_result = calculate_qsofa(obs)
    sofa_result = calculate_sofa(obs, conds, meds, admins)
    news2_result = calculate_news2(obs, conds, meds, admins, procs)
    egfr_result = calculate_egfr(obs, pat)

    alerts = []
    if news2_result["risk_level"] == "HIGH":
        alerts.append(f"NEWS2 = {news2_result['total']} — MỨC ĐỘ CAO")
    if qsofa_result["positive"]:
        alerts.append("qSOFA ≥ 2 — Nguy cơ sepsis cao")
    if map_result.get("value") is not None and map_result["value"] < 65:
        alerts.append(f"MAP = {map_result['value']} mmHg — Dưới ngưỡng (< 65)")
    if egfr_result.get("dose_adjustment"):
        alerts.append(f"eGFR = {egfr_result['egfr']} — Cần điều chỉnh liều thuốc")

    return {
        "map": map_result, "qsofa": qsofa_result, "sofa": sofa_result,
        "news2": news2_result, "egfr": egfr_result,
        "summary": {
            "alerts": alerts, "alert_count": len(alerts),
            "highest_risk": news2_result["risk_level"],
            "sepsis_screen": qsofa_result["positive"],
        },
    }


def print_calculations(calc: dict):
    m = calc["map"]
    print("\n[MAP]")
    if m["missing"]:
        print("  Thiếu dữ liệu (SBP hoặc DBP)")
    else:
        print(f"  {m['value']} mmHg (SBP {m['sbp']} / DBP {m['dbp']})")
        print(f"  → {m['interpretation']}")

    q = calc["qsofa"]
    print(f"\n[qSOFA] = {q['total']}/3 "
          f"({'DƯƠNG TÍNH' if q['positive'] else 'Âm tính'}) [{q['reliability']}]")
    for name, comp in q["components"].items():
        v, s, t = comp["value"], comp["score"], comp["threshold"]
        print(f"  {name:<15} = {v if v is not None else '?':>6}  ({t}) → {s} điểm")

    s = calc["sofa"]
    print(f"\n[SOFA] = {s['total']}/24 | {s['mortality_estimate']} [{s['reliability']}]")
    for organ, comp in s["components"].items():
        v = comp.get("value")
        miss = "(thiếu)" if comp.get("missing") else ""
        print(f"  {organ:<20} score={comp['score']}  "
              f"value={v if v is not None else '?'} {miss}")

    n = calc["news2"]
    print(f"\n[NEWS2] = {n['total']} điểm → {n['risk_vi']} "
          f"(Scale {n['scale']}) [{n['reliability']}]")
    for name, comp in n["components"].items():
        v = comp["value"]
        miss = "(thiếu)" if comp.get("missing") else ""
        print(f"  {name:<15} = {str(v) if v is not None else '?':>6}  → {comp['score']} điểm {miss}")

    e = calc["egfr"]
    print("\n[eGFR]")
    if e["missing"]:
        print(f"  {e['note']}")
    else:
        print(f"  Creatinine: {e['creatinine_umol']} μmol/L ({e['creatinine_mgdl']} mg/dL)")
        print(f"  eGFR: {e['egfr']} mL/min/1.73m² → {e['stage']} ({e['stage_vi']})")
        if e["note"]:
            print(f"  {e['note']}")

    if calc["summary"]["alerts"]:
        print(f"\n{'=' * 55}")
        print("ALERTS:")
        for alert in calc["summary"]["alerts"]:
            print(f"  {alert}")
        print(f"{'=' * 55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clinical risk-score calculator")
    parser.add_argument("--file", help="FHIR Bundle JSON file")
    parser.add_argument("--patient", help="FHIR Patient ID from the live sandbox")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.file:
        client = FHIRClient.from_file(args.file)
    elif args.patient:
        client = FHIRClient(args.patient)
    else:
        parser.error("need --file or --patient")

    context = client.build_patient_context()
    calc = calculate_all(context)

    if args.json:
        print(json.dumps(calc, ensure_ascii=False, indent=2))
    else:
        print_calculations(calc)
