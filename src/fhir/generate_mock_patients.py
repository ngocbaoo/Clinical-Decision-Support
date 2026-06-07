"""
Generate the second batch of mock FHIR R4 patient bundles (patient_H .. patient_Q).

Patients A–G were hand-curated; this script programmatically emits 10 more to
cover a systematic matrix of normal cases and FHIR-shape / clinical edge cases
(see PATIENTS below). Output is the same static Bundle-per-file format that
fhir_client.py --file consumes; the hand-written A–G files are left untouched.

Run:  python src/fhir/generate_mock_patients.py
"""

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MOCK_DIR = Path(__file__).resolve().parents[2] / "data" / "mock"

LOINC = {
    "spo2": ("59408-5", "Oxygen saturation", "%"),
    "heart_rate": ("8867-4", "Heart rate", "/min"),
    "resp_rate": ("9279-1", "Respiratory rate", "/min"),
    "systolic_bp": ("8480-6", "Systolic blood pressure", "mmHg"),
    "diastolic_bp": ("8462-4", "Diastolic blood pressure", "mmHg"),
    "temperature": ("8310-5", "Body temperature", "Cel"),
    "gcs": ("9267-6", "Glasgow coma score total", "{score}"),
    "creatinine": ("2160-0", "Creatinine", "umol/L"),
    "lactate": ("32693-4", "Lactate", "mmol/L"),
    "bilirubin": ("1975-2", "Bilirubin total", "mg/dL"),
    "wbc": ("26464-8", "Leukocytes", "10*3/uL"),
    "platelet": ("777-3", "Platelets", "10*3/uL"),
    "hemoglobin": ("718-7", "Hemoglobin", "g/dL"),
    "pao2": ("2703-7", "Oxygen partial pressure", "mmHg"),
    "potassium": ("2823-3", "Potassium", "mmol/L"),
    "sodium": ("2951-2", "Sodium", "mmol/L"),
}

# Reference value sets ------------------------------------------------------
NORMAL_ADULT = {
    "spo2": 98, "heart_rate": 76, "resp_rate": 14, "systolic_bp": 120,
    "diastolic_bp": 76, "temperature": 36.7, "gcs": 15, "creatinine": 80,
    "lactate": 1.0, "wbc": 7.0, "platelet": 240, "hemoglobin": 14.0,
    "potassium": 4.1, "sodium": 140,
}
NORMAL_PEDIATRIC = {
    "spo2": 99, "heart_rate": 96, "resp_rate": 22, "systolic_bp": 100,
    "diastolic_bp": 64, "temperature": 36.8, "gcs": 15, "creatinine": 38,
    "wbc": 8.0, "platelet": 280, "hemoglobin": 12.5, "potassium": 4.3,
    "sodium": 139,
}

DT = "2026-06-04T08:00:00"
LAB_DT = "2026-06-04T07:00:00"


# Resource builders ---------------------------------------------------------
def patient(pid, gender, birth=None, text=None, family=None, given=None):
    name = {}
    if text:
        name["text"] = text
    if family:
        name["family"] = family
    if given:
        name["given"] = given
    p = {"resourceType": "Patient", "id": pid, "name": [name], "gender": gender}
    if birth:
        p["birthDate"] = birth
    return p


def encounter(eid, pid, service_text=None, service_coding=None, cls="IMP",
              start="2026-06-04T06:00:00", reason="ICU admission", location="ICU"):
    svc = {}
    if service_text:
        svc["text"] = service_text
    if service_coding:
        svc["coding"] = [{"display": service_coding}]
    return {
        "resourceType": "Encounter", "id": eid, "status": "in-progress",
        "class": {"code": cls, "display": "inpatient encounter"},
        "subject": {"reference": f"Patient/{pid}"},
        "serviceType": svc,
        "period": {"start": start},
        "reasonCode": [{"text": reason}],
        "location": [{"location": {"display": location}}],
    }


def obs(oid, pid, key, value, *, unit=None, ts=DT, use_issued=False,
        codeable_text=None, component=None, no_value=False):
    code, display, default_unit = LOINC[key]
    r = {
        "resourceType": "Observation", "id": oid, "status": "final",
        "subject": {"reference": f"Patient/{pid}"},
        "code": {"coding": [{"system": "http://loinc.org", "code": code,
                             "display": display}]},
    }
    r["issued" if use_issued else "effectiveDateTime"] = ts
    if no_value:
        return r  # intentionally value-less -> client should skip it
    if codeable_text is not None:
        r["valueCodeableConcept"] = {"text": codeable_text}
    elif component is not None:
        r["component"] = component
    else:
        r["valueQuantity"] = {"value": value, "unit": unit or default_unit,
                              "system": "http://unitsofmeasure.org"}
    return r


def bp_component(oid, pid, systolic, diastolic, ts=DT):
    """A single BP-panel Observation carrying systolic/diastolic in component[]."""
    return obs(oid, pid, "systolic_bp", None, ts=ts, component=[
        {"code": {"coding": [{"system": "http://loinc.org", "code": "8480-6",
                              "display": "Systolic blood pressure"}]},
         "valueQuantity": {"value": systolic, "unit": "mmHg"}},
        {"code": {"coding": [{"system": "http://loinc.org", "code": "8462-4",
                              "display": "Diastolic blood pressure"}]},
         "valueQuantity": {"value": diastolic, "unit": "mmHg"}},
    ])


def vitals(pid, values, *, prefix):
    """Build one Observation per key in `values` (value or (value, unit) tuple)."""
    out = []
    for i, (key, v) in enumerate(values.items()):
        unit = None
        if isinstance(v, tuple):
            v, unit = v
        ts = LAB_DT if key in ("creatinine", "lactate", "bilirubin", "wbc",
                               "platelet", "hemoglobin", "pao2", "potassium",
                               "sodium") else DT
        out.append(obs(f"obs-{prefix}-{i}", pid, key, v, unit=unit, ts=ts))
    return out


def allergy(aid, pid, name, *, criticality="high", reaction="Rash",
            category="medication", coding_only=False):
    code = {"coding": [{"display": name}]}
    if not coding_only:
        code["text"] = name
    r = {
        "resourceType": "AllergyIntolerance", "id": aid,
        "patient": {"reference": f"Patient/{pid}"},
        "clinicalStatus": {"coding": [{
            "system": "http://terminology.hl7.org/CodeSystem/allergyintolerance-clinical",
            "code": "active"}]},
        "criticality": criticality, "code": code, "category": [category],
    }
    if reaction:
        r["reaction"] = [{"manifestation": [{"coding": [{"display": reaction}]}]}]
    return r


def medication(mid, pid, name, *, dose=None, route="Intravenous",
               freq="mỗi 24 giờ", via_reference=False, no_dosage=False):
    r = {
        "resourceType": "MedicationRequest", "id": mid, "status": "active",
        "subject": {"reference": f"Patient/{pid}"},
    }
    if via_reference:
        r["medicationReference"] = {"display": name}
    else:
        r["medicationCodeableConcept"] = {"text": name}
    if not no_dosage:
        di = {"route": {"coding": [{"display": route}]},
              "timing": {"code": {"text": freq}}}
        if dose:
            di["doseAndRate"] = [{"doseQuantity": {"value": dose[0], "unit": dose[1]}}]
        r["dosageInstruction"] = [di]
    return r


def condition(cid, pid, icd10, display, text, *, severity=None,
              coding_first_noise=False):
    coding = []
    if coding_first_noise:
        coding.append({"system": "http://snomed.info/sct", "code": "00000",
                       "display": "local code"})
    coding.append({"system": "http://hl7.org/fhir/sid/icd-10", "code": icd10,
                   "display": display})
    r = {
        "resourceType": "Condition", "id": cid,
        "subject": {"reference": f"Patient/{pid}"},
        "clinicalStatus": {"coding": [{
            "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
            "code": "active"}]},
        "code": {"coding": coding, "text": text},
        "onsetDateTime": "2026-06-03T20:00:00",
    }
    if severity:
        r["severity"] = {"coding": [{"display": severity}]}
    return r


def med_admin(maid, pid, name, dose, *, status="completed",
              use_period=False, ts="2026-06-04T07:00:00"):
    r = {
        "resourceType": "MedicationAdministration", "id": maid, "status": status,
        "subject": {"reference": f"Patient/{pid}"},
        "medicationCodeableConcept": {"text": name},
        "dosage": {"dose": {"value": dose[0], "unit": dose[1]},
                   "route": {"coding": [{"display": "Intravenous"}]}},
    }
    r["effectivePeriod" if use_period else "effectiveDateTime"] = (
        {"start": ts} if use_period else ts)
    return r


def procedure(prid, pid, name, text, *, status="completed",
              body=None, use_period=False, ts="2026-06-03T18:00:00"):
    r = {
        "resourceType": "Procedure", "id": prid, "status": status,
        "subject": {"reference": f"Patient/{pid}"},
        "code": {"coding": [{"display": name}], "text": text},
    }
    r["performedPeriod" if use_period else "performedDateTime"] = (
        {"start": ts} if use_period else ts)
    if body:
        r["bodySite"] = [{"coding": [{"display": body}]}]
    return r


def report(drid, pid, title, *, status="final", use_issued=False,
           ts="2026-06-04T07:00:00"):
    r = {
        "resourceType": "DiagnosticReport", "id": drid, "status": status,
        "subject": {"reference": f"Patient/{pid}"},
        "code": {"text": title},
    }
    r["issued" if use_issued else "effectiveDateTime"] = ts
    return r


def bundle(entries):
    return {"resourceType": "Bundle", "type": "collection",
            "entry": [{"resource": r} for r in entries]}


# Per-patient assembly ------------------------------------------------------
def build_H():
    pid = "pt-008"
    e = [patient(pid, "female", "1972-04-18", family="Hoàng", given=["Thị Hoa"]),
         encounter("enc-008", pid, service_text="ICU",
                   start="2026-06-03T15:00:00", reason="Post-operative monitoring",
                   location="ICU Phòng 2 Giường 3")]
    e += vitals(pid, NORMAL_ADULT, prefix="h")
    e += [medication("med-h1", pid, "Paracetamol", dose=(1, "g"), freq="mỗi 6 giờ"),
          condition("cond-h1", pid, "I10", "Essential (primary) hypertension",
                    "Tăng huyết áp"),
          procedure("proc-h1", pid, "Cholecystectomy", "Cắt túi mật",
                    body="Gallbladder"),
          report("dr-h1", pid, "Complete Blood Count", use_issued=True)]
    return e


def build_I():
    pid = "pt-009"
    e = [patient(pid, "male", "1948-12-01", text="Vũ Đình Inh"),
         encounter("enc-009", pid, service_coding="Intensive Care Unit",
                   start="2026-06-02T09:00:00", reason="COPD exacerbation",
                   location="ICU Hô hấp Giường 2")]
    e += vitals(pid, {**NORMAL_ADULT, "spo2": 93, "resp_rate": 20}, prefix="i")
    e += [medication("med-i1", pid, "Salbutamol nebuliser", dose=(5, "mg"),
                     route="Inhalation", freq="mỗi 6 giờ"),
          condition("cond-i1", pid, "J44", "Chronic obstructive pulmonary disease",
                    "Bệnh phổi tắc nghẽn mạn tính", severity="Moderate"),
          condition("cond-i2", pid, "I10", "Essential (primary) hypertension",
                    "Tăng huyết áp"),
          report("dr-i1", pid, "Arterial Blood Gas")]
    return e


def build_J():
    pid = "pt-010"
    e = [patient(pid, "male", "2018-06-10", text="Đỗ Gia Kiệt"),
         encounter("enc-010", pid, service_text="PICU",
                   start="2026-06-04T05:00:00", reason="Bronchiolitis observation",
                   location="PICU Giường 1")]
    e += vitals(pid, NORMAL_PEDIATRIC, prefix="j")
    e += [medication("med-j1", pid, "Paracetamol", dose=(250, "mg"), route="Oral",
                     freq="mỗi 6 giờ"),
          condition("cond-j1", pid, "A04.9", "Bacterial intestinal infection",
                    "Nhiễm trùng đường ruột"),
          report("dr-j1", pid, "Respiratory viral panel")]
    return e


def build_K():
    pid = "pt-011"
    e = [patient(pid, "female", "1928-03-25", text="Bùi Thị Lan"),
         encounter("enc-011", pid, service_text="ICU",
                   start="2026-06-03T12:00:00", reason="Urosepsis",
                   location="ICU Phòng 1 Giường 1")]
    e += vitals(pid, {**NORMAL_ADULT, "heart_rate": 102, "temperature": 38.2,
                      "wbc": 15.0}, prefix="k")
    e += [medication("med-k1", pid, "Ceftriaxone", dose=(2, "g"), freq="mỗi 24 giờ"),
          condition("cond-k1", pid, "A41.9", "Sepsis, unspecified organism",
                    "Nhiễm khuẩn huyết"),
          med_admin("ma-k1", pid, "Ceftriaxone", (2, "g")),
          report("dr-k1", pid, "Urine culture", status="preliminary")]
    return e


def build_L():
    # Extreme minimal: only a Patient, no birthDate (age None), name parts only.
    pid = "pt-012"
    return [patient(pid, "unknown", family="Trương", given=["Văn M"])]


def build_M():
    pid = "pt-013"
    e = [patient(pid, "male", "1980-07-07", text="Ngô Bá Năng"),
         encounter("enc-013", pid, service_text="ICU",
                   start="2026-06-04T03:00:00", reason="Observation",
                   location="ICU Phòng 3 Giường 1")]
    # obs timestamps via `issued`; no Condition resources at all (empty section)
    e += [obs("obs-m-0", pid, "spo2", 97, use_issued=True),
          obs("obs-m-1", pid, "heart_rate", 82, use_issued=True),
          obs("obs-m-2", pid, "resp_rate", 16, use_issued=True),
          obs("obs-m-3", pid, "creatinine", 84, ts=LAB_DT, use_issued=True),
          medication("med-m1", pid, "Normal saline 0.9%", no_dosage=True),
          report("dr-m1", pid, "Basic Metabolic Panel")]
    return e


def build_N():
    pid = "pt-014"
    e = [patient(pid, "female", "1990-11-11", text="Lý Thị Oanh"),
         encounter("enc-014", pid, service_text="ICU",
                   start="2026-06-04T01:00:00", reason="Anaphylaxis observation",
                   location="ICU Phòng 2 Giường 2"),
         allergy("alg-n1", pid, "Penicillin", criticality="high",
                 reaction="Anaphylaxis", category="medication"),
         allergy("alg-n2", pid, "Peanut", criticality="high",
                 reaction="Angioedema", category="food"),
         allergy("alg-n3", pid, "Pollen", criticality="low", reaction=None,
                 category="environment"),
         allergy("alg-n4", pid, "Shellfish", criticality="unable-to-assess",
                 reaction="Urticaria", category="food")]
    e += vitals(pid, NORMAL_ADULT, prefix="n")
    e += [medication("med-n1", pid, "Adrenaline", dose=(0.5, "mg"),
                     route="Intramuscular", freq="khi cần"),
          condition("cond-n1", pid, "D69.6", "Thrombocytopenia, unspecified",
                    "Giảm tiểu cầu"),
          report("dr-n1", pid, "Tryptase level")]
    return e


def build_O():
    pid = "pt-015"
    e = [patient(pid, "male", "1962-05-19", text="Phan Văn Phú"),
         encounter("enc-015", pid, service_text="ICU",
                   start="2026-06-03T23:00:00", reason="Diabetic ketoacidosis",
                   location="ICU Phòng 4 Giường 2")]
    # Unit-conversion edge cases: creatinine mg/dL, temperature degF; low GCS, hypoNa
    vals = {**NORMAL_ADULT, "gcs": 8, "sodium": 125,
            "creatinine": (2.4, "mg/dL"), "temperature": (101.3, "[degF]")}
    e += vitals(pid, vals, prefix="o")
    e += [medication("med-o1", pid, "Insulin regular", dose=(0.1, "IU/kg/h"),
                     route="Intravenous infusion", freq="truyền liên tục"),
          condition("cond-o1", pid, "N17.0",
                    "Acute kidney failure with tubular necrosis",
                    "Suy thận cấp hoại tử ống thận"),
          med_admin("ma-o1", pid, "Insulin regular", (6, "IU"), use_period=True),
          report("dr-o1", pid, "Arterial Blood Gas")]
    return e


def build_P():
    pid = "pt-016"
    e = [patient(pid, "female", "1975-08-08", text="Tạ Thị Quỳnh"),
         encounter("enc-016", pid, cls="EMER", service_text="Emergency ICU",
                   start="2026-06-04T06:30:00", reason="Seizure",
                   location="ED Resus Giường 1")]
    # Duplicate SpO2 (older then newer -> client keeps the most recent 91),
    # plus one value-less observation that must be skipped.
    e += [obs("obs-p-spo2-old", pid, "spo2", 96, ts="2026-06-04T06:00:00"),
          obs("obs-p-spo2-new", pid, "spo2", 91, ts="2026-06-04T08:00:00"),
          obs("obs-p-hr", pid, "heart_rate", 110),
          obs("obs-p-rr", pid, "resp_rate", 20),
          obs("obs-p-creat", pid, "creatinine", 90, ts=LAB_DT),
          obs("obs-p-novalue", pid, "lactate", None, no_value=True),
          medication("med-p1", pid, "Levetiracetam", dose=(1000, "mg")),
          condition("cond-p1", pid, "G93.6", "Cerebral oedema", "Phù não"),
          procedure("proc-p1", pid, "Lumbar puncture", "Chọc dò tủy sống"),
          report("dr-p1", pid, "CT head")]
    return e


def build_Q():
    pid = "pt-017"
    e = [patient(pid, "male", "1958-01-30", text="Hồ Văn Sơn"),
         encounter("enc-017", pid, service_text="ICU",
                   start="2026-06-03T20:00:00",
                   reason="Multi-organ failure", location="ICU Phòng 5 Giường 1")]
    # Extreme deranged values across the board (boundary/critical case).
    extreme = {
        "spo2": 78, "heart_rate": 145, "resp_rate": 38, "systolic_bp": 70,
        "diastolic_bp": 40, "temperature": 40.1, "gcs": 5, "creatinine": 420,
        "lactate": 12.0, "bilirubin": 14.0, "wbc": 28.0, "platelet": 15,
        "hemoglobin": 6.5, "pao2": 48, "potassium": 6.8, "sodium": 122,
    }
    e += vitals(pid, extreme, prefix="q")
    e += [bp_component("obs-q-bp", pid, 70, 40),  # also BP via component
          medication("med-q1", pid, "Noradrenaline", via_reference=True,
                     dose=(0.5, "mcg/kg/min")),
          # ICD-10 sits in coding[1] (noise code first) -> tests coding search
          condition("cond-q1", pid, "A41.9", "Sepsis, unspecified organism",
                    "Nhiễm khuẩn huyết", coding_first_noise=True),
          condition("cond-q2", pid, "N17.9", "Acute kidney failure, unspecified",
                    "Suy thận cấp"),
          med_admin("ma-q1", pid, "Noradrenaline", (0.5, "mcg/kg/min"),
                    status="in-progress", use_period=True),
          procedure("proc-q1", pid, "Continuous renal replacement therapy",
                    "Lọc máu liên tục"),  # no bodySite
          report("dr-q1", pid, "Sepsis panel")]
    return e


PATIENTS = [
    ("patient_H.json", "pt-008", "Hoàng Thị Hoa", build_H,
     "Hậu phẫu cắt túi mật, ICU. Toàn bộ vital/lab bình thường (baseline #2).",
     [], ["all-normal baseline", "name family/given only (no text)",
          "DiagnosticReport via issued"]),
    ("patient_I.json", "pt-009", "Vũ Đình Inh", build_I,
     "Đợt cấp COPD ở người cao tuổi, vital gần bình thường.",
     [], ["condition severity field", "encounter serviceType via coding (no text)"]),
    ("patient_J.json", "pt-010", "Đỗ Gia Kiệt", build_J,
     "Bệnh nhi (7 tuổi), viêm tiểu phế quản, vital nhi bình thường.",
     [], ["pediatric age calc", "pediatric normal vitals", "no allergies"]),
    ("patient_K.json", "pt-011", "Bùi Thị Lan", build_K,
     "Cụ bà ~98 tuổi, nhiễm khuẩn tiết niệu.",
     [], ["very elderly age calc", "mild sepsis derangement"]),
    ("patient_L.json", "pt-012", "Trương Văn M", build_L,
     "Hồ sơ tối thiểu tuyệt đối: chỉ có Patient, không birthDate (age None).",
     [], ["only Patient resource", "no birthDate -> age None",
          "name parts only", "every section empty"]),
    ("patient_M.json", "pt-013", "Ngô Bá Năng", build_M,
     "Có vital + thuốc nhưng KHÔNG có Condition (empty conditions).",
     [], ["empty conditions", "medication without dosageInstruction",
          "observation timestamp via issued"]),
    ("patient_N.json", "pt-014", "Lý Thị Oanh", build_N,
     "Đa dị ứng (thuốc/thức ăn/môi trường); một dị ứng không có reaction.",
     [], ["multiple allergy categories", "allergy without reaction",
          "criticality unable-to-assess"]),
    ("patient_O.json", "pt-015", "Phan Văn Phú", build_O,
     "Toan ceton đái tháo đường; creatinine mg/dL + nhiệt độ °F (unit conversion).",
     [], ["creatinine mg/dL -> umol/L", "temperature degF -> Celsius",
          "low GCS", "hyponatremia"]),
    ("patient_P.json", "pt-016", "Tạ Thị Quỳnh", build_P,
     "Co giật, ED resus. SpO2 lặp hai mốc thời gian + một observation thiếu value.",
     [], ["duplicate observation (keep latest)", "value-less observation skipped",
          "emergency encounter class"]),
    ("patient_Q.json", "pt-017", "Hồ Văn Sơn", build_Q,
     "Suy đa tạng, toàn bộ chỉ số nguy kịch (boundary/extreme case).",
     ["S-03", "S-05"], ["extreme deranged vitals/labs", "BP via component",
                        "ICD-10 in coding[1]", "procedure without bodySite",
                        "medicationReference"]),
]


def main():
    MOCK_DIR.mkdir(parents=True, exist_ok=True)
    new_entries = []
    for fname, pid, name, builder, desc, scenarios, edges in PATIENTS:
        b = bundle(builder())
        (MOCK_DIR / fname).write_text(
            json.dumps(b, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"  wrote {fname}: {len(b['entry'])} entries")
        new_entries.append({
            "file": fname, "id": pid, "name": name, "description": desc,
            "covers_scenarios": scenarios, "edge_cases": edges,
        })

    # Merge into index.json: keep hand-written A–G entries, replace/append H–Q.
    index_path = MOCK_DIR / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    new_files = {e["file"] for e in new_entries}
    kept = [p for p in index["patients"] if p["file"] not in new_files]
    index["patients"] = kept + new_entries
    index["total_patients"] = len(index["patients"])
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"index.json -> {index['total_patients']} patients total")


if __name__ == "__main__":
    main()
