"""
Pulls a patient's *full clinical profile* from the public SMART Health IT FHIR R4
sandbox (https://r4.smarthealthit.org, no auth) and consolidates 9 resources into
one object: demographics, active encounter, allergies, vitals & labs, medications
(requested + administered), conditions, procedures, and diagnostic reports.

This is the structured context that later scoring (NEWS2 / eGFR) and the RAG layer
will consume. No clinical calculations here yet.

Run:
    python src/fhir/fhir_client.py --find                   # list sandbox patient IDs
    python src/fhir/fhir_client.py --patient <id>           # readable full profile
    python src/fhir/fhir_client.py --patient <id> --json    # same profile as JSON
    python src/fhir/fhir_client.py --file data/mock/patient_A.json  # mock data
"""

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
from dateutil.relativedelta import relativedelta

# Windows consoles default to cp1252 and choke on Vietnamese — force UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ on sys.path
from paths import DB_PATH  # noqa: E402

# ---------------------------------------------------------------------------
# LOINC codes of interest (vital signs + ICU labs)
# ---------------------------------------------------------------------------
LOINC_VITALS = {
    "59408-5": "spo2",
    "8867-4":  "heart_rate",
    "9279-1":  "resp_rate",
    "8480-6":  "systolic_bp",
    "8462-4":  "diastolic_bp",
    "8310-5":  "temperature",
    "9267-6":  "gcs",
}
LOINC_LABS = {
    "2160-0":  "creatinine",
    "32693-4": "lactate",
    "1975-2":  "bilirubin",
    "26464-8": "wbc",
    "777-3":   "platelet",
    "718-7":   "hemoglobin",
    "2703-7":  "pao2",
    "2823-3":  "potassium",
    "2951-2":  "sodium",
}
ALL_LOINC = {**LOINC_VITALS, **LOINC_LABS}


def _log(msg: str) -> None:
    """Progress/status to stderr so --json keeps stdout pure."""
    print(msg, file=sys.stderr)


# ---------------------------------------------------------------------------
# Patient discovery (Task 2)
# ---------------------------------------------------------------------------
def find_patients(count: int = 10) -> None:
    """Print real patient IDs from the sandbox so we have valid targets."""
    resp = requests.get(
        "https://r4.smarthealthit.org/Patient",
        params={"_count": count},
        headers={"Accept": "application/fhir+json"},
        timeout=10,
    )
    bundle = resp.json()
    for entry in bundle.get("entry", []):
        p = entry["resource"]
        pid = p["id"]
        name = p.get("name", [{}])[0]
        display = name.get("text") or \
            f"{name.get('family', '')} {' '.join(name.get('given', []))}"
        print(f"ID: {pid} | Name: {display.strip()}")


class FHIRClient:
    BASE_URL = "https://r4.smarthealthit.org"

    def __init__(self, patient_id: str):
        self.patient_id = patient_id
        self.encounter_id = None          # set after Encounter query
        # Mock mode: when _mock_mode is set, resources are served from a local
        # FHIR Bundle (see from_file) instead of HTTP.
        self._mock_mode = False
        self._bundle = None
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/fhir+json"})
        self.db = sqlite3.connect(DB_PATH)
        self.missing_resources = []

    @classmethod
    def from_file(cls, filepath: str) -> "FHIRClient":
        """Load patient data from a FHIR Bundle JSON file (replaces HTTP)."""
        with open(filepath, encoding="utf-8") as f:
            bundle = json.load(f)

        patient_id = None
        for entry in bundle.get("entry", []):
            r = entry.get("resource", {})
            if r.get("resourceType") == "Patient":
                patient_id = r.get("id")
                break

        client = cls(patient_id or "mock-patient")
        client._bundle = bundle
        client._mock_mode = True
        return client

    # ---- low-level helpers -------------------------------------------------
    def _get_mock(self, path: str) -> dict:
        """Serve a resource from the loaded Bundle, filtered by resourceType.

        Search paths (e.g. "Observation") return a Bundle of all matching
        resources; read paths (e.g. "Patient/<id>") return the single resource,
        matching how the live sandbox responds to each shape.
        """
        parts = path.split("/")
        resource_type = parts[0]
        matches = [
            e["resource"] for e in self._bundle.get("entry", [])
            if e.get("resource", {}).get("resourceType") == resource_type
        ]
        if len(parts) > 1:  # read by id, e.g. Patient/pt-001 -> bare resource
            for r in matches:
                if r.get("id") == parts[1]:
                    return r
            return matches[0] if matches else {}
        return {"resourceType": "Bundle", "entry": [{"resource": r} for r in matches]}

    def _get(self, path: str, params: dict = None) -> dict:
        """GET with 10s timeout + 1 retry. Never raises; returns {} on failure."""
        if self._mock_mode:
            return self._get_mock(path)
        url = f"{self.BASE_URL}/{path}"
        for attempt in range(2):
            try:
                resp = self.session.get(url, params=params, timeout=10)
                resp.raise_for_status()
                return resp.json()
            except requests.Timeout:
                if attempt == 0:
                    _log(f"  ⏱ Timeout {path}, retrying...")
                    continue
                _log(f"  ✗ Timeout {path} sau 2 lần thử")
                self.missing_resources.append(path.split("/")[0])
                return {}
            except Exception as e:
                _log(f"  ✗ Error {path}: {e}")
                self.missing_resources.append(path.split("/")[0])
                return {}
        return {}

    def _get_entries(self, bundle: dict) -> list:
        """Extract resource list from a FHIR Bundle."""
        return [e["resource"] for e in bundle.get("entry", [])]

    def _lookup_loinc(self, code: str) -> str:
        """Translate a LOINC code -> short name via SQLite (fallback: the code)."""
        row = self.db.execute(
            "SELECT short_name FROM loinc_codes WHERE loinc_code = ?", (code,)
        ).fetchone()
        return row[0] if row and row[0] else code

    def _lookup_icd10(self, code: str) -> tuple:
        """Translate an ICD-10 code -> (name_vi, name_en) via SQLite."""
        row = self.db.execute(
            "SELECT name_vi, name_en FROM icd10_codes WHERE code = ?", (code,)
        ).fetchone()
        return (row[0] or "", row[1] or "") if row else ("", "")

    # ---- 4.1 AllergyIntolerance -------------------------------------------
    def get_allergies(self) -> list:
        bundle = self._get(
            "AllergyIntolerance",
            {"patient": self.patient_id, "clinical-status": "active"},
        )
        out = []
        for r in self._get_entries(bundle):
            code = r.get("code", {})
            allergen = code.get("text")
            if not allergen and code.get("coding"):
                allergen = code["coding"][0].get("display")
            reaction = None
            reactions = r.get("reaction", [])
            if reactions:
                manifestation = reactions[0].get("manifestation", [])
                if manifestation:
                    mcoding = manifestation[0].get("coding", [])
                    reaction = manifestation[0].get("text")
                    if not reaction and mcoding:
                        reaction = mcoding[0].get("display")
            out.append({
                "allergen": allergen,
                "category": (r.get("category") or [None])[0],
                "criticality": r.get("criticality"),
                "reaction": reaction,
            })
        _log(f"AllergyIntolerance: {len(out)} found")
        return out

    # ---- 4.2 Patient -------------------------------------------------------
    def get_patient(self) -> dict:
        r = self._get(f"Patient/{self.patient_id}")
        if not r:
            return {}
        name = (r.get("name") or [{}])[0]
        display = name.get("text") or \
            f"{name.get('family', '')} {' '.join(name.get('given', []))}".strip()
        birth = r.get("birthDate")
        age = None
        if birth:
            try:
                bd = datetime.strptime(birth, "%Y-%m-%d")
                age = relativedelta(datetime.now(), bd).years
            except ValueError:
                age = None
        result = {
            "id": r.get("id"),
            "name": display or "?",
            "gender": r.get("gender"),
            "birthDate": birth,
            "age": age,
        }
        _log(f"Patient: {result['name']}, {age} tuổi, {result['gender']}")
        return result

    # ---- 4.3 Encounter -----------------------------------------------------
    def get_encounter(self) -> dict:
        bundle = self._get(
            "Encounter",
            {"patient": self.patient_id, "status": "in-progress"},
        )
        entries = self._get_entries(bundle)
        if not entries:
            return {}
        r = entries[0]
        self.encounter_id = r.get("id")
        cls = r.get("class", {})
        service = r.get("serviceType", {})
        service_type = service.get("text")
        if not service_type and service.get("coding"):
            service_type = service["coding"][0].get("display")
        result = {
            "id": r.get("id"),
            "status": r.get("status"),
            "class": cls.get("code"),
            "service_type": service_type,
            "period_start": r.get("period", {}).get("start"),
            "reasons": [
                (rc.get("coding", [{}]) or [{}])[0].get("display")
                for rc in r.get("reasonCode", [])
            ],
            "locations": [
                loc.get("location", {}).get("display")
                for loc in r.get("location", [])
            ],
        }
        _log(f"Encounter: {result['id']} | {result['service_type']} | "
             f"từ {result['period_start']}")
        return result

    # ---- 4.4 Observation ---------------------------------------------------
    @staticmethod
    def _find_loinc_code(coding: list) -> str | None:
        for c in coding:
            if c.get("code") in ALL_LOINC:
                return c.get("code")
        return None

    @staticmethod
    def _extract_value(r: dict):
        """Return (value, unit) from an Observation, handling BP components."""
        if "valueQuantity" in r:
            vq = r["valueQuantity"]
            return vq.get("value"), vq.get("unit", "")
        if "valueCodeableConcept" in r:
            return r["valueCodeableConcept"].get("text"), ""
        if "component" in r:
            # blood pressure: take the systolic component (8480-6)
            for comp in r["component"]:
                codes = [c.get("code") for c in comp.get("code", {}).get("coding", [])]
                if "8480-6" in codes and "valueQuantity" in comp:
                    vq = comp["valueQuantity"]
                    return vq.get("value"), vq.get("unit", "")
        return None, None

    def get_observations(self) -> dict:
        params = {"patient": self.patient_id, "_sort": "-date", "_count": 100}
        if self.encounter_id:
            params["encounter"] = self.encounter_id
        bundle = self._get("Observation", params)

        # Seed every key so missing indices stay visible as None.
        result = {name: {"value": None, "unit": None, "timestamp": None,
                         "loinc": code, "name": name}
                  for code, name in ALL_LOINC.items()}

        for r in self._get_entries(bundle):
            coding = r.get("code", {}).get("coding", [])
            code = self._find_loinc_code(coding)
            if not code:
                continue
            key = ALL_LOINC[code]
            ts = r.get("effectiveDateTime") or r.get("issued") or ""
            existing = result[key]
            # keep the most recent reading
            if existing["value"] is not None and existing["timestamp"] and ts \
                    and ts <= existing["timestamp"]:
                continue
            value, unit = self._extract_value(r)
            if value is None:
                continue
            if isinstance(value, float):
                value = round(value, 1)
            # unit conversions
            if key == "creatinine" and unit and "mg/dl" in unit.lower():
                value, unit = round(value * 88.4, 1), "µmol/L"
            elif key == "temperature" and unit and unit.lower() in ("[degf]", "degf", "f"):
                value, unit = round((value - 32) * 5 / 9, 1), "°C"
            label = self._lookup_loinc(code)
            result[key] = {
                "value": value,
                "unit": unit,
                "timestamp": ts,
                "loinc": code,
                "name": key if label == code else label,
            }

        found = sum(1 for v in result.values() if v["value"] is not None)
        missing = [k for k, v in result.items() if v["value"] is None]
        _log(f"Observations: {found} found | missing: {missing}")
        return result

    # ---- 4.5 MedicationRequest --------------------------------------------
    @staticmethod
    def _med_name(r: dict) -> str | None:
        mcc = r.get("medicationCodeableConcept", {})
        if mcc.get("text"):
            return mcc["text"]
        if mcc.get("coding"):
            return mcc["coding"][0].get("display")
        return r.get("medicationReference", {}).get("display")

    def get_medications(self) -> list:
        params = {"patient": self.patient_id, "status": "active"}
        if self.encounter_id:
            params["encounter"] = self.encounter_id
        bundle = self._get("MedicationRequest", params)
        out = []
        for r in self._get_entries(bundle):
            di = (r.get("dosageInstruction") or [{}])[0]
            dar = (di.get("doseAndRate") or [{}])[0]
            dq = dar.get("doseQuantity", {})
            dose = f"{dq.get('value', '')} {dq.get('unit', '')}".strip()
            route = (di.get("route", {}).get("coding", [{}]) or [{}])[0].get("display", "")
            frequency = di.get("timing", {}).get("code", {}).get("text", "")
            out.append({
                "name": self._med_name(r),
                "dose": dose,
                "route": route,
                "frequency": frequency,
            })
        _log(f"MedicationRequest: {len(out)} active")
        return out

    # ---- 4.6 Condition -----------------------------------------------------
    def get_conditions(self) -> list:
        bundle = self._get(
            "Condition",
            {"patient": self.patient_id, "clinical-status": "active"},
        )
        out = []
        for r in self._get_entries(bundle):
            code_block = r.get("code", {})
            coding = code_block.get("coding", [])
            icd_code = None
            icd_display = ""
            for c in coding:
                system = (c.get("system") or "").lower()
                if "icd-10" in system or "icd10" in system:
                    icd_code = c.get("code")
                    icd_display = c.get("display") or ""
                    break
            if not icd_code and coding:
                icd_code = coding[0].get("code")
            # Prefer the resource's own clean names; the SQLite lookup (parsed from
            # icd-10_vn.md) is a noisy fallback for codes the resource doesn't name.
            fhir_text = code_block.get("text")                      # Vietnamese
            fhir_display = icd_display or (coding[0].get("display") if coding else "")  # English
            db_vi, db_en = self._lookup_icd10(icd_code) if icd_code else ("", "")
            name_vi = fhir_text or db_vi
            name_en = fhir_display or db_en
            display = fhir_text or fhir_display or db_vi or ""
            out.append({
                "icd10_code": icd_code or "?",
                "name_vi": name_vi,
                "name_en": name_en,
                "display": display,
                "severity": (r.get("severity", {}).get("coding", [{}]) or [{}])[0].get("display", ""),
                "onset": r.get("onsetDateTime", ""),
            })
        _log(f"Conditions: {len(out)} active")
        for c in out:
            label = c["name_vi"] or c["display"] or "?"
            _log(f"  {c['icd10_code']} → {label}")
        return out

    # ---- 4.7 MedicationAdministration -------------------------------------
    def get_medication_administrations(self) -> list:
        params = {
            "patient": self.patient_id,
            "status": "completed,in-progress",
            "_sort": "-effective-time",
            "_count": 20,
        }
        if self.encounter_id:
            params["encounter"] = self.encounter_id
        bundle = self._get("MedicationAdministration", params)
        out = []
        for r in self._get_entries(bundle):
            dosage = r.get("dosage", {})
            dose_q = dosage.get("dose", {})
            dose = f"{dose_q.get('value', '')} {dose_q.get('unit', '')}".strip()
            route = (dosage.get("route", {}).get("coding", [{}]) or [{}])[0].get("display", "")
            effective = r.get("effectiveDateTime") or \
                r.get("effectivePeriod", {}).get("start") or ""
            out.append({
                "name": self._med_name(r),
                "dose": dose,
                "route": route,
                "effective": effective,
                "status": r.get("status"),
            })
        _log(f"MedicationAdministration: {len(out)} records")
        return out

    # ---- 4.8 Procedure -----------------------------------------------------
    def get_procedures(self) -> list:
        params = {"patient": self.patient_id, "_sort": "-date", "_count": 20}
        if self.encounter_id:
            params["encounter"] = self.encounter_id
        bundle = self._get("Procedure", params)
        out = []
        for r in self._get_entries(bundle):
            code = r.get("code", {})
            name = code.get("text")
            if not name and code.get("coding"):
                name = code["coding"][0].get("display")
            performed = r.get("performedDateTime") or \
                r.get("performedPeriod", {}).get("start") or ""
            body = r.get("bodySite", [])
            body_site = body[0].get("display") if body else None
            out.append({
                "name": name,
                "status": r.get("status"),
                "performed": performed,
                "bodySite": body_site,
            })
        _log(f"Procedure: {len(out)} records")
        return out

    # ---- 4.9 DiagnosticReport ---------------------------------------------
    def get_diagnostic_reports(self) -> list:
        bundle = self._get(
            "DiagnosticReport",
            {"patient": self.patient_id, "_sort": "-date", "_count": 5},
        )
        out = []
        for r in self._get_entries(bundle):
            code = r.get("code", {})
            title = code.get("text")
            if not title and code.get("coding"):
                title = code["coding"][0].get("display")
            out.append({
                "title": title,
                "status": r.get("status"),
                "date": r.get("effectiveDateTime") or r.get("issued") or "",
            })
        _log(f"DiagnosticReport: {len(out)} recent")
        return out

    # ---- Task 5: consolidate ----------------------------------------------
    def build_patient_context(self) -> dict:
        """Run the 9 queries in fixed order; Encounter first so its id can filter."""
        start = time.time()

        allergies = self.get_allergies()
        patient = self.get_patient()
        encounter = self.get_encounter()           # sets encounter_id
        observations = self.get_observations()
        medications = self.get_medications()
        conditions = self.get_conditions()
        administrations = self.get_medication_administrations()
        procedures = self.get_procedures()
        reports = self.get_diagnostic_reports()

        elapsed = round(time.time() - start, 2)

        return {
            "patient_id": self.patient_id,
            "patient": patient,
            "encounter": encounter,
            "allergies": allergies,
            "observations": observations,
            "medications": medications,
            "conditions": conditions,
            "medication_administrations": administrations,
            "procedures": procedures,
            "diagnostic_reports": reports,
            "query_time_s": elapsed,
            "missing_resources": self.missing_resources,
        }


# ---------------------------------------------------------------------------
# Task 6: readable output
# ---------------------------------------------------------------------------
def print_context(ctx: dict):
    p = ctx["patient"]
    e = ctx["encounter"]
    print(f"\n{'=' * 55}")
    print(f"PATIENT:   {p.get('name', '?')} | "
          f"{p.get('gender', '?')} | {p.get('age', '?')} tuổi")
    if e:
        print(f"ENCOUNTER: {e.get('id', '?')} | "
              f"{e.get('service_type', '?')} | "
              f"từ {e.get('period_start', '?')}")
    print(f"Query:     {ctx['query_time_s']}s | "
          f"Missing: {', '.join(ctx['missing_resources']) or 'none'}")
    print(f"{'=' * 55}")

    # Allergies
    print("\n[ALLERGIES]")
    if ctx["allergies"]:
        for a in ctx["allergies"]:
            print(f"  ⚠️  {a.get('allergen', '?')} "
                  f"→ {a.get('reaction', '?')} "
                  f"({a.get('criticality', '?')})")
    else:
        print("  Không có dị ứng được ghi nhận")

    # Observations
    print("\n[OBSERVATIONS]")
    for key, obs in ctx["observations"].items():
        if obs.get("value") is None:
            print(f"  {key:.<20} ⚠️  không có data")
            continue
        v = obs["value"]
        u = obs.get("unit", "") or ""
        t = (obs.get("timestamp") or "")[:10]
        n = obs.get("name", key)
        print(f"  {n:<20} {v} {u:<10} ({t})")

    # Medications
    print(f"\n[MEDICATION REQUEST — {len(ctx['medications'])} active]")
    for i, m in enumerate(ctx["medications"], 1):
        print(f"  {i}. {m.get('name', '?')} "
              f"{m.get('dose', '')} {m.get('route', '')} "
              f"{m.get('frequency', '')}")

    # Medication Administrations
    print(f"\n[MEDICATION ADMINISTRATION — "
          f"{len(ctx['medication_administrations'])} recent]")
    for m in ctx["medication_administrations"][:5]:
        print(f"  {(m.get('effective') or '?')[:10]} | "
              f"{m.get('name', '?')} {m.get('dose', '')} "
              f"[{m.get('status', '?')}]")

    # Conditions
    print(f"\n[CONDITIONS — {len(ctx['conditions'])} active]")
    for c in ctx["conditions"]:
        code = c.get("icd10_code", "?")
        vi = c.get("name_vi") or c.get("display", "?")
        en = c.get("name_en", "")
        print(f"  {code} → {vi}" + (f" ({en})" if en else ""))

    # Procedures
    print(f"\n[PROCEDURES — {len(ctx['procedures'])} records]")
    for pr in ctx["procedures"][:5]:
        print(f"  {(pr.get('performed') or '?')[:10]} | "
              f"{pr.get('name', '?')} [{pr.get('status', '?')}]")

    # DiagnosticReports
    print(f"\n[DIAGNOSTIC REPORTS — "
          f"{len(ctx['diagnostic_reports'])} recent]")
    for r in ctx["diagnostic_reports"]:
        print(f"  {(r.get('date') or '?')[:10]} | "
              f"{r.get('title', '?')} [{r.get('status', '?')}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FHIR R4 patient profile builder")
    parser.add_argument("--patient", help="FHIR Patient ID from the live sandbox")
    parser.add_argument("--file", help="load from a local FHIR Bundle JSON file")
    parser.add_argument("--find", action="store_true",
                        help="list sandbox patient IDs and exit")
    parser.add_argument("--json", action="store_true",
                        help="emit the consolidated profile as JSON")
    args = parser.parse_args()

    if args.find:
        find_patients()
        sys.exit(0)

    if args.file:
        client = FHIRClient.from_file(args.file)
    elif args.patient:
        client = FHIRClient(args.patient)
    else:
        parser.error("need --patient or --file (or --find)")
    context = client.build_patient_context()

    if args.json:
        print(json.dumps(context, ensure_ascii=False, indent=2))
    else:
        print_context(context)
