# ASR + RAG Clinical Assistant — ICU Decision Support

A proof-of-concept that lets an ICU clinician ask a clinical question by voice and get
cited treatment guidance in seconds. This repository holds the **data + retrieval + clinical
scoring** backend: a knowledge base (ICU guidelines), a FHIR R4 patient-context builder, and
clinical risk calculators.

> Full product spec lives in [requirement_analysis/](requirement_analysis/) (PRD, user flow,
> evaluation plan, research background).

---

## Repository layout

```
src/
  paths.py              # single source of truth for repo-relative paths
  preprocessing/        # PDF extraction, markdown cleaning, quality checks
  db/                   # build_clinical_db.py -> db/clinical_db.sqlite (LOINC + ICD-10)
  embedding/            # chunker, embedder, retriever, or_client (OpenRouter embed + chat)
  fhir/                 # fhir_client.py (+ from_file/--file), generate_mock_patients.py
  scoring/              # calculator.py — MAP, qSOFA, SOFA, NEWS2, eGFR
  rag/                  # ask.py CLI, pipeline, router, safety gate, generator
    eval/               # gold_retrieval.json, retrieval_eval.py, answer_eval.py
tests/                  # pytest unit tests (test_calculator.py, test_chunker.py, test_rag.py)
data/
  mock/                 # 17 committed mock FHIR bundles (patient_A..Q) + index.json
  *.md / *.pdf          # source corpora (gitignored)
db/                     # clinical_db.sqlite (gitignored build artifact)
chroma_db/  chunks/     # vector store + chunk artifacts (gitignored)
```

Every module resolves files through [src/paths.py](src/paths.py) (`DB_PATH`, `MOCK_DIR`,
`CHROMA_PATH`, `ENV_FILE`, …) by adding `src/` to `sys.path` — no per-file `parents[N]` guessing.

---

## Setup

```powershell
conda activate vsf
pip install -r requirements.txt
```

Secrets live in `.env` at the repo root (loaded by `src/embedding/or_client.py`):

```
OPEN_ROUTER_KEY=sk-or-...
```

> **Windows note:** run the env's python directly (`python ...`) rather than `conda run`, which
> re-encodes stdout through cp1252 and crashes on Vietnamese text. The modules force UTF-8 stdout
> themselves; if piping, also set `PYTHONIOENCODING=utf-8`.

---

## Components

### 1. Clinical lookup DB
```powershell
python src/db/build_clinical_db.py
```
Builds `db/clinical_db.sqlite` with `loinc_codes` (28 ICU LOINC codes) and `icd10_codes`
(~7,900 ICD-10 codes parsed from `data/icd-10_vn.md`).

> Known issue: the ICD-10 Vietnamese/English names are noisy (best-effort parse of a messy
> bilingual source). The FHIR client treats this lookup only as a **fallback** — see below.

### 2. Embedding / retrieval (RAG knowledge base)
```powershell
python src/embedding/chunker.py        # guidelines -> chunks/icu_chunks.json
python src/embedding/embedder.py       # chunks -> chroma_db (needs OPEN_ROUTER_KEY)
python src/embedding/retriever.py      # 10-query evaluation
```

### 3. FHIR patient context — `src/fhir/fhir_client.py`
Pulls 9 FHIR R4 resources (Patient, Encounter, AllergyIntolerance, Observation,
MedicationRequest, Condition, MedicationAdministration, Procedure, DiagnosticReport) and
consolidates them into one `patient_context` dict (demographics, vitals/labs, conditions,
meds, …).

Two sources:
```powershell
# Local mock bundle (offline, recommended for dev)
python src/fhir/fhir_client.py --file data/mock/patient_A.json
python src/fhir/fhir_client.py --file data/mock/patient_A.json --json

# Live SMART Health IT R4 sandbox
python src/fhir/fhir_client.py --find                 # list sandbox patient IDs
python src/fhir/fhir_client.py --patient <id>
```
Status/progress prints go to **stderr**, so `--json` keeps stdout pure. Observations carry unit
conversions (creatinine mg/dL→µmol/L, temp °F→°C) and keep most-recent readings.

Condition names prefer the resource's own `code.text` / `coding.display` (clean), falling back to
the SQLite lookup only when the resource omits a name.

### 4. Mock patient cohort — `data/mock/`
17 hand-curated + generated FHIR bundles (`patient_A.json` … `patient_Q.json`) plus
[index.json](data/mock/index.json) mapping each to scenarios and edge cases. The sandbox only
serves outpatient Synthea data, so these provide ICU cases (sepsis, ARDS, AKI, MI, hepatic
failure) and edge cases (unit conversions, missing vitals, GCS-as-text, no encounter, empty
sections, extreme values).

Regenerate patients H–Q (A–G are hand-written):
```powershell
python src/fhir/generate_mock_patients.py
```

### 5. Clinical calculators — `src/scoring/calculator.py`
Derives 5 scores from a `patient_context` dict (pure functions, no network):

| Score | Notes |
|-------|-------|
| **MAP** | `(SBP + 2·DBP)/3`, SSC-2021 bands |
| **qSOFA** | GCS<15, RR≥22, SBP≤100; positive if ≥2 |
| **SOFA** | 5/6 organs (pulmonary skipped w/o FiO₂); cardiovascular is dose-aware per the SOFA table |
| **NEWS2** | Scale 1/2 (hypercapnic), per-parameter scoring + risk level |
| **eGFR** | CKD-EPI 2021 (race-free) + CKD staging + dose-adjustment flag |

```powershell
python src/scoring/calculator.py --file data/mock/patient_A.json
python src/scoring/calculator.py --file data/mock/patient_A.json --json
```
`calculate_all()` returns `{map, qsofa, sofa, news2, egfr, summary}` ready to attach to the
patient context for the RAG pipeline. Scoring math coerces values with `get_obs_number` so a
GCS recorded as text (`"10 (E2 V3 M5)"`) never crashes; eGFR guards `age=None`.

### 6. RAG module — `src/rag/`
End-to-end cited Q&A (plan: [requirement_analysis/06_RAG_MODULE_PLAN.md](requirement_analysis/06_RAG_MODULE_PLAN.md)):

```powershell
python src/rag/ask.py --file data/mock/patient_A.json --query "Bệnh nhân dị ứng Penicillin, dùng Amoxicillin được không?"
python src/rag/ask.py --query "chống chỉ định đặt nội khí quản"   # guideline-only, no patient
python src/rag/ask.py --file ... --query "..." --json             # machine-readable
```

Pipeline: LLM intent router (`query_router.py`) → safety-priority retrieval →
allergy gate (`safety.py`, alert always renders first) → grounded generation
(`generator.py`). The hallucination guard is **code-enforced**: a non-fallback
answer without a valid `[n]` citation is replaced by the "Không đủ thông tin"
fallback (F-RAG-09); scoring-intent answers are grounded in `calculate_all()`
instead. Generation model: `qwen/qwen3.6-flash`; judge: `openai/gpt-5.4`
(`src/rag/config.py`).

**Evaluation:**
```powershell
python src/rag/eval/retrieval_eval.py   # Hit@1 / Recall@5 / MRR on 45-query gold set
python src/rag/eval/answer_eval.py      # 12 scenarios + GPT-5.4 judge -> chunks/rag_eval_report.md
```

---

## Tests

```powershell
pytest tests/ -v
```
40 tests: MAP / qSOFA / NEWS2 / eGFR / conversions plus regressions for GCS-as-string and
age-None (`test_calculator.py`), chunker schema/packing (`test_chunker.py`), and the RAG
safety gate / citation guard / fallback contract with a mocked LLM (`test_rag.py`).

---

## Conventions
- Modules force UTF-8 stdout for Vietnamese on Windows consoles.
- New `src/<pkg>` modules: add `__init__.py`, bootstrap `src/` onto `sys.path`, import shared
  paths from `paths`.
- Mock FHIR bundles under `data/mock/` are committed (they replace the sandbox); other `data/`,
  `db/`, `chroma_db/`, `chunks/` artifacts are gitignored.
