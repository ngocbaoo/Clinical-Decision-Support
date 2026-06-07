"""
Central filesystem paths — single source of truth for the whole project.

Every module previously recomputed the repo root with a brittle
`Path(__file__).resolve().parents[N]` (and the N differed per file, one of them
even off-by-one). Instead, modules now bootstrap the `src/` directory onto
sys.path and import what they need from here:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/
    from paths import DB_PATH  # noqa: E402

All paths are absolute and resolved relative to this file's location, so they
work regardless of the current working directory.
"""

from pathlib import Path

# src/paths.py -> parents[0] = src/, parents[1] = repo root
ROOT = Path(__file__).resolve().parents[1]

# Raw inputs / corpora
DATA_DIR = ROOT / "data"
MOCK_DIR = DATA_DIR / "mock"

# Lookup DB (LOINC + ICD-10)
DB_DIR = ROOT / "db"
DB_PATH = DB_DIR / "clinical_db.sqlite"

# Vector store
CHROMA_PATH = ROOT / "chroma_db"

# Chunking pipeline artifacts
CHUNKS_DIR = ROOT / "chunks"
CHUNKS_FILE = CHUNKS_DIR / "icu_chunks.json"
REPORT_FILE = CHUNKS_DIR / "evaluation_report.md"

# Secrets
ENV_FILE = ROOT / ".env"
