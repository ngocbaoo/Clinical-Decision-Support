"""
Task 0 — Build clinical_db.sqlite (LOINC + ICD-10 lookup tables).

Sources:
    data/loinc_icu_codes.csv  -> table `loinc_codes`  (28 ICU-relevant LOINC codes)
    data/icd-10_vn.md         -> table `icd10_codes`  (bilingual ICD-10, best-effort parse)

The DB is a standalone lookup artifact (not wired into the vector retriever).
Run:  python src/db/build_clinical_db.py
"""

import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

# Windows consoles default to cp1252 and choke on Vietnamese — force UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Paths (resolve relative to week 2/ regardless of CWD)
# ---------------------------------------------------------------------------
WEEK2_DIR = Path(__file__).resolve().parents[2]   # .../week 2
DATA_DIR = WEEK2_DIR / "data"
DB_DIR = WEEK2_DIR / "db"
DB_PATH = DB_DIR / "clinical_db.sqlite"

LOINC_CSV = DATA_DIR / "loinc_icu_codes.csv"
ICD10_MD = DATA_DIR / "icd-10_vn.md"

# ---------------------------------------------------------------------------
# ICD-10 chapter ranges (deterministic from the code — body markers are noisy)
#   (start_letter, start_num, end_letter, end_num, roman, name_vi)
# ---------------------------------------------------------------------------
ICD10_CHAPTERS = [
    ("A", 0, "B", 99, "I", "Bệnh nhiễm trùng và ký sinh trùng"),
    ("C", 0, "D", 48, "II", "Bướu tân sinh"),
    ("D", 50, "D", 89, "III", "Bệnh của máu, cơ quan tạo máu và rối loạn miễn dịch"),
    ("E", 0, "E", 90, "IV", "Bệnh nội tiết, dinh dưỡng và chuyển hóa"),
    ("F", 0, "F", 99, "V", "Rối loạn tâm thần và hành vi"),
    ("G", 0, "G", 99, "VI", "Bệnh hệ thần kinh"),
    ("H", 0, "H", 59, "VII", "Bệnh mắt và phần phụ"),
    ("H", 60, "H", 95, "VIII", "Bệnh tai và xương chũm"),
    ("I", 0, "I", 99, "IX", "Bệnh hệ tuần hoàn"),
    ("J", 0, "J", 99, "X", "Bệnh hệ hô hấp"),
    ("K", 0, "K", 93, "XI", "Bệnh hệ tiêu hóa"),
    ("L", 0, "L", 99, "XII", "Bệnh da và mô dưới da"),
    ("M", 0, "M", 99, "XIII", "Bệnh hệ cơ, xương, khớp và mô liên kết"),
    ("N", 0, "N", 99, "XIV", "Bệnh hệ sinh dục - tiết niệu"),
    ("O", 0, "O", 99, "XV", "Thai nghén, sinh đẻ và hậu sản"),
    ("P", 0, "P", 96, "XVI", "Bệnh lý xuất phát trong thời kỳ chu sinh"),
    ("Q", 0, "Q", 99, "XVII", "Dị tật bẩm sinh, biến dạng và bất thường nhiễm sắc thể"),
    ("R", 0, "R", 99, "XVIII", "Triệu chứng, dấu hiệu và biểu hiện bất thường"),
    ("S", 0, "T", 98, "XIX", "Chấn thương, ngộ độc và hậu quả do nguyên nhân bên ngoài"),
    ("V", 1, "Y", 98, "XX", "Nguyên nhân ngoại sinh của bệnh tật và tử vong"),
    ("Z", 0, "Z", 99, "XXI", "Yếu tố ảnh hưởng sức khỏe và tiếp xúc dịch vụ y tế"),
    ("U", 0, "U", 99, "XXII", "Mã phục vụ mục đích đặc biệt"),
]

CODE_RE = re.compile(r"\b([A-Z]\d{2}(?:\.\d+)?)\b")


def _code_key(code: str) -> tuple[int, int]:
    """Sort/compare key for a code base, e.g. 'A00.1' -> (ord('A'), 0)."""
    base = code.split(".")[0]
    return (ord(base[0]), int(base[1:3]))


def chapter_for(code: str) -> tuple[str | None, str | None]:
    """Return (roman, name_vi) chapter for an ICD-10 code, or (None, None)."""
    key = _code_key(code)
    for sl, sn, el, en, roman, name in ICD10_CHAPTERS:
        if (ord(sl), sn) <= key <= (ord(el), en):
            return roman, name
    return None, None


# ---------------------------------------------------------------------------
# LOINC
# ---------------------------------------------------------------------------
def build_loinc(conn: sqlite3.Connection) -> int:
    df = pd.read_csv(LOINC_CSV)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]

    conn.execute("DROP TABLE IF EXISTS loinc_codes")
    conn.execute(
        """
        CREATE TABLE loinc_codes (
            loinc_code TEXT PRIMARY KEY,
            full_name  TEXT,
            short_name TEXT,
            component  TEXT,
            system     TEXT,
            unit       TEXT,
            class      TEXT,
            category   TEXT
        )
        """
    )
    cols = ["loinc_code", "full_name", "short_name", "component",
            "system", "unit", "class", "category"]
    rows = [tuple(r) for r in df[cols].astype(object).where(df[cols].notna(), None).values]
    conn.executemany(
        f"INSERT OR REPLACE INTO loinc_codes ({','.join(cols)}) "
        f"VALUES ({','.join('?' * len(cols))})",
        rows,
    )
    conn.commit()
    return len(rows)


# ---------------------------------------------------------------------------
# ICD-10
# ---------------------------------------------------------------------------
def parse_icd10(text: str) -> list[tuple]:
    """
    Best-effort parse of the bilingual layout `CODE En-name CODE Vi-name`.

    The doc repeats each leaf code twice in a row (English desc, then Vietnamese
    desc). We tokenise the whole stream by code occurrences and pair consecutive
    identical codes. Range headings (A00-A09) are not matched by CODE_RE (the dash
    breaks the word boundary into a separate token), so they are naturally skipped.
    """
    # Flatten to a single whitespace-normalised stream so wrapped lines join up.
    stream = re.sub(r"\s+", " ", text)

    # Skip the front matter (committee list + usage guide). The real leaf-code
    # tables begin at the first "A00 Cholera" entry; everything before it only
    # mentions codes in prose/examples and pollutes the table.
    anchor = re.search(r"A00\s+Cholera", stream)
    if anchor:
        stream = stream[anchor.start():]

    matches = list(CODE_RE.finditer(stream))

    records: dict[str, tuple] = {}   # code -> (code, name_en, name_vi, chapter)
    i = 0
    n = len(matches)
    while i < n:
        code = matches[i].group(1)
        # Only accept *paired* occurrences (CODE En ... CODE Vi). Single mentions
        # are almost always prose references in the intro/usage section, which
        # otherwise pollute the table with garbage descriptions.
        if i + 1 < n and matches[i + 1].group(1) == code:
            en = stream[matches[i].end():matches[i + 1].start()].strip(" .-")
            vi_end = matches[i + 2].start() if i + 2 < n else len(stream)
            vi = stream[matches[i + 1].end():vi_end].strip(" .-")
            if code not in records and (en or vi):
                roman, _ = chapter_for(code)
                records[code] = (code, en or None, vi or None, roman)
            i += 2
        else:
            i += 1

    return list(records.values())


def build_icd10(conn: sqlite3.Connection) -> int:
    text = ICD10_MD.read_text(encoding="utf-8")
    rows = parse_icd10(text)

    conn.execute("DROP TABLE IF EXISTS icd10_codes")
    conn.execute(
        """
        CREATE TABLE icd10_codes (
            code    TEXT PRIMARY KEY,
            name_en TEXT,
            name_vi TEXT,
            chapter TEXT
        )
        """
    )
    conn.executemany(
        "INSERT OR REPLACE INTO icd10_codes (code, name_en, name_vi, chapter) "
        "VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_icd10_chapter ON icd10_codes(chapter)")
    conn.commit()
    return len(rows)


# ---------------------------------------------------------------------------
def _print_samples(conn: sqlite3.Connection, table: str, n: int = 3) -> None:
    cur = conn.execute(f"SELECT * FROM {table} LIMIT {n}")
    cols = [d[0] for d in cur.description]
    for row in cur.fetchall():
        rec = {c: (str(v)[:60] if v is not None else None) for c, v in zip(cols, row)}
        print(f"    {rec}")


def main() -> None:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Building clinical DB at: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    try:
        print("\nLoading LOINC codes...")
        n_loinc = build_loinc(conn)
        print(f"  loinc_codes: {n_loinc} rows")
        _print_samples(conn, "loinc_codes")

        print("\nParsing ICD-10 codes...")
        n_icd = build_icd10(conn)
        print(f"  icd10_codes: {n_icd} rows")
        _print_samples(conn, "icd10_codes")
    finally:
        conn.close()

    size_mb = DB_PATH.stat().st_size / (1024 * 1024)
    print(f"\nDone. SQLite saved to: {DB_PATH} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
