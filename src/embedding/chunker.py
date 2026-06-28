"""
Task 1 — 3-tier chunking of the ICU procedure guidelines (BYT VN 2014).

Tier 1 (procedure):          split on `## ` headings, keep real procedures.
Tier 2 (procedure_section):  split procedures > 6000 chars on `### ` sections.
Tier 3 (contraindication):   extract CHỐNG CHỈ ĐỊNH sections as safety chunks.

Run:  python src/embedding/chunker.py
Output: chunks/icu_chunks.json
"""

import json
import re
import sys
from pathlib import Path

# Windows consoles default to cp1252 and choke on Vietnamese — force UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ on sys.path
from paths import DATA_DIR, CHUNKS_FILE  # noqa: E402

DATA_FILE = DATA_DIR / "quy_trinh_icu_vn.md"
ICU2015_FILE = DATA_DIR / "icu_2015.md"
TT51_FILE = DATA_DIR / "tt51_phan_ve.md"
SSC_FILE = DATA_DIR / "ssc_2021.md"
OUTPUT_FILE = CHUNKS_FILE

SOURCE = "Quy trình ICU — BYT VN 2014"
ICU2015_SOURCE = "Hồi sức tích cực — BYT VN 2015"
TT51_SOURCE = "Thông tư 51/2017/TT-BYT — Phản vệ"
SSC_SOURCE = "Surviving Sepsis Campaign 2021"
MAX_CHARS = 6000
MIN_CHARS = 800
TT51_MIN_CHARS = 200
MIN_SAFETY_CHARS = 100

# A real procedure must contain at least one of these section markers.
STRUCTURE_MARKERS = ("CHỈ ĐỊNH", "TIẾN HÀNH", "CHUẨN BỊ", "ĐẠI CƯƠNG")

# Unicode fixes carried over from src/preprocessing/clean.py
UNICODE_FIXES = {
    "Ƣ": "Ư", "ƣ": "ư",
    "“": '"', "”": '"',
    "‘": "'", "’": "'",
}

H2_RE = re.compile(r"^##\s+(.*)$", re.MULTILINE)
H3_RE = re.compile(r"^###\s+", re.MULTILINE)
PAGENUM_RE = re.compile(r"\s+\d+\s*$")
# icu_2015 topics each begin with "### 1. ĐẠI CƯƠNG"; the title is the trailing
# ALL-CAPS run on the line just before this boundary.
ICU2015_TOPIC_RE = re.compile(r"^###\s+1\.\s*ĐẠI CƯƠNG", re.MULTILINE)
CONTRA_RE = re.compile(
    r"^###\s+[^\n]*CHỐNG CHỈ ĐỊNH[^\n]*$",  # heading line (content may follow it)
    re.MULTILINE,
)


INLINE_PROC_TITLE_RE = re.compile(
    r"(?<=[^\s#]) +(QUY TRÌNH KỸ THUẬT (?!TRONG )(?:(?!QUY TRÌNH|ĐẠI CƯƠNG)[^\n]){3,80}?)"
    r"\s+((?:I\.\s*)?ĐẠI CƯƠNG(?:\s*I\.)?)"
)


def _promote_inline_titles(text: str) -> tuple[str, int]:
    """Re-insert '## ' headers for procedure titles mashed inline into body text; returns (text, n)."""
    return INLINE_PROC_TITLE_RE.subn(r"\n\n## \1\n\n\2", text)


def load_and_clean(filepath: str | Path = DATA_FILE, promote_titles: bool = False) -> str:
    """Load markdown, apply unicode fixes, normalize whitespace; with `promote_titles`, recover
    '## ' headers lost inline in the body."""
    text = Path(filepath).read_text(encoding="utf-8")
    for wrong, right in UNICODE_FIXES.items():
        text = text.replace(wrong, right)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    if promote_titles:
        text, n = _promote_inline_titles(text)
        print(f"  [chunker] promoted {n} inline procedure title(s) to '## ' headers")
    return text


SECTION_MARKER_RE = re.compile(r"\s+[IVX]{1,4}\.\s")


def _clean_title(raw: str) -> str:
    """
    First heading line, cleaned. Many content headings merge the title with the
    body on one line (e.g. "... ĐẶT NỘI KHÍ QUẢN I. ĐẠI CƯƠNG Đặt nội..."), so we
    cut at the first roman-numeral section marker and strip the trailing page no.
    """
    title = raw.strip().splitlines()[0].strip() if raw.strip() else ""
    title = PAGENUM_RE.sub("", title)
    m = SECTION_MARKER_RE.search(title)
    if m:
        title = title[: m.start()]
    return title.strip(" .-")[:150]


def _split_h2_blocks(text: str) -> list[tuple[str, str]]:
    """Return [(title_line, block_text)] for each `## ` heading."""
    blocks = []
    matches = list(H2_RE.finditer(text))
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end].strip()
        blocks.append((m.group(1), block))
    return blocks


def _extract_contraindication(block: str) -> str | None:
    """Extract the CHỐNG CHỈ ĐỊNH section body (>100 chars) from a procedure block."""
    m = CONTRA_RE.search(block)
    if not m:
        return None
    # Section spans from this heading to the next ### / ## heading (or EOF).
    rest = block[m.start():]
    nxt = re.search(r"^(?:###|##)\s+", rest[3:], re.MULTILINE)  # skip own '###'
    section = rest[: nxt.start() + 3] if nxt else rest
    # Drop the heading line itself, keep the body.
    body = section.split("\n", 1)[1].strip() if "\n" in section else ""
    # Same-line content (e.g. "### III. CHỐNG CHỈ ĐỊNH Rối loạn...") -> recover it.
    head_inline = re.sub(r"^###\s+[^\n]*?CHỐNG CHỈ ĐỊNH", "", section.split("\n", 1)[0]).strip()
    full = (head_inline + "\n" + body).strip() if head_inline else body
    return full if len(full) >= MIN_SAFETY_CHARS else None


def _hard_wrap(text: str, cap: int) -> list[str]:
    """Wrap an oversized section on paragraph boundaries, each <= cap."""
    out, buf = [], ""
    for para in text.split("\n\n"):
        if len(buf) + len(para) + 2 > cap and buf:
            out.append(buf.strip())
            buf = para
        else:
            buf = f"{buf}\n\n{para}" if buf else para
    if buf.strip():
        out.append(buf.strip())
    return out


def _split_sections(block: str) -> list[str]:
    """
    Split an oversized procedure into chunks <= MAX_CHARS by *packing* `### `
    sections greedily (merging small sections so we don't emit tiny fragments).
    Sections that alone exceed MAX_CHARS are hard-wrapped on paragraphs.
    """
    parts = H3_RE.split(block)
    headers = [block[m.start():m.end()] for m in H3_RE.finditer(block)]
    sections = [parts[0]] + [headers[i] + parts[i + 1] for i in range(len(headers))]

    units: list[str] = []
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        units.extend(_hard_wrap(sec, MAX_CHARS) if len(sec) > MAX_CHARS else [sec])

    # Greedy pack adjacent units up to MAX_CHARS.
    packed: list[str] = []
    buf = ""
    for u in units:
        if buf and len(buf) + len(u) + 2 > MAX_CHARS:
            packed.append(buf)
            buf = u
        else:
            buf = f"{buf}\n\n{u}" if buf else u
    if buf:
        packed.append(buf)

    # Merge a tiny tail (< MIN_CHARS) into the previous chunk.
    if len(packed) >= 2 and len(packed[-1]) < MIN_CHARS:
        packed[-2] = f"{packed[-2]}\n\n{packed[-1]}"
        packed.pop()
    return packed


def chunk_procedures(text: str) -> list[dict]:
    """Main chunking function — returns chunk objects per the schema."""
    chunks: list[dict] = []
    counter = 0

    def new_id() -> str:
        nonlocal counter
        cid = f"icu_{counter:04d}"
        counter += 1
        return cid

    for title_line, block in _split_h2_blocks(text):
        if len(block) < MIN_CHARS:
            continue
        upper = block.upper()
        if not any(mk in upper for mk in STRUCTURE_MARKERS):
            continue

        title = _clean_title(title_line)
        has_contra = "CHỐNG CHỈ ĐỊNH" in upper
        has_steps = ("TIẾN HÀNH" in upper) or ("CÁC BƯỚC" in upper)

        # --- Tier 1 / Tier 2 ---
        if len(block) > MAX_CHARS:
            for sec in _split_sections(block):
                chunks.append({
                    "id": new_id(),
                    "text": sec,
                    "title": title,
                    "chunk_type": "procedure_section",
                    "source": SOURCE,
                    "language": "vi",
                    "metadata": {
                        "procedure_title": title,
                        "has_contraindication": "CHỐNG CHỈ ĐỊNH" in sec.upper(),
                        "has_steps": ("TIẾN HÀNH" in sec.upper()) or ("CÁC BƯỚC" in sec.upper()),
                        "is_partial": True,
                        "char_count": len(sec),
                        "type": "standard",
                    },
                })
        else:
            chunks.append({
                "id": new_id(),
                "text": block,
                "title": title,
                "chunk_type": "procedure",
                "source": SOURCE,
                "language": "vi",
                "metadata": {
                    "procedure_title": title,
                    "has_contraindication": has_contra,
                    "has_steps": has_steps,
                    "is_partial": False,
                    "char_count": len(block),
                    "type": "standard",
                },
            })

        # --- Tier 3 (safety) ---
        contra = _extract_contraindication(block)
        if contra:
            ctext = f"Chống chỉ định — {title}\n\n{contra}"
            chunks.append({
                "id": new_id(),
                "text": ctext,
                "title": title,
                "chunk_type": "contraindication",
                "source": SOURCE,
                "language": "vi",
                "metadata": {
                    "procedure_title": title,
                    "has_contraindication": True,
                    "has_steps": False,
                    "is_partial": False,
                    "char_count": len(ctext),
                    "type": "safety_critical",
                },
            })

    _print_stats(chunks)
    return chunks


def _make_chunk(cid: str, text: str, title: str, chunk_type: str, source: str,
                is_partial: bool) -> dict:
    """Assemble a chunk object in the shared schema."""
    upper = text.upper()
    return {
        "id": cid,
        "text": text,
        "title": title,
        "chunk_type": chunk_type,
        "source": source,
        "language": "vi",
        "metadata": {
            "procedure_title": title,
            "has_contraindication": "CHỐNG CHỈ ĐỊNH" in upper,
            "has_steps": ("XỬ TRÍ" in upper) or ("TIẾN HÀNH" in upper) or ("CÁC BƯỚC" in upper),
            "is_partial": is_partial,
            "char_count": len(text),
            "type": "standard",
        },
    }


def _trailing_caps_title(pre: str) -> str:
    """Extract the trailing run of ALL-CAPS tokens (the icu_2015 topic title)."""
    tokens = pre.strip().split()
    out: list[str] = []
    for tok in reversed(tokens):
        letters = [c for c in tok if c.isalpha()]
        if letters and all(c.isupper() for c in letters):
            out.insert(0, tok)
        else:
            break
    title = " ".join(out).strip(" .:–-")
    # drop a leading roman-numeral/chapter fragment like "I:" or "Chương I:"
    title = re.sub(r"^[IVX]+[:.]?\s+", "", title)
    return title[:150]


def chunk_icu2015(text: str, start_index: int = 0) -> list[dict]:
    """Chunk the 2015 ICU diagnosis/management guideline (topic = ĐẠI CƯƠNG block)."""
    bounds = [m.start() for m in ICU2015_TOPIC_RE.finditer(text)]
    chunks: list[dict] = []
    counter = start_index
    for i, start in enumerate(bounds):
        end = bounds[i + 1] if i + 1 < len(bounds) else len(text)
        topic = text[start:end].strip()
        pre = text[(bounds[i - 1] if i > 0 else 0):start]
        title = _trailing_caps_title(pre) or f"Hồi sức tích cực 2015 — chủ đề {i + 1}"

        packed = _split_sections(topic)  # pack ### sections to <= MAX_CHARS
        partial = len(packed) > 1
        for sec in packed:
            chunks.append(_make_chunk(
                f"icu_{counter:04d}", sec, title,
                "procedure_section" if partial else "procedure",
                ICU2015_SOURCE, partial,
            ))
            counter += 1
    return chunks


def chunk_tt51(text: str, start_index: int = 0) -> list[dict]:
    """Chunk the 2017 anaphylaxis circular (Điều articles + Phụ lục appendices)."""
    chunks: list[dict] = []
    counter = start_index
    for title_line, block in _split_h2_blocks(text):
        if len(block) < TT51_MIN_CHARS:
            continue
        title = PAGENUM_RE.sub("", title_line.strip())[:150]
        packed = _split_sections(block) if len(block) > MAX_CHARS else [block]
        partial = len(packed) > 1
        for sec in packed:
            chunks.append(_make_chunk(
                f"icu_{counter:04d}", sec, title,
                "procedure_section" if partial else "procedure",
                TT51_SOURCE, partial,
            ))
            counter += 1
    return chunks


def _pack_lines(text: str, cap: int) -> list[str]:
    """Greedily pack lines into blocks <= cap chars (line is the atomic unit)."""
    out, buf = [], ""
    for line in text.split("\n"):
        if buf and len(buf) + len(line) + 1 > cap:
            out.append(buf.strip())
            buf = line
        else:
            buf = f"{buf}\n{line}" if buf else line
    if buf.strip():
        out.append(buf.strip())
    return out


def chunk_generic(text: str, source: str, language: str = "en",
                  chunk_type: str = "guideline", title: str = "Document",
                  start_index: int = 0) -> list[dict]:
    """Size-based chunker for structureless docs (no markdown headings).

    Used for the Surviving Sepsis Campaign 2021 guideline, which is two-column
    PDF text with no headings — packed into <= MAX_CHARS blocks on line
    boundaries. Sentences are imperfect (column interleaving in the source), but
    topical keywords stay intact for semantic retrieval.
    """
    blocks = [b for b in _pack_lines(text, MAX_CHARS) if len(b) >= MIN_CHARS]
    partial = len(blocks) > 1
    chunks: list[dict] = []
    for i, block in enumerate(blocks):
        chunks.append({
            "id": f"icu_{start_index + i:04d}",
            "text": block,
            "title": f"{title} — part {i + 1}",
            "chunk_type": chunk_type,
            "source": source,
            "language": language,
            "metadata": {
                "procedure_title": title,
                "has_contraindication": False,
                "has_steps": False,
                "is_partial": partial,
                "char_count": len(block),
                "type": "standard",
            },
        })
    return chunks


def _print_stats(chunks: list[dict]) -> None:
    by_type: dict[str, int] = {}
    for c in chunks:
        by_type[c["chunk_type"]] = by_type.get(c["chunk_type"], 0) + 1
    sizes = [len(c["text"]) for c in chunks] or [0]
    print(f"Total chunks: {len(chunks)}")
    print(f"  - procedure:           {by_type.get('procedure', 0)}")
    print(f"  - procedure_section:   {by_type.get('procedure_section', 0)}")
    print(f"  - contraindication:    {by_type.get('contraindication', 0)}")
    if by_type.get("guideline"):
        print(f"  - guideline:           {by_type.get('guideline', 0)}")
    print(f"Size: min={min(sizes)} / max={max(sizes)} / avg={sum(sizes) // len(sizes)}")


def save_chunks(chunks: list[dict], output_path: str | Path = OUTPUT_FILE) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nSaved: {output_path}")


def validate_chunks(chunks: list[dict]) -> bool:
    required_top = {"id", "text", "title", "chunk_type", "source", "language", "metadata"}
    print("\nValidation:")

    no_empty = all(c.get("text", "").strip() for c in chunks)
    ids = [c["id"] for c in chunks]
    no_dup = len(ids) == len(set(ids))
    all_fields = all(required_top.issubset(c.keys()) for c in chunks)
    enough = len(chunks) >= 200

    def mark(ok: bool) -> str:
        return "✅" if ok else "❌"

    print(f"  {mark(no_empty)} No empty chunks")
    print(f"  {mark(no_dup)} No duplicate IDs")
    print(f"  {mark(all_fields)} All required fields present")
    print(f"  {mark(enough)} Chunk count >= 200 (got {len(chunks)})")

    passed = no_empty and no_dup and all_fields and enough
    print(f"  {mark(passed)} All chunks pass")
    return passed


def main() -> None:
    print(f"Loading {DATA_FILE.name}...")
    chunks = chunk_procedures(load_and_clean(DATA_FILE, promote_titles=True))
    print(f"  -> {len(chunks)} from BYT 2014 procedures\n")

    print(f"Loading {ICU2015_FILE.name}...")
    icu2015 = chunk_icu2015(load_and_clean(ICU2015_FILE))
    print(f"  -> {len(icu2015)} from Hồi sức tích cực 2015\n")

    print(f"Loading {TT51_FILE.name}...")
    tt51 = chunk_tt51(load_and_clean(TT51_FILE))
    print(f"  -> {len(tt51)} from Thông tư 51/2017 (phản vệ)\n")

    ssc = []
    if SSC_FILE.exists():
        print(f"Loading {SSC_FILE.name}...")
        ssc = chunk_generic(load_and_clean(SSC_FILE), SSC_SOURCE, language="en",
                            title="Surviving Sepsis Campaign 2021")
        print(f"  -> {len(ssc)} from Surviving Sepsis Campaign 2021\n")

    chunks = chunks + icu2015 + tt51 + ssc
    # Re-assign globally unique, sequential IDs across all sources.
    for i, c in enumerate(chunks):
        c["id"] = f"icu_{i:04d}"

    print("Combined corpus:")
    _print_stats(chunks)
    ok = validate_chunks(chunks)
    save_chunks(chunks)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
