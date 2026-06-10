"""Unit tests for the generic (size-based) chunker used for English guidelines."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from embedding.chunker import (  # noqa: E402
    _pack_lines, chunk_generic, MAX_CHARS, MIN_CHARS,
)
from embedding.embedder import _flatten_metadata  # noqa: E402


def _sample_text(n_lines: int = 4000, line: str = "Sepsis management line with some words.") -> str:
    return "\n".join(f"{line} {i}" for i in range(n_lines))


# ── _pack_lines ────────────────────────────────────────
def test_pack_lines_respects_cap():
    blocks = _pack_lines(_sample_text(), MAX_CHARS)
    assert blocks
    assert all(len(b) <= MAX_CHARS for b in blocks)


def test_pack_lines_preserves_lines():
    text = _sample_text(500)
    blocks = _pack_lines(text, MAX_CHARS)
    # No line is split mid-line: re-joining all block lines == original lines.
    rejoined = "\n".join("\n".join(b.split("\n")) for b in blocks).split("\n")
    assert rejoined == text.split("\n")


def test_pack_lines_single_block_when_small():
    assert len(_pack_lines("one\ntwo\nthree", MAX_CHARS)) == 1


# ── chunk_generic ──────────────────────────────────────
def test_chunk_generic_under_cap_and_multiple():
    chunks = chunk_generic(_sample_text(), "Surviving Sepsis Campaign 2021",
                           title="SSC 2021")
    assert len(chunks) > 1
    assert all(len(c["text"]) <= MAX_CHARS for c in chunks)


def test_chunk_generic_schema_and_ids():
    chunks = chunk_generic(_sample_text(), "Surviving Sepsis Campaign 2021",
                           title="SSC 2021", start_index=5)
    first = chunks[0]
    assert first["id"] == "icu_0005"
    assert chunks[1]["id"] == "icu_0006"          # sequential from start_index
    assert first["source"] == "Surviving Sepsis Campaign 2021"
    assert first["language"] == "en"
    assert first["chunk_type"] == "guideline"
    assert first["metadata"]["is_partial"] is True  # >1 chunk
    # Must carry the full metadata schema the embedder flattens (no KeyError).
    flat = _flatten_metadata(first)
    assert flat["source"] == "Surviving Sepsis Campaign 2021"
    assert flat["language"] == "en"
    assert flat["char_count"] == len(first["text"])


def test_chunk_generic_drops_tiny_blocks():
    # A single short line is below MIN_CHARS and should produce no chunks.
    assert chunk_generic("too short", "X", title="X") == []
    assert MIN_CHARS > len("too short")
