"""
Per-request structured logging for the RAG pipeline.

One JSON line per `RAGPipeline.ask()` call is appended to `logs/rag-YYYYMMDD.jsonl`, giving
an auditable trace: query -> routing -> retrieved chunks -> safety alerts -> raw generation
-> verifier verdicts/branch -> final answer + timings. Writing is best-effort and wrapped so
logging can never raise into (or slow down a failure of) the request path.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ on sys.path
from paths import LOG_DIR  # noqa: E402


def log_request(record: dict, log_dir: Path = LOG_DIR) -> None:
    """Append one JSON line to logs/rag-YYYYMMDD.jsonl. Never raises."""
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        record = {"ts": datetime.now(timezone.utc).isoformat(), **record}
        path = log_dir / f"rag-{datetime.now().strftime('%Y%m%d')}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception as err:  # noqa: BLE001 — logging must never break a request
        print(f"  [log_request failed] {err}", file=sys.stderr)
