"""
Human-review harness for verifier safety fallbacks.

Re-runs the scenarios that hit `verifier_unsupported_safety` and prints, per claim:
verdict, safety flag, claim text, the model's self-declared evidence quote, AND the
cited chunk's real text — so a human can label each fallback as JUSTIFIED (the claim
truly is unsupported by its chunk) or OVER-BLOCK (the chunk does support it and the
verifier was too aggressive).

Usage:
  python src/rag/eval/review_fallbacks.py --only A-04,A-08 --gen-model qwen/qwen3.6-flash
"""

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/ on sys.path
from paths import MOCK_DIR  # noqa: E402
from fhir.fhir_client import FHIRClient  # noqa: E402
from scoring.calculator import calculate_all  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402
from rag.config import GEN_MODEL  # noqa: E402
from rag.eval.answer_eval import SCENARIOS  # noqa: E402

BY_ID = {s["id"]: s for s in SCENARIOS}


def review(ids: list[str], gen_model: str, backend: str) -> None:
    pipeline = RAGPipeline(gen_model=gen_model, verify=True, backend=backend)
    for sid in ids:
        sc = BY_ID[sid]
        ctx, calc = None, None
        if sc["patient"]:
            client = FHIRClient.from_file(str(MOCK_DIR / sc["patient"]))
            ctx = client.build_patient_context()
            calc = calculate_all(ctx)
        res = pipeline.ask(sc["query"], ctx, calc)
        resp, chunks = res["response"], res["chunks"]
        v = resp.get("verify") or {}

        print("=" * 100)
        print(f"{sid}  | expect={sc['expect']}  | model={gen_model}")
        print(f"Q: {sc['query']}")
        print(f"-> branch={v.get('branch')}  reason={resp['fallback_reason']}  "
              f"unsupported_ratio={v.get('unsupported_ratio')}  "
              f"is_ordered={v.get('is_ordered_procedure')}")
        verdicts = v.get("verdicts") or []
        if not verdicts:
            print("  (no per-claim verdicts — backend error or legacy path)")
        for j, c in enumerate(verdicts):
            cit = c.get("citation")
            chunk = chunks[cit - 1] if isinstance(cit, int) and 1 <= cit <= len(chunks) else None
            flag = "  <<< this claim forced the fallback" if (
                c.get("safety") and c.get("verdict") != "supported") else ""
            print(f"\n  [claim {j}] verdict={c.get('verdict').upper()}  "
                  f"safety={c.get('safety')}  cites=[{cit}]{flag}")
            print(f"    CLAIM    : {c.get('text', '').strip()}")
            print(f"    EVIDENCE : {c.get('evidence', '').strip() or '(none)'}")
            if chunk:
                print(f"    CHUNK[{cit}] ({chunk.get('title')}):")
                print("      " + chunk["text"][:900].replace("\n", "\n      "))
            else:
                print(f"    CHUNK[{cit}]: (citation out of range / missing)")
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="A-04,A-08", help="comma-separated scenario ids")
    ap.add_argument("--gen-model", default=GEN_MODEL)
    ap.add_argument("--backend", default="llm")
    args = ap.parse_args()
    review([s.strip() for s in args.only.split(",")], args.gen_model, args.backend)
