"""
R5 — CLI for the RAG pipeline.

Run:
    python src/rag/ask.py --file data/mock/patient_A.json --query "Bệnh nhân dị ứng Penicillin, dùng Amoxicillin được không?"
    python src/rag/ask.py --query "chống chỉ định đặt nội khí quản"          # no patient
    python src/rag/ask.py --file ... --query "..." --json                    # machine-readable

Status/timing prints go to stderr so --json keeps stdout pure.
"""

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ on sys.path
from fhir.fhir_client import FHIRClient  # noqa: E402
from scoring.calculator import calculate_all  # noqa: E402
from rag.config import GEN_MODEL  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="ICU RAG clinical assistant")
    parser.add_argument("--query", required=True, help="clinical question")
    parser.add_argument("--file", help="mock FHIR Bundle JSON (patient context)")
    parser.add_argument("--patient", help="FHIR Patient ID from the live sandbox")
    parser.add_argument("--model", default=GEN_MODEL, help="generation model slug")
    parser.add_argument("--no-verify", action="store_true", help="disable the verifier")
    parser.add_argument("--backend", default=None,
                        help="verifier backend: llm | local_nli | hybrid")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    patient_context, calc = None, None
    if args.file or args.patient:
        client = (FHIRClient.from_file(args.file) if args.file
                  else FHIRClient(args.patient))
        patient_context = client.build_patient_context()
        calc = calculate_all(patient_context)

    from rag.config import VERIFIER_BACKEND
    backend = args.backend or VERIFIER_BACKEND
    _log(f"Model: {args.model} | verify: {not args.no_verify} ({backend})")
    pipeline = RAGPipeline(gen_model=args.model, verify=not args.no_verify,
                           backend=backend)
    result = pipeline.ask(args.query, patient_context, calc)

    if args.json:
        slim = {k: v for k, v in result.items() if k != "chunks"}
        print(json.dumps(slim, ensure_ascii=False, indent=2))
        return

    r = result["response"]
    _log(f"Request: {result['request_id']}")
    _log(f"Intent: {result['routing']['intent']} (via {result['routing']['via']})")
    _log(f"Verify: {r.get('verify')}")
    _log(f"Timings: {result['timings_s']}")
    print("=" * 60)
    print(r["answer"])
    if r["cited_sources"]:
        print("\nNguồn:")
        for c in r["cited_sources"]:
            print(f"  [{c['n']}] {c['source']} — {c['title'][:70]}")
    if r["fallback"]:
        _log(f"(fallback: {r['fallback_reason']})")
    print("=" * 60)


if __name__ == "__main__":
    main()
