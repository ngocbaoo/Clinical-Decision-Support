"""
Phase 0 — latency measurement harness (the trustworthy baseline).

Why this exists (methodology, see plan): upstream OpenRouter variance (router tail to 151s)
can exceed a lever's saving, so a lever is judged by the **isolated per-stage median over
N≥5 iterations on a fixed query set**, never an end-to-end before/after delta. And the
dominant stage (generation) must be decomposed into prefill vs decode BEFORE trimming it —
otherwise Phase 3 is guessing which knob to turn.

What it measures, per scenario × N:
  - per-stage wall-clock from RAGPipeline.ask (router, retrieval, safety, generation, verify),
  - generation token usage (prompt_tokens / completion_tokens) via ChatClient.last_usage,
  - embed cost in ISOLATION (embed_one timed on its own — splits the ~1s embed out of the
    1.1s "retrieval" line so we can finally see it as its own number).

Optional --ttft: prefill-vs-decode TIME split via the max_tokens=1 trick (prefill+TTFT) vs a
full generation on the same messages — run on a few scenarios (paid, so opt-in).

Usage:
  python src/rag/eval/latency_probe.py --n 5 --gen-model qwen/qwen3.6-flash
  python src/rag/eval/latency_probe.py --n 1 --only A-01,A-04 --ttft   # quick smoke
"""

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/ on sys.path
from paths import MOCK_DIR, ROOT  # noqa: E402
from fhir.fhir_client import FHIRClient  # noqa: E402
from scoring.calculator import calculate_all  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402
from rag.config import GEN_MODEL  # noqa: E402
from rag.context_builder import build_messages, summarize_patient  # noqa: E402
from rag.safety import format_alerts, check_allergies, check_contraindications, \
    check_drug_interactions  # noqa: E402
from rag.eval.answer_eval import SCENARIOS  # noqa: E402

BY_ID = {s["id"]: s for s in SCENARIOS}
STAGES = ("router", "retrieval", "safety", "generation", "verify", "total")


def _pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    f = int(k)
    return round(xs[f] if f + 1 >= len(xs) else xs[f] + (xs[f + 1] - xs[f]) * (k - f), 2)


def _load_ctx(sc):
    if not sc["patient"]:
        return None, None
    client = FHIRClient.from_file(str(MOCK_DIR / sc["patient"]))
    ctx = client.build_patient_context()
    return ctx, calculate_all(ctx)


def run(ids, n, gen_model, do_ttft):
    pipe = RAGPipeline(gen_model=gen_model, verify=True, backend="llm")
    rows = []          # one dict per (scenario, iter)
    embed_times = []   # isolated embed_one timings
    ttft_rows = []     # prefill/decode time split

    for sid in ids:
        sc = BY_ID[sid]
        ctx, calc = _load_ctx(sc)
        # isolated embed cost (one short query) — measured once per scenario
        t = time.perf_counter()
        try:
            pipe.embedder.embed_one(sc["query"])
            embed_times.append(round(time.perf_counter() - t, 3))
        except Exception as err:
            print(f"  [embed probe failed {sid}] {err}", file=sys.stderr)

        for i in range(n):
            try:
                res = pipe.ask(sc["query"], ctx, calc)
            except Exception as err:  # one bad scenario must not lose the whole run
                print(f"  [ask failed {sid} #{i + 1}] {err}", file=sys.stderr)
                continue
            t_s = res["timings_s"]
            row = {"id": sid, "iter": i, **{s: t_s.get(s) for s in STAGES}}
            u = pipe.chat.last_usage or {}   # last chat call on self.chat == generation
            row["prompt_tokens"] = u.get("prompt_tokens")
            row["completion_tokens"] = u.get("completion_tokens")
            rows.append(row)
            print(f"  {sid} #{i + 1}: total={row['total']}s "
                  f"[router={row['router']} retr={row['retrieval']} safety={row['safety']} "
                  f"gen={row['generation']} verify={row['verify']}] "
                  f"tok(in/out)={row['prompt_tokens']}/{row['completion_tokens']}")
        # incremental dump so a later crash can't erase finished scenarios
        (ROOT / "chunks" / "latency_baseline.json").write_text(
            json.dumps({"model": gen_model, "n": n, "rows": rows,
                        "embed_times": embed_times, "ttft": ttft_rows},
                       ensure_ascii=False, indent=2), encoding="utf-8")

        if do_ttft:
            # repeat the split so per-call queue variance is medianed out, not trusted once
            reps = [_ttft_split(pipe, sc, ctx, calc, gen_model) for _ in range(3)]
            ttft_rows.append({
                "id": sid,
                "prefill_s": round(st.median(r["prefill_s"] for r in reps), 2),
                "decode_s": round(st.median(r["decode_s"] for r in reps), 2),
                "full_s": round(st.median(r["full_s"] for r in reps), 2),
                "completion_tokens": reps[-1]["completion_tokens"],
                "raw": reps,
            })

    _report(rows, embed_times, ttft_rows, gen_model, n)


def _ttft_split(pipe, sc, ctx, calc, gen_model):
    """Prefill+TTFT (max_tokens=1) vs full generation on the SAME messages."""
    # reuse the real routing so chunks/intent match production
    routing = pipe_route(pipe, sc["query"])
    chunks = pipe.retrieve_for_intent(sc["query"], routing["intent"])
    alerts = (check_allergies(routing["drugs"], ctx or {})
              + check_contraindications(routing["drugs"], ctx or {})
              + check_drug_interactions(routing["drugs"], ctx or {}))
    summary = summarize_patient(ctx or {}, calc or {})
    messages = build_messages(sc["query"], chunks, summary, format_alerts(alerts),
                              intent=routing["intent"])
    t = time.perf_counter()
    pipe.chat.chat(messages, temperature=0.1, max_tokens=1)
    prefill = round(time.perf_counter() - t, 2)
    t = time.perf_counter()
    pipe.chat.chat(messages, temperature=0.1, max_tokens=900)
    full = round(time.perf_counter() - t, 2)
    out_tok = (pipe.chat.last_usage or {}).get("completion_tokens")
    return {"id": sc["id"], "prefill_s": prefill, "full_s": full,
            "decode_s": round(full - prefill, 2), "completion_tokens": out_tok}


def pipe_route(pipe, query):
    from rag.query_router import route
    return route(query, pipe.chat)


def _report(rows, embed_times, ttft_rows, gen_model, n):
    print("\n" + "=" * 78)
    print(f"LATENCY BASELINE  model={gen_model}  scenarios={len({r['id'] for r in rows})}  "
          f"N={n}  runs={len(rows)}")
    print("=" * 78)
    print(f"{'stage':<14}{'median':>10}{'p95':>10}{'max':>10}")
    for s in STAGES:
        xs = [r[s] for r in rows if r.get(s) is not None]
        if xs:
            print(f"{s:<14}{round(st.median(xs), 2):>10}{_pct(xs, .95):>10}{round(max(xs), 2):>10}")
    if embed_times:
        print(f"{'embed(isol.)':<14}{round(st.median(embed_times), 3):>10}"
              f"{_pct(embed_times, .95):>10}{round(max(embed_times), 3):>10}")

    pin = [r["prompt_tokens"] for r in rows if r.get("prompt_tokens")]
    pout = [r["completion_tokens"] for r in rows if r.get("completion_tokens")]
    if pin and pout:
        mi, mo = st.median(pin), st.median(pout)
        print(f"\nGeneration tokens (median): prompt(prefill)={mi:.0f}  "
              f"completion(decode)={mo:.0f}  decode-share={mo / (mi + mo):.0%}")
        gens = [r["generation"] for r in rows if r.get("generation")]
        if gens:
            print(f"Decode throughput ≈ {mo / st.median(gens):.1f} tok/s "
                  f"(median completion / median generation time)")

    if ttft_rows:
        print("\nPrefill vs decode TIME split (max_tokens=1 trick):")
        print(f"  {'id':<8}{'prefill_s':>11}{'decode_s':>10}{'full_s':>9}{'out_tok':>9}")
        for t in ttft_rows:
            print(f"  {t['id']:<8}{t['prefill_s']:>11}{t['decode_s']:>10}"
                  f"{t['full_s']:>9}{str(t['completion_tokens']):>9}")

    out = ROOT / "chunks" / "latency_baseline.json"
    out.write_text(json.dumps({"model": gen_model, "n": n, "rows": rows,
                               "embed_times": embed_times, "ttft": ttft_rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--only", default=None, help="comma-separated scenario ids")
    ap.add_argument("--gen-model", default=GEN_MODEL)
    ap.add_argument("--ttft", action="store_true", help="also measure prefill/decode split")
    args = ap.parse_args()
    ids = ([s.strip() for s in args.only.split(",")] if args.only
           else [s["id"] for s in SCENARIOS])
    run(ids, args.n, args.gen_model, args.ttft)
