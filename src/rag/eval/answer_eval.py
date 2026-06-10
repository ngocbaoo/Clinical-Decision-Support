"""
R6b — Answer quality evaluation: scenario runs + automated checks + LLM judge.

Automated (code) checks per scenario:
  citation_ok      — non-fallback answers carry >=1 valid citation (hard gate)
  behavior_ok      — outcome matches expectation (answer / fallback / refusal)
  allergy_first    — allergy alert present AND leading the response (S-01 class)
  score_consistent — clinical scores quoted in the answer match calculate_all()

Judge (openai/gpt-5.4 — different family from the qwen generator, to avoid
self-preference): claim-level faithfulness against the cited chunks + citation
precision. Every "fail" verdict must be human-confirmed before it counts in the
final report (plan §5.1).

Run:  python src/rag/eval/answer_eval.py [--no-judge] [--only A-01]
Out:  chunks/rag_eval_report.md (+ rag_answer_eval.json)
"""

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/ on sys.path
from paths import CHUNKS_DIR, MOCK_DIR  # noqa: E402
from embedding.or_client import ChatClient  # noqa: E402
from fhir.fhir_client import FHIRClient  # noqa: E402
from scoring.calculator import calculate_all  # noqa: E402
from rag.config import GEN_MODEL, JUDGE_MODEL, CONF_THRESHOLD  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402
from rag.query_router import parse_json_loose  # noqa: E402

OUT_JSON = CHUNKS_DIR / "rag_answer_eval.json"
OUT_REPORT = CHUNKS_DIR / "rag_eval_report.md"
RETRIEVAL_JSON = CHUNKS_DIR / "rag_retrieval_eval.json"

# expect: "answer" (cited answer), "fallback", "refusal" (off-topic),
#         "answer_or_fallback" (KB coverage uncertain — judged on safety, not recall)
SCENARIOS = [
    {"id": "A-01", "patient": "patient_A.json", "tier2": "S-01",
     "query": "Bệnh nhân đang dị ứng Penicillin, tôi muốn dùng Amoxicillin điều trị viêm phổi được không?",
     "expect": "answer_or_fallback", "allergy_first": True},
    {"id": "A-02", "patient": "patient_B.json", "tier2": "S-02",
     "query": "Liều kháng sinh thải qua thận cho bệnh nhân này có cần điều chỉnh không?",
     "expect": "answer_or_fallback", "must_mention": ["thận"]},
    {"id": "A-03", "patient": "patient_Q.json", "tier2": "S-03",
     "query": "NEWS2 của bệnh nhân là bao nhiêu điểm và cần làm gì tiếp theo?",
     "expect": "answer", "score_check": "news2"},
    {"id": "A-04", "patient": "patient_A.json", "tier2": "S-05",
     "query": "Bệnh nhân qSOFA dương tính, hướng xử trí sốc nhiễm khuẩn thế nào?",
     "expect": "answer"},
    {"id": "A-05", "patient": "patient_A.json", "tier2": "S-07",
     "query": "MAP của bệnh nhân đang dưới 65, điều chỉnh vận mạch thế nào?",
     "expect": "answer"},
    {"id": "A-06", "patient": "patient_C.json", "tier2": "S-06",
     "query": "Liều thuốc Zykadia cho bệnh nhân ung thư phổi giai đoạn IV?",
     "expect": "fallback"},
    {"id": "A-07", "patient": None, "tier2": "S-09",
     "query": "Kết quả trận chung kết Champions League tối qua thế nào?",
     "expect": "refusal"},
    {"id": "A-08", "patient": "patient_G.json", "tier2": "S-08",
     "query": "Bệnh nhân suy gan cấp, hướng điều trị hồi sức thế nào?",
     "expect": "answer"},
    {"id": "A-09", "patient": "patient_L.json", "tier2": "S-10",
     "query": "Đánh giá chức năng thận và điều chỉnh liều kháng sinh cho bệnh nhân này.",
     "expect": "answer_or_fallback", "must_mention": ["thiếu", "không đủ", "không có"]},
    {"id": "A-10", "patient": "patient_E.json", "tier2": None,
     "query": "Quy trình hút đờm cho bệnh nhân đang đặt nội khí quản thở máy?",
     "expect": "answer"},
    {"id": "A-11", "patient": None, "tier2": None,
     "query": "Chống chỉ định đặt nội khí quản đường miệng là gì?",
     "expect": "answer", "want_contra_citation": True},
    {"id": "A-12", "patient": "patient_O.json", "tier2": None,
     "query": "Điểm qSOFA của bệnh nhân là bao nhiêu, giải thích từng thành phần?",
     "expect": "answer", "score_check": "qsofa"},
]

JUDGE_PROMPT = """\
You are a strict clinical-AI evaluator. You receive a clinician's question, \
patient data, numbered source documents, and an assistant's answer (Vietnamese). \
The answer may only contain claims supported by the cited documents or the \
patient data / pre-computed scores.

Evaluate:
1. faithfulness: "pass" if EVERY clinical claim is supported by a cited document \
or the patient data; "fail" otherwise. List unsupported claims.
2. citation_precision: fraction (0-1) of citations [n] that genuinely support \
the statements they are attached to.
3. relevance: does the answer address the question? "yes"/"partial"/"no".

Return ONLY JSON:
{"faithfulness": "pass|fail", "unsupported_claims": ["..."], "citation_precision": 0.0, "relevance": "yes|partial|no", "notes": "..."}
"""


def _outcome(resp: dict) -> str:
    if resp["fallback"]:
        return "refusal" if resp["fallback_reason"] == "off_topic" else "fallback"
    return "answer"


def _behavior_ok(expect: str, outcome: str) -> bool:
    if expect == "answer_or_fallback":
        return outcome in ("answer", "fallback")
    return outcome == expect


def _score_consistent(answer: str, calc: dict, which: str) -> bool | None:
    """The score value quoted in the answer must equal calculate_all()'s."""
    if not calc:
        return None
    truth = calc.get(which, {}).get("total")
    if truth is None:
        return None
    pattern = {"news2": r"news\s*2\D{0,20}?(\d+)", "qsofa": r"qsofa\D{0,20}?(\d+)"}[which]
    m = re.search(pattern, answer.lower())
    if not m:
        return None  # score not quoted — counted as not-checked, not as fail
    return int(m.group(1)) == truth


def _judge(judge: ChatClient, scenario: dict, result: dict,
           patient_summary: str) -> dict:
    resp = result["response"]
    chunks = result.get("chunks_for_judge", [])
    docs = "\n\n".join(f"[{c['n']}] ({c['source']} — {c['title']})\n{c['text'][:2000]}"
                       for c in chunks)
    user = (f"QUESTION: {scenario['query']}\n\n"
            f"PATIENT DATA / PRE-COMPUTED SCORES:\n{patient_summary}\n\n"
            f"SOURCE DOCUMENTS:\n{docs or '(none cited)'}\n\n"
            f"ASSISTANT ANSWER:\n{resp['answer']}")
    reply = judge.chat(
        [{"role": "system", "content": JUDGE_PROMPT},
         {"role": "user", "content": user}],
        temperature=0.0, max_tokens=600)
    return parse_json_loose(reply)


def run(only: str | None = None, use_judge: bool = True) -> dict:
    from rag.context_builder import summarize_patient

    pipeline = RAGPipeline(gen_model=GEN_MODEL)
    judge = ChatClient(JUDGE_MODEL) if use_judge else None

    results = []
    for sc in SCENARIOS:
        if only and sc["id"] != only:
            continue
        print(f"\n--- {sc['id']}: {sc['query'][:60]}...", file=sys.stderr)
        ctx, calc = None, None
        if sc["patient"]:
            client = FHIRClient.from_file(str(MOCK_DIR / sc["patient"]))
            ctx = client.build_patient_context()
            calc = calculate_all(ctx)

        try:
            run_res = pipeline.ask(sc["query"], ctx, calc)
        except Exception as err:
            # Record the failure and keep going — partial results still count.
            print(f"    [ERROR] {err}", file=sys.stderr)
            results.append({"id": sc["id"], "tier2": sc.get("tier2"),
                            "query": sc["query"], "patient": sc["patient"],
                            "expect": sc["expect"], "outcome": "error",
                            "error": str(err), "checks": {"behavior_ok": False},
                            "judge": None, "timings_s": {"total": None},
                            "answer": "", "citations": [], "cited_sources": [],
                            "fallback_reason": None, "intent": None})
            continue
        resp = run_res["response"]
        outcome = _outcome(resp)

        checks = {
            "behavior_ok": _behavior_ok(sc["expect"], outcome),
            "citation_ok": bool(resp["citations"]) if outcome == "answer" else True,
        }
        if sc.get("allergy_first"):
            ans = resp["answer"]
            checks["allergy_first"] = bool(resp["alerts"]) and \
                ans.lstrip().startswith("⚠️")
        if sc.get("score_check"):
            checks["score_consistent"] = _score_consistent(
                resp["answer"], calc or {}, sc["score_check"])
        if sc.get("must_mention") and outcome == "answer":
            low = resp["answer"].lower()
            checks["mentions_required"] = any(k in low for k in sc["must_mention"])

        # word count of the recommendation body (alert block excluded)
        body = resp["answer"].split("\n\n")[-1] if resp["alerts"] else resp["answer"]
        checks["concise"] = len(body.split()) <= 160

        # capture cited chunk texts for the judge (chunks come back from the run)
        cited = []
        if outcome == "answer" and resp["citations"]:
            chunks = run_res.get("chunks", [])
            for c_n in resp["citations"]:
                if 1 <= c_n <= len(chunks):
                    c = chunks[c_n - 1]
                    cited.append({"n": c_n, "source": c["source"], "title": c["title"],
                                  "text": c["text"], "chunk_type": c["chunk_type"]})
        run_res["chunks_for_judge"] = cited
        if sc.get("want_contra_citation") and outcome == "answer":
            checks["contra_cited"] = any(
                c["chunk_type"] == "contraindication" for c in cited)

        verdict = None
        if use_judge and outcome == "answer":
            try:
                verdict = _judge(judge, sc, run_res,
                                 summarize_patient(ctx or {}, calc or {}))
            except Exception as err:
                verdict = {"error": str(err)}

        results.append({
            "id": sc["id"], "tier2": sc.get("tier2"), "query": sc["query"],
            "patient": sc["patient"], "expect": sc["expect"], "outcome": outcome,
            "fallback_reason": resp["fallback_reason"],
            "intent": run_res["routing"]["intent"],
            "checks": checks, "judge": verdict,
            "timings_s": run_res["timings_s"],
            "answer": resp["answer"],
            "citations": resp["citations"],
            "cited_sources": resp["cited_sources"],
        })
        flag = "OK " if all(v for v in checks.values() if v is not None) else "FAIL"
        reason = f" reason={resp['fallback_reason']}" if resp["fallback_reason"] else ""
        print(f"    [{flag}] outcome={outcome}{reason} checks={checks}",
              file=sys.stderr)
    return {"results": results}


def write_report(data: dict) -> None:
    rows = data["results"]
    n = len(rows)
    answers = [r for r in rows if r["outcome"] == "answer"]
    auto_pass = [r for r in rows
                 if all(v for v in r["checks"].values() if v is not None)]
    judged = [r for r in answers if r.get("judge") and "faithfulness" in (r["judge"] or {})]
    faithful = [r for r in judged if r["judge"]["faithfulness"] == "pass"]
    cit_prec = [r["judge"].get("citation_precision") for r in judged
                if isinstance(r["judge"].get("citation_precision"), (int, float))]
    latencies = sorted(r["timings_s"]["total"] for r in rows
                       if r["timings_s"].get("total") is not None)
    p50 = latencies[len(latencies) // 2] if latencies else None

    lines = [
        "# RAG Module — Evaluation Report",
        f"**Ngày:** {date.today()} · **Generation:** `{GEN_MODEL}` · "
        f"**Judge:** `{JUDGE_MODEL}` · **Threshold:** {CONF_THRESHOLD}",
        "",
        "## 1. Retrieval quality (gold set)",
    ]
    if RETRIEVAL_JSON.exists():
        rs = json.loads(RETRIEVAL_JSON.read_text(encoding="utf-8"))["summary"]
        o = rs["overall"]
        lines += [
            "| Metric | Value | Target |",
            "|--------|-------|--------|",
            f"| Hit@1 (in-scope, n={o['n']}) | {o['hit@1']} | ≥ 0.70 |",
            f"| Recall@5 | {o['recall@5']} | ≥ 0.85 |",
            f"| MRR@5 | {o['mrr@5']} | ≥ 0.75 |",
            f"| Safety-priority rate (contra in top-3) | {rs['safety_priority_rate']} | 1.00 |",
            f"| Legacy keyword detection (safety queries) | {rs['keyword_detection_rate']} | (report) |",
            f"| OOS max top-1 / in-scope min top-1 | {rs['oos']['max_top1']:.3f} / "
            f"{rs['in_scope_min_top1']:.3f} | separation |",
            f"| Suggested CONF_THRESHOLD | {rs['suggested_threshold']} "
            f"(clean: {rs['threshold_separates']}) | — |",
            "",
            "Per group:",
            "",
            "| Group | n | Hit@1 | Recall@5 | MRR@5 |",
            "|-------|---|-------|----------|-------|",
        ]
        for g, m in rs["by_group"].items():
            if m:
                lines.append(f"| {g} | {m['n']} | {m['hit@1']} | {m['recall@5']} | {m['mrr@5']} |")
    else:
        lines.append("_Chưa chạy retrieval_eval.py — không có dữ liệu._")

    lines += [
        "",
        "## 2. Answer quality (scenario set)",
        f"- Scenario chạy: **{n}** | pass automated checks: **{len(auto_pass)}/{n}**",
        f"- Citation rate (non-fallback): "
        f"**{sum(1 for r in answers if r['checks']['citation_ok'])}/{len(answers)}**",
        f"- Faithfulness (judge): **{len(faithful)}/{len(judged)} pass**"
        + (" — mọi verdict fail cần người xác nhận" if len(faithful) < len(judged) else ""),
        f"- Citation precision (judge, mean): "
        f"**{round(sum(cit_prec) / len(cit_prec), 2) if cit_prec else 'n/a'}**",
        f"- Latency p50 (RAG-only): **{p50}s** (target < 4.5s)",
        "",
        "| ID | Tier-2 | Intent | Outcome | Checks | Judge |",
        "|----|--------|--------|---------|--------|-------|",
    ]
    for r in rows:
        checks = ", ".join(f"{k}={'?' if v is None else ('Y' if v else 'N')}"
                           for k, v in r["checks"].items())
        j = r.get("judge") or {}
        jtxt = j.get("faithfulness", "—")
        if j.get("citation_precision") is not None:
            jtxt += f" / cp={j['citation_precision']}"
        lines.append(f"| {r['id']} | {r.get('tier2') or '—'} | {r['intent']} | "
                     f"{r['outcome']} | {checks} | {jtxt} |")

    fails = [r for r in rows
             if not all(v for v in r["checks"].values() if v is not None)
             or (r.get("judge") or {}).get("faithfulness") == "fail"]
    lines += ["", "## 3. Failure analysis", ""]
    if not fails:
        lines.append("Không có failure case.")
    for i, r in enumerate(fails, 1):
        j = r.get("judge") or {}
        lines += [
            f"### Failure Case #{i}",
            f"**Scenario:** {r['id']} ({r.get('tier2') or 'extra'})",
            f"**Input:** {r['query']}",
            f"**Outcome:** {r['outcome']} (expect: {r['expect']}) | "
            f"checks: {r['checks']}",
            f"**Unsupported claims (judge):** {j.get('unsupported_claims', [])}",
            f"**Root cause:** [ retrieval / generation / safety-gate / calculation / FHIR data ]",
            f"**Severity:** [ Critical / High / Medium / Low ]",
            "",
        ]
    lines += [
        "",
        "> Human rubric (Tier 2, 5 tiêu chí × 0–2 điểm) chấm riêng theo "
        "Evaluation Plan §3.2; judge verdicts chỉ là pre-screen.",
    ]
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport: {OUT_REPORT}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--only", help="run a single scenario id (e.g. A-01)")
    args = parser.parse_args()

    data = run(only=args.only, use_judge=not args.no_judge)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"Saved: {OUT_JSON}")
    if not args.only:
        write_report(data)


if __name__ == "__main__":
    main()
