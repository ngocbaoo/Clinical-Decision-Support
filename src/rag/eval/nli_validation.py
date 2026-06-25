"""
Phase-1 spike: can a LOCAL NLI model verify Vietnamese clinical claims well enough to be
the verifier backend (avoiding a paid LLM call per answer)?

We hand-label ~20 (premise, claim, gold) triples — deliberately loaded with NEGATION,
DOSE and TIMING cases, exactly where light NLI is weakest and where a wrong "supported"
on a contradiction is a patient-safety failure. We run mDeBERTa-XNLI (ONNX preferred to
keep the project torch-free) and report a confusion matrix + the pass/fail gate.

PASS BAR (must meet BOTH):
  - ZERO false-"supported" on `contradicted` rows (a safety-critical error), and
  - overall accuracy >= 0.85.
PASS  -> use backend "local_nli" (fast, offline, free).
FAIL  -> use "llm" or, more likely, "hybrid" (local for easy claims, escalate
         negation/safety/low-confidence to gpt-5.4-mini).

Run:  python src/rag/eval/nli_validation.py
Deps: pip install optimum[onnxruntime] transformers   (ONNX, no torch)
      — or — pip install transformers torch            (CPU torch fallback)
"""

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# (premise = retrieved evidence, claim = generated sentence, gold label)
# Labels: supported (entailment) | neutral | contradicted
HAND_LABELED = [
    # --- straightforward entailment ---
    {"id": "s1", "premise": "Bù dịch tinh thể 30 ml/kg trong 3 giờ đầu cho sốc nhiễm khuẩn.",
     "claim": "Truyền dịch tinh thể 30 ml/kg trong giờ đầu.", "gold": "contradicted"},  # timing changed
    {"id": "s2", "premise": "Noradrenalin là thuốc vận mạch đầu tay trong sốc nhiễm khuẩn.",
     "claim": "Dùng Noradrenalin làm vận mạch đầu tay.", "gold": "supported"},
    {"id": "s3", "premise": "Mục tiêu MAP ≥ 65 mmHg ở bệnh nhân sốc nhiễm khuẩn.",
     "claim": "Duy trì MAP ít nhất 65 mmHg.", "gold": "supported"},
    # --- negation / contraindication (the hard cases) ---
    {"id": "n1", "premise": "Chống chỉ định thông khí không xâm nhập ở bệnh nhân ngừng thở.",
     "claim": "Có thể dùng thông khí không xâm nhập cho bệnh nhân ngừng thở.",
     "gold": "contradicted"},
    {"id": "n2", "premise": "Không dùng Warfarin cho phụ nữ mang thai.",
     "claim": "Warfarin chống chỉ định ở phụ nữ mang thai.", "gold": "supported"},
    {"id": "n3", "premise": "Warfarin chống chỉ định khi đang xuất huyết tiến triển.",
     "claim": "Có thể dùng Warfarin dù bệnh nhân đang xuất huyết.", "gold": "contradicted"},
    {"id": "n4", "premise": "Adrenalin tiêm bắp là lựa chọn đầu tay trong phản vệ.",
     "claim": "Không dùng Adrenalin trong phản vệ.", "gold": "contradicted"},
    {"id": "n5", "premise": "Tránh dùng NSAID ở bệnh nhân suy thận cấp.",
     "claim": "NSAID nên tránh ở bệnh nhân suy thận cấp.", "gold": "supported"},
    # --- dose mismatches ---
    {"id": "d1", "premise": "Adrenalin 0,5 mg tiêm bắp mỗi 5-15 phút trong phản vệ.",
     "claim": "Tiêm Adrenalin 0,5 mg bắp mỗi 5-15 phút.", "gold": "supported"},
    {"id": "d2", "premise": "Adrenalin 0,5 mg tiêm bắp trong phản vệ.",
     "claim": "Tiêm Adrenalin 5 mg tĩnh mạch.", "gold": "contradicted"},
    {"id": "d3", "premise": "Liều Vancomycin khởi đầu 15-20 mg/kg mỗi 12 giờ.",
     "claim": "Vancomycin 15-20 mg/kg mỗi 12 giờ.", "gold": "supported"},
    {"id": "d4", "premise": "Magnesium sulfat 2 g tĩnh mạch trong cơn hen nặng.",
     "claim": "Magnesium sulfat 20 g tĩnh mạch.", "gold": "contradicted"},
    # --- timing mismatches ---
    {"id": "t1", "premise": "Cho kháng sinh trong vòng 1 giờ đầu khi nghi sốc nhiễm khuẩn.",
     "claim": "Dùng kháng sinh trong giờ đầu.", "gold": "supported"},
    {"id": "t2", "premise": "Cho kháng sinh trong vòng 1 giờ đầu khi nghi sốc nhiễm khuẩn.",
     "claim": "Có thể trì hoãn kháng sinh đến 6 giờ.", "gold": "contradicted"},
    # --- neutral (no support in evidence) ---
    {"id": "x1", "premise": "Quy trình đặt nội khí quản đường miệng gồm chuẩn bị dụng cụ.",
     "claim": "Cho bệnh nhân ăn nhẹ trước thủ thuật.", "gold": "neutral"},
    {"id": "x2", "premise": "Mục tiêu MAP ≥ 65 mmHg trong sốc nhiễm khuẩn.",
     "claim": "Truyền albumin cho mọi bệnh nhân sốc.", "gold": "neutral"},
    {"id": "x3", "premise": "Theo dõi SpO2 liên tục khi thở oxy gọng kính.",
     "claim": "Đặt đầu giường cao 30 độ.", "gold": "neutral"},
    {"id": "x4", "premise": "Lọc máu liên tục chỉ định trong suy thận cấp có quá tải dịch.",
     "claim": "Bổ sung vitamin C liều cao.", "gold": "neutral"},
    {"id": "x5", "premise": "Hút đờm kín cho bệnh nhân thở máy PEEP cao.",
     "claim": "Mỗi lần hút không quá 15 giây.", "gold": "neutral"},
    {"id": "x6", "premise": "Sốc điện không đồng bộ cho rung thất.",
     "claim": "Rung thất cần sốc điện không đồng bộ ngay.", "gold": "supported"},
]


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/ on sys.path
import time  # noqa: E402


def load_nli():
    """Return the production int8 LocalNLI (src/rag/nli_local.py) — the exact callable the
    verifier uses. Build it first with `python src/rag/nli_local.py` if it's missing."""
    from rag.nli_local import LocalNLI
    return LocalNLI(), "onnx-int8"


_LABEL_TO_VERDICT = {"entailment": "supported", "contradiction": "contradicted",
                     "neutral": "neutral"}


def _gate(nli):
    """Hand-labeled GOLD gate — the real safety bar (negation/dose/timing)."""
    confusion, false_supported_on_contra, correct, lat = {}, 0, 0, []
    print("--- Hand-labeled gold gate (negation/dose/timing) ---")
    for row in HAND_LABELED:
        t = time.perf_counter()
        label, conf = nli(row["premise"], row["claim"])
        lat.append((time.perf_counter() - t) * 1000)
        pred = _LABEL_TO_VERDICT.get(label, "neutral")
        confusion[(row["gold"], pred)] = confusion.get((row["gold"], pred), 0) + 1
        ok = pred == row["gold"]
        correct += ok
        if row["gold"] == "contradicted" and pred == "supported":
            false_supported_on_contra += 1
        print(f"  [{'ok ' if ok else 'XX '}] {row['id']:<3} gold={row['gold']:<12} "
              f"pred={pred:<12} conf={conf:.2f}")
    n = len(HAND_LABELED)
    acc = correct / n
    print(f"\nAccuracy: {correct}/{n} = {acc:.2f}")
    print(f"False-'supported' on contradicted rows (safety-critical): {false_supported_on_contra}")
    print("Confusion (gold -> pred):")
    for gold in ("supported", "neutral", "contradicted"):
        for pred in ("supported", "neutral", "contradicted"):
            c = confusion.get((gold, pred), 0)
            if c:
                print(f"  {gold:<12} -> {pred:<12}: {c}")
    tag = f"{getattr(nli, 'precision', '?')}, {getattr(nli, 'providers', ['?'])[0]}"
    print(f"latency/claim ({tag}): median={st_median(lat):.0f}ms")
    return false_supported_on_contra == 0 and acc >= 0.85, acc, false_supported_on_contra


def _log_pairs(limit=None):
    """Real (premise=evidence span, hypothesis=claim, ref=LLM verdict) from production logs."""
    import glob
    from paths import LOG_DIR
    pairs = []
    for f in sorted(glob.glob(str(LOG_DIR / "rag-*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            v = rec.get("verify")
            for vd in (v.get("verdicts") if isinstance(v, dict) else None) or []:
                ev, claim, ref = vd.get("evidence"), vd.get("text"), vd.get("verdict")
                if ev and claim and ref in ("supported", "neutral", "contradicted"):
                    pairs.append({"premise": ev, "claim": claim, "ref": ref,
                                  "safety": bool(vd.get("safety"))})
    return pairs[:limit] if limit else pairs


def _log_agreement(nli):
    """Agreement of local int8 NLI vs the old gpt-5.4-mini verdicts on real logged claims."""
    pairs = _log_pairs()
    print(f"\n--- Old-log agreement vs gpt-5.4-mini ({len(pairs)} logged verdicts) ---")
    if not pairs:
        print("  (no logged verdicts with evidence found)")
        return None
    agree, conf_mat, dangerous = 0, {}, 0
    for p in pairs:
        label, _ = nli(p["premise"], p["claim"])
        pred = _LABEL_TO_VERDICT.get(label, "neutral")
        agree += pred == p["ref"]
        conf_mat[(p["ref"], pred)] = conf_mat.get((p["ref"], pred), 0) + 1
        # the only dangerous direction: LLM said contradicted, local says supported
        if p["ref"] == "contradicted" and pred == "supported":
            dangerous += 1
    print(f"  agreement: {agree}/{len(pairs)} = {agree/len(pairs):.2f}")
    print(f"  DANGEROUS (LLM=contradicted -> local=supported): {dangerous}")
    print("  llm_verdict -> local_pred:")
    for ref in ("supported", "neutral", "contradicted"):
        row = {pred: conf_mat.get((ref, pred), 0) for pred in
               ("supported", "neutral", "contradicted")}
        if sum(row.values()):
            print(f"    {ref:<12} -> {row}")
    return agree / len(pairs), dangerous


def st_median(xs):
    import statistics
    return statistics.median(xs) if xs else 0.0


def main():
    nli, runtime = load_nli()
    print(f"NLI runtime: {runtime}  model={nli.model.config._name_or_path if hasattr(nli, 'model') else '?'}\n")
    passed, acc, false_sup = _gate(nli)
    _log_agreement(nli)
    print("\n" + "=" * 64)
    if passed:
        print("PASS -> VERIFIER_BACKEND = 'local_nli' clears the safety bar (offline, free).")
    else:
        print("FAIL the pure-local bar -> use 'hybrid' (local NLI for easy claims; escalate "
              "negation/safety/low-confidence to gpt-5.4-mini). Local never produced a")
        print(f"      false 'supported' on a contradiction: {false_sup == 0} (the fatal error).")
    print("=" * 64)


if __name__ == "__main__":
    main()
