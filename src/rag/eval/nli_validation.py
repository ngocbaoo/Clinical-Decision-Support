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


def load_nli():
    """Return an nli(premise, hypothesis) -> (label, confidence) callable.

    Prefers ONNX Runtime (optimum) to keep the project torch-free; falls back to a
    transformers/torch pipeline; raises with install guidance if neither is available.
    """
    model_id = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    try:  # ONNX path — no torch at inference time
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer
        import scipy.special as sp  # noqa: F401  (softmax)
        tok = AutoTokenizer.from_pretrained(model_id)
        model = ORTModelForSequenceClassification.from_pretrained(model_id, export=True)
        id2label = model.config.id2label

        def _nli(premise, hypothesis):
            import numpy as np
            inp = tok(premise, hypothesis, return_tensors="np", truncation=True,
                      max_length=512)
            logits = model(**inp).logits[0]
            probs = np.exp(logits) / np.exp(logits).sum()
            idx = int(probs.argmax())
            return id2label[idx].lower(), float(probs[idx])
        return _nli, "onnx"
    except Exception as onnx_err:  # noqa: BLE001
        try:
            from transformers import pipeline
            pipe = pipeline("text-classification", model=model_id, top_k=None)

            def _nli(premise, hypothesis):
                out = pipe({"text": premise, "text_pair": hypothesis})
                best = max(out, key=lambda d: d["score"])
                return best["label"].lower(), float(best["score"])
            return _nli, "torch"
        except Exception as torch_err:  # noqa: BLE001
            raise RuntimeError(
                "No local NLI runtime available.\n"
                "  ONNX (preferred): pip install optimum[onnxruntime] transformers scipy\n"
                "  or torch:         pip install transformers torch\n"
                f"  (onnx error: {onnx_err}\n   torch error: {torch_err})")


_LABEL_TO_VERDICT = {"entailment": "supported", "contradiction": "contradicted",
                     "neutral": "neutral"}


def main():
    nli, runtime = load_nli()
    print(f"NLI runtime: {runtime}\n")
    confusion = {}  # (gold, pred) -> count
    false_supported_on_contra = 0
    correct = 0
    for row in HAND_LABELED:
        label, conf = nli(row["premise"], row["claim"])
        pred = _LABEL_TO_VERDICT.get(label, "neutral")
        confusion[(row["gold"], pred)] = confusion.get((row["gold"], pred), 0) + 1
        ok = pred == row["gold"]
        correct += ok
        if row["gold"] == "contradicted" and pred == "supported":
            false_supported_on_contra += 1
        flag = "ok " if ok else "XX "
        print(f"  [{flag}] {row['id']:<3} gold={row['gold']:<12} pred={pred:<12} "
              f"conf={conf:.2f}")

    n = len(HAND_LABELED)
    acc = correct / n
    print(f"\nAccuracy: {correct}/{n} = {acc:.2f}")
    print(f"False-'supported' on contradicted rows (safety-critical): "
          f"{false_supported_on_contra}")
    print("\nConfusion (gold -> pred):")
    for gold in ("supported", "neutral", "contradicted"):
        for pred in ("supported", "neutral", "contradicted"):
            c = confusion.get((gold, pred), 0)
            if c:
                print(f"  {gold:<12} -> {pred:<12}: {c}")

    passed = false_supported_on_contra == 0 and acc >= 0.85
    print("\n" + "=" * 60)
    if passed:
        print("PASS -> set VERIFIER_BACKEND = 'local_nli' (offline, free).")
    else:
        print("FAIL -> use 'hybrid' (local NLI + escalate negation/safety/low-conf to "
              "gpt-5.4-mini) or 'llm'.")
    print("=" * 60)


if __name__ == "__main__":
    main()
