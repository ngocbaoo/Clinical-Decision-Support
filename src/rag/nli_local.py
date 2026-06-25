"""
Local NLI verifier backend — mDeBERTa-v3 XNLI, ONNX, runs on GPU (fp16) or CPU (fp32/int8).

Replaces the per-answer gpt-5.4-mini entailment call (src/rag/verifier.classify_with_llm) with
an offline model. For each generated claim the verifier feeds:
    premise    = the cited chunk from the vector DB (or the patient-data line)
    hypothesis = the claim sentence the LLM produced
and gets back P(entailment), P(neutral), P(contradiction). Entailment -> "supported",
contradiction -> "contradicted", neutral -> "neutral" (verifier._MAP).

Precision (config.NLI_PRECISION), chosen by validation (docs/RAG_VERIFIER_LOCAL_NLI.md):
  - "fp16": near-lossless, runs on CUDA — DEFAULT. int8 dynamic quantization wrecked this model's
    Vietnamese negation (acc 0.85 -> 0.65, false-"supported" on contradictions 1 -> 3), so we use
    fp16 on the GPU instead: full accuracy, ~half the fp32 size, faster than int8-on-CPU.
  - "fp32": the plain ONNX export (CPU fallback when no GPU; same accuracy as fp16).
  - "int8": kept only for the eval harness / comparison — NOT for production (unsafe on negation).

BUILD once (`python src/rag/nli_local.py`) -> exports fp32 ONNX, casts an fp16 copy, and writes an
int8 copy, all under NLI_ONNX_DIR (gitignored). Inference uses onnxruntime only (no transformers
generation stack); on CUDA it needs torch's bundled CUDA/cuDNN DLLs, which _enable_cuda_dlls() adds
to the loader search path. Label order is read from the model config (this checkpoint is
{0: entailment, 1: neutral, 2: contradiction}) — never hardcoded.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ on sys.path
from rag.config import NLI_MODEL, NLI_ONNX_DIR, NLI_PRECISION  # noqa: E402

_LABELS = ("entailment", "neutral", "contradiction")
_FILES = {"fp16": "model_fp16.onnx", "fp32": "model.onnx", "int8": "model_quantized.onnx"}


def _enable_cuda_dlls() -> None:
    """Let onnxruntime-gpu find CUDA/cuDNN by adding torch's bundled lib dir (no-op off Windows
    or if torch is absent). Must run before the CUDA InferenceSession is created."""
    if not hasattr(sys, "getwindowsversion"):
        return
    try:
        import os
        import torch
        lib = Path(torch.__file__).parent / "lib"
        if lib.is_dir():
            os.add_dll_directory(str(lib))
    except Exception:  # noqa: BLE001 — best-effort; CPU path doesn't need it
        pass


class LocalNLI:
    """Loaded ONNX NLI model. Callable for the verifier's nli(premise, hyp) contract."""

    def __init__(self, onnx_dir: Path | str = NLI_ONNX_DIR, precision: str | None = None,
                 max_length: int = 256):
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer
        import onnxruntime as ort

        onnx_dir = Path(onnx_dir)
        want = precision or NLI_PRECISION
        have_cuda = "CUDAExecutionProvider" in ort.get_available_providers()

        # fp16 only pays off on CUDA; without a GPU fall back to the fp32 model on CPU.
        if want == "fp16" and not have_cuda:
            want = "fp32"
        fname = _FILES.get(want, _FILES["fp32"])
        if not (onnx_dir / fname).is_file():
            raise FileNotFoundError(
                f"NLI model {fname} not found at {onnx_dir}. Build it once with "
                f"`python src/rag/nli_local.py` (exports {NLI_MODEL} to fp32/fp16/int8).")

        if want == "fp16":
            _enable_cuda_dlls()
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.tokenizer = AutoTokenizer.from_pretrained(str(onnx_dir))
        self.model = ORTModelForSequenceClassification.from_pretrained(
            str(onnx_dir), file_name=fname, providers=providers)
        # On CUDA, optimum defaults to IO binding which only accepts torch tensors; we feed numpy
        # (one short pair per call), so disable it — the per-call copy cost is negligible here.
        self.model.use_io_binding = False
        self.max_length = max_length
        self.precision = want
        self.providers = list(getattr(self.model.model, "get_providers", lambda: providers)())

        id2label = {int(i): str(v).lower() for i, v in self.model.config.id2label.items()}
        self._col = {lab: i for i, lab in id2label.items()}
        missing = [lab for lab in _LABELS if lab not in self._col]
        if missing:
            raise ValueError(f"NLI model labels {id2label} missing {missing}; expected the "
                             "3-way XNLI {entailment, neutral, contradiction}.")

    def probs(self, premise: str, hypothesis: str) -> dict:
        """Return {entailment, neutral, contradiction} probabilities (sum to 1.0)."""
        inp = self.tokenizer(premise or "", hypothesis or "", return_tensors="np",
                             truncation=True, max_length=self.max_length)
        logits = np.asarray(self.model(**inp).logits)[0].astype(np.float64)
        e = np.exp(logits - logits.max())
        p = e / e.sum()
        return {lab: float(p[self._col[lab]]) for lab in _LABELS}

    def __call__(self, premise: str, hypothesis: str) -> tuple[str, float]:
        """verifier contract: -> (label, confidence) where label is the argmax of the 3 probs."""
        p = self.probs(premise, hypothesis)
        label = max(p, key=p.get)
        return label, p[label]


def build(model_id: str = NLI_MODEL, out_dir: Path | str = NLI_ONNX_DIR,
          force: bool = False) -> Path:
    """Export `model_id` to ONNX and write fp32 + fp16 + int8 copies into `out_dir` (idempotent)."""
    import shutil
    import tempfile

    from optimum.exporters.onnx import main_export
    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from transformers import AutoTokenizer

    out_dir = Path(out_dir)
    if (out_dir / _FILES["fp16"]).is_file() and not force:
        print(f"[nli_local] artifacts already at {out_dir} (use force=True to rebuild)")
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) fp32 ONNX export (build-time may use torch)
    model = ORTModelForSequenceClassification.from_pretrained(model_id, export=True)
    model.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(model_id).save_pretrained(out_dir)

    # 2) fp16 — export DIRECTLY in fp16 via the optimum exporter (device=cuda). A post-hoc
    #    onnxconverter cast leaves DeBERTa's embedding Cast node type-inconsistent, so we let the
    #    exporter emit a valid fp16 graph and copy just the model file in. This is the production
    #    precision: full accuracy, runs on CUDA.
    with tempfile.TemporaryDirectory() as tmp:
        main_export(model_id, output=tmp, task="text-classification", dtype="fp16",
                    device="cuda", local_files_only=True)
        for f in Path(tmp).glob("model.onnx*"):  # model.onnx (+ external data .onnx_data if any)
            dst = _FILES["fp16"] if f.name == "model.onnx" else f.name
            shutil.copy2(f, out_dir / dst)

    # 3) int8 dynamic quantization (CPU) — comparison only; validation showed it's unsafe here.
    #    file_name pins the fp32 source (the dir now also holds model_fp16.onnx).
    quantizer = ORTQuantizer.from_pretrained(out_dir, file_name=_FILES["fp32"])
    qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=out_dir, quantization_config=qconfig)

    sz = {p: (out_dir / f).stat().st_size / 1e6 for p, f in _FILES.items() if (out_dir / f).is_file()}
    print(f"[nli_local] {model_id}\n  id2label={model.config.id2label}\n"
          f"  sizes(MB)={ {k: round(v) for k, v in sz.items()} }  ({out_dir})")
    return out_dir


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Export NLI verifier model to ONNX (fp32/fp16/int8)")
    ap.add_argument("--force", action="store_true", help="rebuild even if artifacts exist")
    args = ap.parse_args()
    out = build(force=args.force)
    nli = LocalNLI(out)
    print(f"loaded precision={nli.precision} providers={nli.providers}")
    print("entail demo:", nli.probs("Mục tiêu MAP ≥ 65 mmHg.", "Duy trì MAP ít nhất 65 mmHg."))
    print("contra demo:", nli.probs("Không dùng Warfarin cho phụ nữ mang thai.",
                                    "Có thể dùng Warfarin cho phụ nữ mang thai."))
