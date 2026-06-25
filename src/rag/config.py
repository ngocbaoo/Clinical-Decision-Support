"""RAG module configuration — models and thresholds in one place."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ on sys.path
from paths import DATA_DIR  # noqa: E402

# Generation + routing model (OpenRouter slug). PROVISIONAL default = flash: with reasoning
# DISABLED (see GEN_REASONING_ENABLED) flash measured citation precision 0.87-0.89 (≈ plus's
# 0.90) at p50 ~8-12s vs plus ~106s. NOT a frozen decision — n is small (5-9 in-scope/run,
# 2 runs) and the real remaining blocker (answer rate 45-55%) + independent validation
# (NLI set, human-review of safety fallbacks) have NOT landed. Freeze only after those.
# plus stays reachable via --gen-model.
GEN_MODEL = "qwen/qwen3.6-flash"

# qwen3.6 is a REASONING model: by default it emits ~4000+ hidden chain-of-thought
# ("reasoning") tokens per call that are billed and dominate latency (~90% of completion
# tokens, ~22s of the ~23s generation p50) but never appear in the answer. Disabling
# reasoning cuts generation ~7x (24.5s -> 3.3s, measured 2026-06-17). Gated on the
# faithfulness eval: only keep it off if citation precision / faithfulness hold vs §8.2.
# None = provider default (reasoning on); False = disabled.
GEN_REASONING_ENABLED: bool | None = False

# Judge model for answer evaluation. Must be a DIFFERENT family from
# GEN_MODEL to avoid self-preference bias.
JUDGE_MODEL = "openai/gpt-5.4"

# --- Faithfulness verifier (claim-level entailment) ---------------------------
# Backend for the post-generation verifier:
#   "local_nli" — offline mDeBERTa-XNLI, ONNX + int8 (no torch at inference). Premise =
#                 cited chunk, hypothesis = generated claim -> P(entail/neutral/contra).
#   "llm"       — openai/gpt-5.4-mini (different family from the qwen generator)
#   "hybrid"    — local NLI for easy claims, escalate low-confidence/safety to LLM
# Evaluated 2026-06-25 (docs/RAG_VERIFIER_LOCAL_NLI.md). A local mDeBERTa-XNLI model (fp16 on GPU)
# was built to replace this paid call. Findings: int8 was REJECTED (wrecked VN-negation accuracy
# 0.85->0.65); fp16 recovered it. "hybrid" (local fp16 + LLM escalation) did NOT regress answer-rate
# (7/11 = llm) and is fail-safe, BUT faithfulness trended slightly WORSE on a small judged sample
# (4/7 vs 5/7; citation precision 0.78 vs 0.90). Since this layer guards Risk #1 and the local pass
# didn't IMPROVE the safety metric, the shipped default stays "llm" (fail-closed) until a larger
# eval justifies the switch. "hybrid"/"local_nli" remain available (the fp16 model is built) — flip
# this flag to enable them. "local_nli" is unsafe standalone; "hybrid" is the cost-saving option.
VERIFIER_BACKEND = "llm"
VERIFIER_MODEL = "openai/gpt-5.4-mini"     # used by "llm" and "hybrid" backends
VERIFY_ENABLED = True
# Below this NLI max-probability a claim is "low confidence" -> hybrid escalates.
VERIFY_NLI_CONF = 0.60

# Local NLI verifier model (src/rag/nli_local.py). Exported to ONNX (fp32/fp16/int8) under
# NLI_ONNX_DIR (gitignored); built once via `python src/rag/nli_local.py`.
#   NLI_PRECISION "fp16" = run on CUDA, near-lossless (DEFAULT). int8 dynamic quantization
#   destroyed this model's VN-negation accuracy (0.85 -> 0.65), so fp16-on-GPU is used instead;
#   falls back to fp32-on-CPU automatically when no GPU is present (docs/RAG_VERIFIER_LOCAL_NLI.md).
NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
NLI_ONNX_DIR = DATA_DIR / "nli_onnx" / "mdeberta-v3-xnli"
NLI_PRECISION = "fp16"

# Retrieval confidence threshold for the F-RAG-09 fallback ("Không đủ thông
# tin"). Calibrated 2026-06-10 by src/rag/eval/retrieval_eval.py: truly
# off-topic queries (weather/diet/vaccine) score <= 0.37, so 0.40 rejects them
# while admitting weakly-matched but legitimate clinical questions (e.g. the
# vasopressor/MAP query at 0.45). Known limitation: medical-but-absent topics
# (e.g. truyền máu, 0.601) pass the threshold and rely on the generator's
# insufficient/citation guard instead. Re-run the calibration after re-indexing.
CONF_THRESHOLD = 0.40

# How many chunks to retrieve / pass to the LLM.
TOP_K = 5

# Per-chunk character cap inside the prompt (keeps total prompt bounded).
CHUNK_CHAR_CAP = 2500

DISCLAIMER = "Cần bác sĩ xác nhận trước khi thực hiện."
