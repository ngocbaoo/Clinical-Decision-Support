"""RAG module configuration — models and thresholds in one place."""

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
#   "local_nli" — offline mDeBERTa-XNLI via ONNX (no torch; preferred if it passes
#                 the Phase-1 spike on negation/safety pairs)
#   "llm"       — openai/gpt-5.4-mini (different family from the qwen generator)
#   "hybrid"    — local NLI for easy claims, escalate low-confidence/safety to LLM
VERIFIER_BACKEND = "llm"
VERIFIER_MODEL = "openai/gpt-5.4-mini"     # used by "llm" and "hybrid" backends
VERIFY_ENABLED = True
# Below this NLI max-probability a claim is "low confidence" -> hybrid escalates.
VERIFY_NLI_CONF = 0.60

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
