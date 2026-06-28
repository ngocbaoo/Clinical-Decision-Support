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

# Lever 1 (src/rag/verifier.py _evidence_grounded): a claim citing chunk [n] is "grounded" only if
# >= this fraction of its quoted-evidence tokens actually appear in chunk [n] (fuzzy, diacritic/
# unit-insensitive). Ungrounded claims are dropped by code instead of trusted. 0.8 tolerates VN
# formatting noise while rejecting fabricated quotes; lower it if answer-rate collapses, raise it to
# be stricter. Tuned 2026-06-25 against answer_eval (faithfulness up vs answer-rate).
EVIDENCE_MIN_COVERAGE = 0.8

# Lever 2 (src/rag/verifier.py): for a claim whose evidence IS grounded (Lever 1), additionally
# require the claim to be ENTAILED by its own evidence span via the local mDeBERTa-XNLI model
# (premise = the ~30-token evidence quote, hypothesis = the claim). A tight premise is exactly
# where NLI is crisp, so this catches "real quote, insufficient claim" over-claims (e.g. a claim
# that quotes a liver-failure sentence then appends an unsupported H2-blocker recommendation) that
# Lever 1 (evidence exists) and the lenient LLM verifier both miss. Not-entailed -> neutral/
# contradicted -> stripped or fallback. Offline, $0, no extra inference cost beyond the NLI it
# already loads. Set False to fall back to Lever-1-only (grounded -> supported without entailment).
VERIFY_EVIDENCE_NLI = True

# Lever 2 confidence gate. As a HARD gate (drop on ANY non-entailment) the local mDeBERTa NLI
# over-rejected — answer-rate collapsed to 45% and valid paraphrases (e.g. A-11 contraindication)
# were killed, because mDeBERTa marks faithful VN paraphrases "neutral". So a claim is dropped only
# when NLI is CONFIDENT it is not entailed (max-prob >= this on a neutral/contradiction label);
# entailed OR low-confidence -> kept (benefit of the doubt). Higher = fewer false rejects but fewer
# catches; lower = more catches but more false rejects. Tuned 2026-06-25 on answer_eval.
NLI_REJECT_CONF = 0.7

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

# How many chunks to retrieve / pass to the LLM. VALIDATED 2026-06-25 (docs/RAG_EVAL_REPORT.md §4.1):
# lowered 5->4 to cut prefill (generation is ~96% of wall-clock, ~92% prefill). TOP_K=3 was tried
# first and REVERTED — it dropped scenario A-05's supporting RANK-4 chunk (a vasopressor titration
# dose) -> faithfulness 5/7->4/7. TOP_K=4 keeps the rank-4 chunk (A-05 recovered to pass/cp=1.0) and
# drops only rank-5 (~17% less prefill). Measured at K=4: faithfulness 6/8, citation-precision 0.94,
# Recall@4=1.0, safety-priority@4=1.0, answer-rate 8/11 — all >= TOP_K=5 baseline. Re-run
# topk_sweep.py + answer_eval after any re-index before changing.
TOP_K = 4

# Per-chunk character cap inside the prompt (keeps total prompt bounded). Kept at 2500 after a
# 2026-06-25 experiment (reviewer prio #2, docs/RAG_EVAL_REPORT.md §4.3): lowering it to cut prefill
# was REJECTED. cap=1800 truncated multi-step procedure chunks mid-list -> verifier integrity-break
# -> A-04 & A-08 forced to fallback (behavior regression). cap=2200 avoided that but truncated
# A-05's vasopressor-titration grounding span -> faithfulness 6/8->5/8 for a marginal prefill saving
# (median chunk ~2600 chars). Char-cap MUTILATES the chunks you keep, unlike TOP_K which drops a
# whole low-value chunk cleanly -> no safe headroom here. The clean prefill win was TOP_K 5->4.
CHUNK_CHAR_CAP = 2500

# --- Comorbidity-aware retrieval (src/rag/fusion.py) --------------------------
# Retrieval is query-driven and therefore BLIND to the patient's background pathology: a
# "sốt cao → sốc nhiễm khuẩn" query pulls the generic fluid-bolus protocol but never the
# "bù dịch thận trọng ở bệnh nhân xơ gan" caveat that IS in the corpus — so the generator gets a
# protocol it cannot reconcile against the patient's comorbidities (the pt-007 liver-failure case).
# Fix: also retrieve one auxiliary query per active comorbidity and APPEND the best comorbidity chunk
# (chosen by RRF across the per-comorbidity lists) to the primary top-K. Recall protection is
# ABSOLUTE: the primary list keeps its full TOP_K budget, comorbidity chunks are appended not
# substituted, so the primary result is identical to baseline and recall cannot drop. (A first design
# RESERVED a slot inside TOP_K and evicted the primary rank-K chunk — which silently deleted the
# answer's core for a query whose key chunk sat exactly at rank K, e.g. the sepsis protocol at rank 4
# for the pt-007 fever query. Never take from the primary budget.) Cost: comorbidity patients get up
# to TOP_K + COMORBIDITY_SLOTS chunks (bounded prefill increase, back to the old TOP_K=5 for slots=1).
# Guarded by src/rag/eval/comorbidity_retrieval_eval.py (Recall@K + Safety-priority@K hold; comorbidity
# content surfaces across organ systems).
COMORBIDITY_RETRIEVAL = True
RRF_K = 60                      # RRF rank-damping constant (standard default)
COMORBIDITY_SLOTS = 1           # max comorbidity chunks APPENDED beyond the primary TOP_K
COMORBIDITY_MIN_SCORE = 0.45    # an aux chunk must be at least this relevant to be appended (else skip → baseline)
MAX_COMORBIDITY_QUERIES = 3     # cap auxiliary queries (1 embed + 1 search each) for latency
RETRIEVE_CANDIDATES = 8         # candidates pulled per list to feed RRF
COMORBIDITY_QUERY_TEMPLATE = "xử trí và thận trọng ở bệnh nhân {cond}"

# --- Cross-encoder reranker (src/rag/reranker.py) -----------------------------
# Second-stage retrieval precision. The bi-encoder (cosine of two independent embeddings) is weak at
# precision: on a vague query the candidates cluster near the fallback threshold and an off-topic
# chunk can win rank-1 (the mislabeled antivenom adverse-reaction chunk that poisoned the pt-007
# answer). A cross-encoder scores the (query, chunk) pair JOINTLY and resolves that near-tie. We pull
# a larger bi-encoder pool (RERANK_CANDIDATES) and rerank it down to TOP_K. Torch is isolated to the
# lazy-loaded reranker module (offline path stays torch-free). Guarded by src/rag/eval/rerank_eval.py
# (Hit@1 should rise, Recall@K + Safety-priority@K must hold). bge-reranker-v2-m3 = cross-lingual,
# strong on Vietnamese; ~568M params, fp16 on the 4GB GPU. Set RERANK_ENABLED=False to bypass.
RERANK_ENABLED = True
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_CANDIDATES = 20          # bi-encoder pool size handed to the reranker (then cut to TOP_K)
RERANK_MAX_LENGTH = 512         # tokens per (query, chunk) pair
RERANK_BATCH = 16               # pairs per forward pass (bounds GPU memory on the 4GB card)

# --- Comorbidity-conflict enforcement gate (src/rag/comorbidity_gate.py) ------
# Deterministic Risk-#1 backstop: a curated rules table of (dangerous recommendation × conflicting
# comorbidity). When the FINAL answer contains a flagged recommendation AND the patient carries the
# comorbidity, a mandatory warning is prepended (answer not deleted — needed care stays). Catches the
# fluid-bolus-in-cirrhosis / paracetamol-in-liver-failure class that grounded retrieval+generation
# can't (no corpus sentence to cite). Curated conservatively; extend rules one-at-a-time.
COMORBIDITY_GATE_ENABLED = True

DISCLAIMER = "Cần bác sĩ xác nhận trước khi thực hiện."
