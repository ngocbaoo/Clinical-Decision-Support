"""RAG module configuration — models and thresholds in one place."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA_DIR  # noqa: E402

GEN_MODEL = "qwen/qwen3.6-flash"
GEN_REASONING_ENABLED: bool | None = False
JUDGE_MODEL = "openai/gpt-5.4"

VERIFIER_BACKEND = "llm"
VERIFIER_MODEL = "openai/gpt-5.4-mini"
VERIFY_ENABLED = True
VERIFY_NLI_CONF = 0.60
EVIDENCE_MIN_COVERAGE = 0.8
VERIFY_EVIDENCE_NLI = True
NLI_REJECT_CONF = 0.7

NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
NLI_ONNX_DIR = DATA_DIR / "nli_onnx" / "mdeberta-v3-xnli"
NLI_PRECISION = "fp16"

CONF_THRESHOLD = 0.40
TOP_K = 4
CHUNK_CHAR_CAP = 2500

COMORBIDITY_RETRIEVAL = True
RRF_K = 60
COMORBIDITY_SLOTS = 1
COMORBIDITY_MIN_SCORE = 0.45
MAX_COMORBIDITY_QUERIES = 3
RETRIEVE_CANDIDATES = 8
COMORBIDITY_QUERY_TEMPLATE = "xử trí và thận trọng ở bệnh nhân {cond}"

RERANK_ENABLED = True
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_CANDIDATES = 20
RERANK_MAX_LENGTH = 512
RERANK_BATCH = 16

COMORBIDITY_GATE_ENABLED = True

DISCLAIMER = "Cần bác sĩ xác nhận trước khi thực hiện."
