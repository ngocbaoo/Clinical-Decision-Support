"""RAG module configuration — models and thresholds in one place."""

# Generation + routing model (OpenRouter slug). Chosen for Vietnamese quality
# at low cost; swap here to A/B another model.
GEN_MODEL = "qwen/qwen3.6-flash"

# Judge model for answer evaluation. Must be a DIFFERENT family from
# GEN_MODEL to avoid self-preference bias.
JUDGE_MODEL = "openai/gpt-5.4"

# Retrieval confidence threshold for the F-RAG-09 fallback ("Không đủ thông
# tin"). Calibrated 2026-06-10 by src/rag/eval/retrieval_eval.py: off-topic
# queries score <= 0.37, in-scope min top-1 = 0.577 -> 0.50 splits with margin.
# Known limitation: medical-but-absent topics (e.g. truyền máu, 0.601) pass the
# threshold and rely on the generator's insufficient/citation guard instead.
# Re-run the calibration after re-indexing.
CONF_THRESHOLD = 0.50

# How many chunks to retrieve / pass to the LLM.
TOP_K = 5

# Per-chunk character cap inside the prompt (keeps total prompt bounded).
CHUNK_CHAR_CAP = 2500

DISCLAIMER = "Cần bác sĩ xác nhận trước khi thực hiện."
