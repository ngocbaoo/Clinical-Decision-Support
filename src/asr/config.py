"""
ASR module configuration — the MultiMed-ST whisper candidates and eval paths.

This module is the ONLY place that introduces torch/transformers into the project. It is kept
isolated from the no-torch RAG/embedding core: nothing under src/rag, src/embedding, src/fhir or
src/scoring imports it, and the offline tests never load a model. See docs/RAG_REPORT.md for the
bake-off decision that fills ASR_MODEL.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ on sys.path
from paths import CHUNKS_DIR, DATA_DIR  # noqa: E402

# Whisper expects 16 kHz mono audio.
ASR_SAMPLE_RATE = 16000

# The two candidates under evaluation. Both are whisper-small fine-tunes living in subfolders of
# one HuggingFace repo; crucially the *weights* are in a checkpoint subfolder while the *processor*
# (tokenizer + feature extractor) sits at the parent folder — so they load from different paths.
ASR_REPO = "leduckhai/MultiMed-ST"
ASR_MODELS: dict[str, dict] = {
    "whisper-vi": {
        "label": "MultiMed-ST whisper-small (Vietnamese monolingual)",
        "model_subfolder": "asr/whisper-small-vietnamese/checkpoint-5000",
        "processor_subfolder": "asr/whisper-small-vietnamese",
    },
    "whisper-multilingual": {
        "label": "MultiMed-ST whisper-small (multilingual)",
        "model_subfolder": "asr/whisper-small-multilingual/checkpoint",
        "processor_subfolder": "asr/whisper-small-multilingual",
    },
}

# Chosen by the bake-off (chunks/asr_bakeoff_report.md, 2026-06-18): the multilingual model beat
# the VN monolingual on BOTH drug-name accuracy (30.6% vs 17.7%) and WER (0.280 vs 0.465) on the
# curated VN drug-name set — it handles the English drug names (code-switching) better. PROVISIONAL
# per project discipline: n is small and TTS is clean (drug accuracy is a deflated lower bound, see
# the report's caveat); confirm on real bedside audio before freezing.
ASR_MODEL: str = "whisper-multilingual"

# Eval artifacts.
ASR_TEST_DIR = DATA_DIR / "asr_test"               # generated TTS audio (gitignored)
ASR_REPORT_FILE = CHUNKS_DIR / "asr_bakeoff_report.md"
ASR_REPORT_JSON = CHUNKS_DIR / "asr_bakeoff_report.json"

# Vietnamese TTS variants for the curated drug-name test set. We use gTTS (Google) because the
# edge-tts neural voices currently fail auth (NoAudioReceived) — gTTS is reliable, no API key.
# gTTS has one VN voice, so the two "variants" differ by speaking rate (normal vs slow); this is a
# mild robustness check, not true speaker diversity (stated as a caveat in the report).
TTS_VARIANTS = {"normal": {"slow": False}, "slow": {"slow": True}}
