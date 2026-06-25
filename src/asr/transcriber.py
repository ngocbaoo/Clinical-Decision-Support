"""
Thin wrapper around the MultiMed-ST whisper-small checkpoint, served via CTranslate2 (faster-whisper).

The weights are converted once to an int8_float16 CTranslate2 model (src/asr/eval/convert_ct2.py →
data/asr_ct2/) and loaded here with faster-whisper. This replaced the transformers runtime: it is
~2x faster on the decoder loop and uses ~half the VRAM, with recoverable drug-recall byte-equal to
the old fp16 baseline (docs/ASR_CT2_MIGRATION.md). The transformers generation stack is gone;
ctranslate2 still pulls torch for optional interop, but the offline catalog/RAG core stays torch-free.

Runs on GPU when CUDA is visible to CTranslate2, else CPU. Used by the bake-off harness and the live
push-to-talk endpoint, which share this class so production runs exactly what the experiments score.
"""

import sys
import time
from pathlib import Path

import ctranslate2
import librosa
from faster_whisper import WhisperModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ on sys.path
from asr.config import ASR_MODEL, ASR_MODELS, ASR_SAMPLE_RATE, DECODE  # noqa: E402

# faster-whisper transcribe() knobs the harness/config may override. compute_type is a *load*-time
# arg (the quantization), so it is handled separately and not forwarded to transcribe().
_GEN_KEYS = {"beam_size", "best_of", "patience", "length_penalty", "temperature",
             "no_repeat_ngram_size", "compression_ratio_threshold", "log_prob_threshold",
             "no_speech_threshold"}


class WhisperTranscriber:
    """One loaded CTranslate2 whisper-small model. `transcribe()` returns text + wall-clock latency."""

    def __init__(self, key: str = ASR_MODEL, device: str | None = None,
                 compute_type: str | None = None):
        if key not in ASR_MODELS:
            raise ValueError(f"unknown ASR model '{key}'; choices: {list(ASR_MODELS)}")
        spec = ASR_MODELS[key]
        self.key = key
        self.label = spec["label"]
        ct2_dir = Path(spec["ct2_dir"])
        if not (ct2_dir / "model.bin").is_file():
            raise FileNotFoundError(
                f"CTranslate2 model not found at {ct2_dir}. Build it once with "
                f"`python src/asr/eval/convert_ct2.py` (converts the cached HF weights to int8_float16).")

        self.device = device or ("cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu")
        # int8_float16 is GPU-only; fall back to int8 on CPU so the wrapper still loads there.
        self.compute_type = compute_type or DECODE.get("compute_type", "int8_float16")
        if self.device == "cpu" and "float16" in self.compute_type:
            self.compute_type = "int8"

        self.model = WhisperModel(str(ct2_dir), device=self.device, compute_type=self.compute_type)

    def transcribe(self, audio, sr: int, prompt: str | None = None,
                   decode: dict | None = None) -> dict:
        """audio: 1-D float waveform. `prompt` biases the decoder vocabulary toward those words
        (e.g. ICU drug names) so rare terms are likelier to be spelled out — this is faster-whisper's
        `initial_prompt`. `decode` overrides the generation params (beam_size, temperature, …); when
        omitted the experiment-chosen config.DECODE is used. Returns {text, latency_s}."""
        if sr != ASR_SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=ASR_SAMPLE_RATE)

        params = decode if decode is not None else DECODE
        gen = {k: v for k, v in params.items() if k in _GEN_KEYS}

        t0 = time.perf_counter()
        segments, _ = self.model.transcribe(
            audio, language="vi", task="transcribe", initial_prompt=prompt,
            condition_on_previous_text=False, without_timestamps=True, vad_filter=False, **gen)
        text = " ".join(s.text for s in segments).strip()  # generator runs the decode here
        latency = time.perf_counter() - t0

        return {"text": text, "latency_s": round(latency, 3)}
