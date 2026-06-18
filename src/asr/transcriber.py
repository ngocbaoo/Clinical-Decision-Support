"""
Thin wrapper around a MultiMed-ST whisper-small checkpoint for Vietnamese ASR.

Loads the processor (tokenizer + feature extractor) from the parent model folder and the weights
from its checkpoint subfolder (the two live at different paths in the repo). Runs on GPU when
available (fp16), else CPU. Used by the bake-off harness; a future web endpoint can reuse it.
"""

import sys
import time
from pathlib import Path

import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ on sys.path
from asr.config import ASR_MODELS, ASR_REPO, ASR_SAMPLE_RATE  # noqa: E402


class WhisperTranscriber:
    """One loaded whisper-small checkpoint. `transcribe()` returns text + wall-clock latency."""

    def __init__(self, key: str, device: str | None = None):
        if key not in ASR_MODELS:
            raise ValueError(f"unknown ASR model '{key}'; choices: {list(ASR_MODELS)}")
        spec = ASR_MODELS[key]
        self.key = key
        self.label = spec["label"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.processor = WhisperProcessor.from_pretrained(
            ASR_REPO, subfolder=spec["processor_subfolder"])
        self.model = WhisperForConditionalGeneration.from_pretrained(
            ASR_REPO, subfolder=spec["model_subfolder"], torch_dtype=self.dtype)
        self.model.to(self.device).eval()

        # Force Vietnamese transcription (not translation) so the multilingual model doesn't
        # auto-detect a wrong language or translate to English.
        self._forced = self.processor.get_decoder_prompt_ids(language="vi", task="transcribe")

    @torch.inference_mode()
    def transcribe(self, audio, sr: int, prompt: str | None = None) -> dict:
        """audio: 1-D float waveform. `prompt` biases the decoder vocabulary toward those words
        (e.g. ICU drug names) so rare terms are likelier to be spelled out. Returns {text, latency_s}."""
        if sr != ASR_SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=ASR_SAMPLE_RATE)
        feats = self.processor(
            audio, sampling_rate=ASR_SAMPLE_RATE, return_tensors="pt"
        ).input_features.to(self.device, self.dtype)

        # Sentences are short; cap low so a biasing prompt fits the 448-token decoder budget.
        gen = {"forced_decoder_ids": self._forced, "max_new_tokens": 128}
        if prompt:
            gen["prompt_ids"] = self.processor.get_prompt_ids(
                prompt, return_tensors="pt").to(self.device)

        t0 = time.perf_counter()
        ids = self.model.generate(feats, **gen)
        latency = time.perf_counter() - t0

        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        return {"text": text, "latency_s": round(latency, 3)}
