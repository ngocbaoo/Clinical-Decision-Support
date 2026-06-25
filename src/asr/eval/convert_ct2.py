"""
Convert the chosen MultiMed-ST whisper-small fine-tune to a CTranslate2 int8_float16 model.

This is the one-time build step behind the production runtime (src/asr/transcriber.py serves the
output with faster-whisper). The HF repo splits the model across two folders — weights in a
`checkpoint/` subfolder, processor (tokenizer + feature extractor) in the parent — so we stage a
merged copy, convert it, and emit the fast `tokenizer.json` faster-whisper loads.

    python src/asr/eval/convert_ct2.py            # build data/asr_ct2/whisper-multilingual-int8_float16

Idempotent: re-running overwrites the output dir. Needs the HF weights cached already (the bake-off
/ a previous run downloads them); pass --no-offline to allow a download.
"""

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/ on sys.path
from asr.config import ASR_MODEL, ASR_MODELS, ASR_REPO, DECODE  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert the whisper model to CTranslate2 int8_float16")
    ap.add_argument("--model", default=ASR_MODEL, choices=list(ASR_MODELS))
    ap.add_argument("--quantization", default=DECODE.get("compute_type", "int8_float16"))
    ap.add_argument("--no-offline", action="store_true", help="allow downloading weights if not cached")
    args = ap.parse_args()

    if not args.no_offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from huggingface_hub import snapshot_download

    spec = ASR_MODELS[args.model]
    snap = Path(snapshot_download(ASR_REPO, local_files_only=not args.no_offline))
    proc = snap / spec["processor_subfolder"]   # tokenizer + feature extractor
    ckpt = snap / spec["model_subfolder"]        # weights + config
    out = Path(spec["ct2_dir"])
    assert (ckpt / "model.safetensors").is_file(), f"weights not found at {ckpt}"

    # Stage a merged dir: parent processor files, then checkpoint weights/configs win on name clash.
    stage = Path(tempfile.mkdtemp(prefix="ct2_stage_"))
    try:
        for f in proc.iterdir():
            if f.is_file():
                shutil.copy2(f, stage / f.name)
        for f in ckpt.iterdir():
            if f.is_file():
                shutil.copy2(f, stage / f.name)

        if out.exists():
            shutil.rmtree(out)
        out.parent.mkdir(parents=True, exist_ok=True)

        from ctranslate2.converters import TransformersConverter
        copy_files = [n for n in (
            "preprocessor_config.json", "tokenizer_config.json", "vocab.json", "merges.txt",
            "special_tokens_map.json", "added_tokens.json", "normalizer.json", "generation_config.json",
        ) if (stage / n).is_file()]
        TransformersConverter(str(stage), copy_files=copy_files).convert(
            str(out), quantization=args.quantization, force=True)

        # faster-whisper wants a fast tokenizer.json; the source only ships the slow (vocab/merges)
        # form, so build it once and drop it in.
        from transformers import WhisperTokenizerFast
        WhisperTokenizerFast.from_pretrained(str(stage)).save_pretrained(str(out))
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"[convert_ct2] {args.model} -> {out}  ({args.quantization}, {size:.0f} MB)")
    assert (out / "model.bin").is_file() and (out / "tokenizer.json").is_file()


if __name__ == "__main__":
    main()
