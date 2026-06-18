"""
Render the curated drug-name test sentences to audio with gTTS (Google, Vietnamese).

Each case is spoken in every variant in config.TTS_VARIANTS (normal + slow speaking rate), giving
20 cases x 2 variants = 40 clips. Idempotent: skips a clip whose mp3 already exists. Run once
before the bake-off:

    python src/asr/eval/gen_audio.py

(edge-tts neural voices were the first choice but currently fail Microsoft auth; gTTS is reliable.)
"""

import sys
from pathlib import Path

from gtts import gTTS

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/ on sys.path
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from asr.config import ASR_TEST_DIR, TTS_VARIANTS  # noqa: E402
from asr.eval.drug_test_cases import DRUG_TEST_CASES  # noqa: E402


def clip_path(case_id: str, variant: str) -> Path:
    return ASR_TEST_DIR / f"{case_id}_{variant}.mp3"


def main() -> None:
    ASR_TEST_DIR.mkdir(parents=True, exist_ok=True)
    made = skipped = 0
    for case in DRUG_TEST_CASES:
        for variant, params in TTS_VARIANTS.items():
            out = clip_path(case["id"], variant)
            if out.exists() and out.stat().st_size > 0:
                skipped += 1
                continue
            gTTS(case["text"], lang="vi", slow=params["slow"]).save(str(out))
            made += 1
            print(f"  rendered {out.name}")
    print(f"\nDone: {made} rendered, {skipped} already present -> {ASR_TEST_DIR}")


if __name__ == "__main__":
    main()
