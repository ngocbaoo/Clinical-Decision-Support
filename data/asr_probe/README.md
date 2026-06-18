# Real-voice ASR probe — recording instructions

The TTS bake-off proved the matcher works on clean synthetic audio, but gTTS pronounces English
drug names unnaturally. This probe re-measures on **real human voice** — it is the gate before any
ASR decision is frozen. Audio files here are **gitignored**; only this README is committed.

## What to record
Read the curated sentences from `src/asr/eval/drug_test_cases.py` (d01–d20). You don't need all 20
— ~12 covering the core ICU drugs is enough. Then re-read 2–3 with background noise (TV/fan/ward
sounds) for a stress check.

## File naming (this is how the harness finds ground truth)
- Filename must start with the case id: **`d01.wav`, `d02.wav`, … `d12.wav`**.
  The harness maps `dNN` back to that case's text + expected drugs automatically.
- Noisy re-takes: put `noisy` anywhere in the name, e.g. **`d05_noisy.wav`** (scored as a separate
  stress variant).
- WAV preferred (mono, 16 kHz ideal but any rate works — it's resampled). mp3/m4a/ogg/flac also fine.

Example sentence (d01): *"Bệnh nhân đang dùng Vancomycin một gam, có thêm Gentamicin được không?"*

## Recording tips
- Quiet room for the main set; speak at a normal clinical pace (not slow/over-enunciated).
- Pronounce drug names the way you actually would at the bedside — that's the point.
- One sentence per file.

## Then run
```powershell
python src/asr/eval/asr_bakeoff.py --probe --initial-prompt
```
The report headline (`chunks/asr_bakeoff_report.md`) will then rest on real voice: inspect **raw vs
recoverable** drug recall per model and whether `whisper-multilingual` still wins.
