"""
ASR bake-off + gate-effectiveness: MultiMed-ST whisper-vi vs whisper-multilingual.

Each model transcribes Vietnamese drug-name audio; we score two things:
  - **raw drug-name recall** — did the drug name survive ASR verbatim? (F-ASR-03)
  - **recoverable recall** — would the post-ASR matcher (src/asr/drug_match.py) OFFER the right
    drug to the doctor in the confirm box, even when ASR garbled it? This is the real KPI: the goal
    is a confirmable suggestion, not perfect ASR.

Audio sources: curated gTTS clips (normal + slow rate) and, with --probe, real human recordings in
data/asr_probe/. The **slow** TTS variant is a known artifact and is EXCLUDED from the headline
(reported as a stress row only). --initial-prompt adds a second pass that biases the decoder toward
ICU drug names. --real N adds reference WER on real MultiMed-ST clips.

    python src/asr/eval/gen_audio.py                       # render TTS audio (once)
    python src/asr/eval/asr_bakeoff.py --initial-prompt     # TTS bake-off, base + prompt passes
    python src/asr/eval/asr_bakeoff.py --probe --initial-prompt   # add real-voice probe (the gate)

Writes chunks/asr_bakeoff_report.md (+ .json). Crash-resilient: per-sample try/except; the report
is rewritten after every (model, condition), so a mid-run failure never loses completed work.
"""

import argparse
import io
import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

import librosa

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/ on sys.path
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from asr.config import (ASR_MODELS, ASR_REPO, ASR_REPORT_FILE, ASR_REPORT_JSON,  # noqa: E402
                        ASR_TEST_DIR, TTS_VARIANTS)
from asr.drug_lexicon import PROMPT_DRUGS  # noqa: E402
from asr.drug_match import suggest_drugs  # noqa: E402
from asr.eval.drug_test_cases import DRUG_TEST_CASES  # noqa: E402
from asr.eval.gen_audio import clip_path  # noqa: E402
from asr.eval.metrics import cer, drug_hits, recoverable_hits, wer  # noqa: E402

_CASE_BY_ID = {c["id"]: c for c in DRUG_TEST_CASES}
_HEADLINE_EXCLUDE = {"slow"}  # variants kept out of the headline aggregate (TTS artifact)
_AUDIO_EXTS = ("*.wav", "*.webm", "*.mp3", "*.m4a", "*.ogg", "*.flac")


def _curated_clips() -> list[dict]:
    clips = []
    for case in DRUG_TEST_CASES:
        for vkey in TTS_VARIANTS:
            p = clip_path(case["id"], vkey)
            if p.exists() and p.stat().st_size > 0:
                clips.append({"sample_id": f"{case['id']}_{vkey}", "variant": vkey,
                              "text": case["text"], "drugs": case["drugs"], "path": str(p)})
    return clips


def _probe_clips() -> list[dict]:
    """Real recordings in data/asr_probe/. Filename starting 'dNN' maps to that curated case
    (ground-truth text+drugs); 'noisy' in the name tags it as the noise stress variant."""
    from asr.config import DATA_DIR
    probe_dir = DATA_DIR / "asr_probe"
    files = sorted(p for ext in _AUDIO_EXTS for p in probe_dir.glob(ext))
    clips = []
    for p in files:
        m = re.match(r"(d\d+)", p.stem)
        if not m or m.group(1) not in _CASE_BY_ID:
            print(f"  [probe] skip {p.name}: no matching case id (expected dNN...)")
            continue
        case = _CASE_BY_ID[m.group(1)]
        variant = "probe_noisy" if "noisy" in p.stem.lower() else "probe"
        clips.append({"sample_id": f"{p.stem}", "variant": variant,
                      "text": case["text"], "drugs": case["drugs"], "path": str(p)})
    return clips


def _real_clips(n: int) -> list[dict]:
    """N real MultiMed-ST VN clips (<=20s) — reference WER only, no drug scoring."""
    import pandas as pd
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(ASR_REPO, "vietnamese/corrected.test-00000-of-00001.parquet",
                        repo_type="dataset")
    df = pd.read_parquet(p)
    df = df[df["duration"] <= 20].head(n)
    out = []
    for i, row in enumerate(df.itertuples(index=False)):
        a = row.audio
        raw = a["bytes"] if isinstance(a, dict) else a
        y, sr = sf.read(io.BytesIO(raw), dtype="float32")
        if getattr(y, "ndim", 1) > 1:
            y = y.mean(axis=1)
        out.append({"sample_id": f"real{i:02d}", "text": row.text, "audio": y, "sr": sr})
    return out


def _agg(rows: list[dict], field: str, exclude: set | None = None) -> tuple[int, int]:
    """(hits, total) over rows for 'raw' or 'rec', optionally excluding some variants."""
    hit = tot = 0
    for r in rows:
        if "drug_hits" not in r or (exclude and r["variant"] in exclude):
            continue
        if field == "raw":
            hit += sum(h["hit"] for h in r["drug_hits"]); tot += len(r["drug_hits"])
        else:
            hit += sum(h["recoverable"] for h in r["rec_hits"]); tot += len(r["rec_hits"])
    return hit, tot


def _pct(hit: int, tot: int):
    return round(hit / tot, 4) if tot else None


def _run(key: str, samples: list[dict], prompt: str | None, condition: str,
         real: list[dict]) -> dict:
    """Transcribe all samples (optionally with a biasing prompt); score raw + recoverable."""
    from asr.transcriber import WhisperTranscriber
    print(f"\n=== {key} [{condition}] ({ASR_MODELS[key]['label']}) ===")
    tr = WhisperTranscriber(key)
    print(f"  device={tr.device} dtype={tr.dtype} prompt={'yes' if prompt else 'no'}")

    rows, latencies = [], []
    for s in samples:
        try:
            y, sr = librosa.load(s["path"], sr=None, mono=True)
            res = tr.transcribe(y, sr, prompt=prompt)
            hyp = res["text"]
            rows.append({"sample_id": s["sample_id"], "variant": s["variant"], "ref": s["text"],
                         "hyp": hyp, "wer": round(wer(s["text"], hyp), 4),
                         "cer": round(cer(s["text"], hyp), 4),
                         "drug_hits": drug_hits(hyp, s["drugs"]),
                         "rec_hits": recoverable_hits(hyp, s["drugs"]),
                         "suggestions": suggest_drugs(hyp), "latency_s": res["latency_s"]})
            latencies.append(res["latency_s"])
            raw_h, raw_t = _agg(rows, "raw"); rec_h, rec_t = _agg(rows, "rec")
            print(f"  {s['sample_id']:16} wer={rows[-1]['wer']:.2f} "
                  f"raw={raw_h}/{raw_t} recoverable={rec_h}/{rec_t}")
        except Exception as err:
            rows.append({"sample_id": s["sample_id"], "variant": s["variant"], "error": str(err)})
            print(f"  {s['sample_id']}: ERROR {err}")

    real_rows = []
    for s in real:
        try:
            res = tr.transcribe(s["audio"], s["sr"], prompt=prompt)
            real_rows.append({"sample_id": s["sample_id"], "wer": round(wer(s["text"], res["text"]), 4)})
        except Exception as err:
            real_rows.append({"sample_id": s["sample_id"], "error": str(err)})

    scored = [r for r in rows if "wer" in r]
    rw = [r["wer"] for r in real_rows if "wer" in r]
    hraw_h, hraw_t = _agg(rows, "raw", _HEADLINE_EXCLUDE)
    hrec_h, hrec_t = _agg(rows, "rec", _HEADLINE_EXCLUDE)
    return {
        "key": key, "label": ASR_MODELS[key]["label"], "condition": condition,
        "headline_raw": _pct(hraw_h, hraw_t), "headline_raw_n": [hraw_h, hraw_t],
        "headline_rec": _pct(hrec_h, hrec_t), "headline_rec_n": [hrec_h, hrec_t],
        "wer_mean": round(statistics.mean(r["wer"] for r in scored), 4) if scored else None,
        "latency_median_s": round(statistics.median(latencies), 3) if latencies else None,
        "real_wer_mean": round(statistics.mean(rw), 4) if rw else None, "real_n": len(rw),
        "rows": rows,
    }


def _pick_winner(results: list[dict]) -> dict:
    """Best (model, condition) by headline recoverable recall, then raw, then WER."""
    return sorted(results, key=lambda r: (-(r["headline_rec"] or 0), -(r["headline_raw"] or 0),
                                          r["wer_mean"] or 1e9))[0]


def _variant_breakdown(res: dict) -> dict:
    variants = sorted({r["variant"] for r in res["rows"] if "drug_hits" in r})
    out = {}
    for v in variants:
        rows = [r for r in res["rows"] if r.get("variant") == v]
        out[v] = {"raw": _agg(rows, "raw"), "rec": _agg(rows, "rec")}
    return out


def _per_drug_recoverable(res: dict) -> dict:
    agg: dict[str, list[int]] = {}
    for r in res["rows"]:
        for h in r.get("rec_hits", []):
            a = agg.setdefault(h["drug"], [0, 0]); a[1] += 1; a[0] += int(h["recoverable"])
    return agg


def _fmt(p, n):
    return f"{p*100:.1f}% ({n[0]}/{n[1]})" if p is not None else "—"


def _write_report(results: list[dict], winner: dict) -> None:
    ASR_REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    ASR_REPORT_JSON.write_text(json.dumps(
        {"generated": datetime.now().isoformat(timespec="seconds"),
         "winner": {"key": winner["key"], "condition": winner["condition"]},
         "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")

    sources = sorted({r["variant"] for res in results for r in res["rows"] if "drug_hits" in r})
    L = ["# ASR bake-off + gate effectiveness — MultiMed-ST whisper", "",
         f"Generated: {datetime.now():%Y-%m-%d %H:%M}", "",
         "Two metrics per model: **raw** drug-name recall (survived ASR verbatim) and "
         "**recoverable** recall (the post-ASR matcher would OFFER the right drug in the confirm "
         "box). Recoverable is the real KPI — the goal is a confirmable suggestion, not perfect "
         f"ASR. Sources present: {', '.join(sources)}. Headline EXCLUDES the *slow* TTS variant "
         "(an artifact). WER/latency secondary.", "",
         "## Summary (headline — slow TTS excluded)", "",
         "| Model | Condition | Raw drug recall | **Recoverable** | WER | Latency | Real WER |",
         "|-------|-----------|----------------:|----------------:|----:|--------:|---------:|"]
    for r in results:
        star = " 🏆" if r is winner else ""
        rwer = f"{r['real_wer_mean']:.3f} (n={r['real_n']})" if r.get("real_wer_mean") is not None else "—"
        wer = f"{r['wer_mean']:.3f}" if r["wer_mean"] is not None else "—"
        lat = f"{r['latency_median_s']}s" if r["latency_median_s"] is not None else "—"
        L.append(f"| {r['label']}{star} | {r['condition']} | {_fmt(r['headline_raw'], r['headline_raw_n'])} "
                 f"| **{_fmt(r['headline_rec'], r['headline_rec_n'])}** | {wer} | {lat} | {rwer} |")

    L += ["", "## Recommendation", "",
          f"**{winner['label']}** (condition: {winner['condition']}) — recoverable recall "
          f"{_fmt(winner['headline_rec'], winner['headline_rec_n'])} vs raw "
          f"{_fmt(winner['headline_raw'], winner['headline_raw_n'])}; the post-ASR matcher is what "
          "turns garbled drug names into a correct, confirmable suggestion. `ASR_MODEL` records the "
          "model pick.", "",
          "_Provisional: confirm on the real-voice probe before freezing. The matcher SUGGESTS only "
          "— the doctor confirms; it never auto-rewrites a drug name._", ""]

    # Gate effectiveness per (model, condition): raw vs recoverable by source variant.
    L += ["## Gate effectiveness — raw vs recoverable by source", "",
          "| Model | Condition | Source | Raw | Recoverable |",
          "|-------|-----------|--------|----:|------------:|"]
    for r in results:
        for v, d in _variant_breakdown(r).items():
            L.append(f"| {r['label']} | {r['condition']} | {v} | {_fmt(_pct(*d['raw']), d['raw'])} "
                     f"| {_fmt(_pct(*d['rec']), d['rec'])} |")
    L.append("")

    # Detail for the winning block only (keeps the report readable).
    L += [f"## Per-drug recoverability — {winner['label']} [{winner['condition']}]", "",
          "| Drug | Recoverable / total |", "|------|------------------:|"]
    for drug, (h, t) in sorted(_per_drug_recoverable(winner).items()):
        L.append(f"| {drug} | {h}/{t}{'' if h == t else '  ⚠️'} |")
    L += ["", f"### Sample transcripts + matcher suggestions — {winner['label']} [{winner['condition']}]",
          "", "| Sample | WER | Raw miss | Matcher offers | Transcript |",
          "|--------|----:|----------|----------------|------------|"]
    for row in winner["rows"]:
        if "error" in row:
            L.append(f"| {row['sample_id']} | ERR | — | — | {row['error']} |")
            continue
        miss = ", ".join(h["drug"] for h in row["drug_hits"] if not h["hit"]) or "—"
        offers = ", ".join(f"{s['span']}→{s['suggestion']}" for s in row["suggestions"]) or "—"
        offers = offers.replace("|", "\\|")
        hyp = row["hyp"].replace("|", "\\|")
        L.append(f"| {row['sample_id']} | {row['wer']:.2f} | {miss} | {offers} | {hyp} |")
    L.append("")

    ASR_REPORT_FILE.write_text("\n".join(L), encoding="utf-8")
    print(f"\nReport -> {ASR_REPORT_FILE}\nJSON   -> {ASR_REPORT_JSON}")


def main() -> None:
    ap = argparse.ArgumentParser(description="MultiMed-ST whisper ASR bake-off + gate effectiveness")
    ap.add_argument("--probe", action="store_true", help="use real recordings in data/asr_probe/ (the gate)")
    ap.add_argument("--tts", action="store_true", help="use the curated gTTS clips (default if neither flag)")
    ap.add_argument("--initial-prompt", action="store_true", help="add a second pass biasing toward ICU drugs")
    ap.add_argument("--real", type=int, default=0, help="also run N real MultiMed-ST clips (reference WER)")
    ap.add_argument("--models", nargs="+", default=list(ASR_MODELS), help="subset of model keys")
    args = ap.parse_args()

    samples = []
    if args.probe:
        samples += _probe_clips()
    if args.tts or not args.probe:
        samples += _curated_clips()
    if not samples:
        sys.exit("No audio. Run gen_audio.py for TTS, or record into data/asr_probe/ for --probe.")
    print(f"{len(samples)} clips; real reference: {args.real}")
    real = _real_clips(args.real) if args.real else []

    conditions = [("base", None)] + ([("prompt", PROMPT_DRUGS)] if args.initial_prompt else [])
    results = []
    for key in args.models:
        for cond, prompt in conditions:
            results.append(_run(key, samples, prompt, cond, real if cond == "base" else []))
            _write_report(results, _pick_winner(results))  # incremental safety write

    winner = _pick_winner(results)
    _write_report(results, winner)
    print(f"\n🏆 Winner: {winner['key']} [{winner['condition']}] "
          f"recoverable={_fmt(winner['headline_rec'], winner['headline_rec_n'])}")


if __name__ == "__main__":
    main()
