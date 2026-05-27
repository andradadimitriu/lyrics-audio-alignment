"""
Multilingual eval on JamendoLyrics FR/ES/DE splits.

We reuse Wav2Vec2Aligner with a per-language CTC checkpoint. The widely-used
XLSR-53 CTC fine-tunes are hosted by `jonatasgrosman` on the Hub (Facebook
itself only ships the pretrained-only `facebook/wav2vec2-large-xlsr-53`,
which has no CTC head). If a user-supplied `--models` mapping is passed it
takes precedence — drop in any model with a CTC head.

Usage:
    python multilingual_eval.py --jamendo jamendolyrics --out outputs_multilang
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from pipeline import (
    Wav2Vec2Aligner, get_device, load_audio, separate_vocals, load_ground_truth,
)
from evaluate import evaluate_m3, greedy_ctc_decode_chars, aggregate
import torch.nn.functional as F


DEFAULT_MODELS = {
    "French":  "jonatasgrosman/wav2vec2-large-xlsr-53-french",
    "Spanish": "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
    "German":  "jonatasgrosman/wav2vec2-large-xlsr-53-german",
}


def songs_by_language(jamendo_root: str) -> Dict[str, List[str]]:
    """Read JamendoLyrics.csv, group basenames by language."""
    out: Dict[str, List[str]] = {}
    with open(os.path.join(jamendo_root, "JamendoLyrics.csv")) as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) < 7:
                continue
            base = parts[1].replace(".mp3", "")
            lang = parts[6].strip()
            out.setdefault(lang, []).append(base)
    return out


def run_language(language: str, model_id: str, songs: List[str],
                 jamendo_root: str, out_dir: str, demucs: bool, device: str) -> dict:
    print(f"\n=== {language} :: {model_id} ===")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    try:
        aligner = Wav2Vec2Aligner(model_id=model_id, device=device)
    except Exception as e:
        print(f"  could not load model: {e}")
        return {"language": language, "model_id": model_id, "error": str(e),
                "per_song": [], "aggregate": {}}

    per_song = []
    for base in songs:
        try:
            t = time.time()
            mp3 = os.path.join(jamendo_root, "mp3", base + ".mp3")
            words_txt = os.path.join(jamendo_root, "lyrics", base + ".words.txt")
            gt_csv = os.path.join(jamendo_root, "annotations", "words", base + ".csv")
            vocals_wav = os.path.join(out_dir, base + "_vocals.wav")
            align_json = os.path.join(out_dir, base + "_alignment.json")

            if demucs and not os.path.exists(vocals_wav):
                separate_vocals(mp3, vocals_wav, device=device)
            elif not os.path.exists(vocals_wav):
                wav, _ = load_audio(mp3, 16000)
                import soundfile as sf
                sf.write(vocals_wav, wav.squeeze(0).cpu().numpy(), 16000)

            with open(words_txt) as f:
                words = [w.strip() for w in f if w.strip()]
            waveform, _ = load_audio(vocals_wav, 16000)
            spans = aligner.align(waveform, words)
            with open(align_json, "w") as f:
                json.dump([asdict(s) for s in spans], f, indent=2)

            # Greedy decode for word-accuracy
            log_probs = aligner._emissions(waveform)
            decoded = greedy_ctc_decode_chars(
                log_probs, aligner.id_to_token,
                aligner.pad_id, aligner.word_delim_id)

            gt = load_ground_truth(gt_csv)
            m = evaluate_m3(spans, gt, decoded_text=decoded, lyric_words=words)
            row = {"name": base, **asdict(m), "seconds": round(time.time()-t, 1)}
            per_song.append(row)
            print(f"  [{base}] MAE={m.MAE:.3f} PCO0.3={m.PCO_03:.3f} "
                  f"IntAcc0.5={m.IntervalAcc_05:.3f} WordAcc={m.WordAcc:.3f}")
        except Exception as e:
            print(f"  FAILED {base}: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()

    agg = aggregate(per_song)
    return {"language": language, "model_id": model_id,
            "per_song": per_song, "aggregate": agg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jamendo", default="jamendolyrics")
    ap.add_argument("--out", default="outputs_multilang")
    ap.add_argument("--no-demucs", action="store_true")
    ap.add_argument("--languages", nargs="*", default=["French", "Spanish", "German"])
    ap.add_argument("--limit", type=int, default=None,
                    help="cap songs per language for a fast smoke run")
    args = ap.parse_args()

    device = get_device()
    print(f"device: {device}")
    Path(args.out).mkdir(parents=True, exist_ok=True)

    by_lang = songs_by_language(args.jamendo)
    results = {}
    for lang in args.languages:
        songs = by_lang.get(lang, [])
        if args.limit:
            songs = songs[: args.limit]
        if not songs:
            print(f"no songs found for language={lang}, skipping")
            continue
        model_id = DEFAULT_MODELS.get(lang)
        if not model_id:
            print(f"no default model for language={lang}, skipping")
            continue
        results[lang] = run_language(lang, model_id, songs, args.jamendo,
                                     os.path.join(args.out, lang.lower()),
                                     demucs=not args.no_demucs, device=device)

    out_path = os.path.join(args.out, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")
    for lang, r in results.items():
        agg = r.get("aggregate", {})
        if agg:
            print(f"  {lang}: MAE={agg.get('MAE', float('nan')):.3f}  "
                  f"PCO0.3={agg.get('PCO_03', 0):.3f}  "
                  f"IntAcc0.5={agg.get('IntervalAcc_05', 0):.3f}  "
                  f"({agg.get('n_songs', 0)} songs)")


if __name__ == "__main__":
    main()
