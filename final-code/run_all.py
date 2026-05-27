"""End-to-end evaluation: English JamendoLyrics + FR/ES/DE multilingual.

If `checkpoints/m3_epoch*.pt` exists and is head-compatible, the latest is
loaded; otherwise the pretrained `facebook/wav2vec2-base-960h` baseline is
used. A phoneme-head checkpoint is silently skipped (head dim mismatch).

Usage:
    python run_all.py
    python run_all.py --no-multilingual         # English-only
    python run_all.py --no-vad                  # disable VAD gating
    python run_all.py --songs HILA_-_Give_Me_the_Same
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import soundfile as sf
import torch

from pipeline import (
    Wav2Vec2Aligner, get_device, load_audio, separate_vocals, load_ground_truth,
)
from evaluate import evaluate_m3, greedy_ctc_decode_chars, aggregate
from vad import detect_voice_intervals
from multilingual_eval import run_language, songs_by_language, DEFAULT_MODELS


def english_songs(jamendo_root: str) -> List[str]:
    by_lang = songs_by_language(jamendo_root)
    return by_lang.get("English", [])


def build_aligner(device: str, checkpoint_path: Optional[str]):
    """Return aligner + (is_finetuned, supports_greedy).
       Three paths:
         - no checkpoint              -> pretrained Wav2Vec2Aligner
         - char-head ckpt (new)       -> Wav2Vec2Aligner with patched lm_head
         - phoneme-head ckpt (legacy) -> PhonemeAligner (no char vocab, no greedy decode)
    """
    if not checkpoint_path:
        print(f"[run_all] no fine-tuned checkpoint; using pretrained character-head baseline")
        return Wav2Vec2Aligner(device=device), False, True
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    kind = ckpt.get("kind")
    if kind == "lora_finetune":
        from peft import LoraConfig, get_peft_model
        base_id = ckpt.get("base_id", "facebook/wav2vec2-base-960h")
        aligner = Wav2Vec2Aligner(model_id=base_id, device="cpu")
        lcfg = ckpt["lora_config"]
        cfg = LoraConfig(r=lcfg["r"], lora_alpha=lcfg["lora_alpha"],
                         target_modules=lcfg["target_modules"],
                         lora_dropout=lcfg.get("lora_dropout", 0.0),
                         bias=lcfg.get("bias", "none"))
        wrapped = get_peft_model(aligner.model, cfg)
        for p in wrapped.base_model.model.lm_head.parameters():
            p.requires_grad_(True)
        # adapter_state holds lora_* + lm_head only; everything else stays at pretrained.
        missing, unexpected = wrapped.load_state_dict(ckpt["adapter_state"], strict=False)
        # Merge LoRA into the base weights and unwrap for clean inference.
        merged = wrapped.merge_and_unload()
        aligner.model = merged.to(device).eval()
        aligner.device = device  # was "cpu" from the temporary aligner construction
        print(f"[run_all] loaded LoRA adapter from {checkpoint_path} "
              f"(epoch {ckpt.get('epoch')}, jl_val_mae={ckpt.get('jl_val_mae_mean')}, "
              f"merged into base weights)")
        return aligner, True, True
    if kind == "character_head_finetune" or "lm_head_state" in ckpt:
        base_id = ckpt.get("base_id", "facebook/wav2vec2-base-960h")
        aligner = Wav2Vec2Aligner(model_id=base_id, device=device)
        aligner.model.lm_head.load_state_dict(ckpt["lm_head_state"])
        aligner.model.eval()
        print(f"[run_all] loaded fine-tuned char head from {checkpoint_path} "
              f"(epoch {ckpt.get('epoch')}, jl_val_mae={ckpt.get('jl_val_mae_mean')})")
        return aligner, True, True
    # Legacy phoneme-head path
    from phoneme_aligner import PhonemeAligner
    print(f"[run_all] loaded legacy phoneme-head aligner from {checkpoint_path}")
    return PhonemeAligner(checkpoint_path, device=device), True, False


def run_song_m3(base: str, jamendo_root: str, out_dir: str,
                aligner, use_demucs: bool,
                use_vad: bool, demucs_device: str,
                supports_greedy_decode: bool = True) -> dict:
    mp3 = os.path.join(jamendo_root, "mp3", base + ".mp3")
    words_txt = os.path.join(jamendo_root, "lyrics", base + ".words.txt")
    gt_csv = os.path.join(jamendo_root, "annotations", "words", base + ".csv")
    vocals_wav = os.path.join(out_dir, base + "_vocals.wav")
    align_json = os.path.join(out_dir, base + "_alignment.json")
    vad_json = os.path.join(out_dir, base + "_vad.json")

    if not os.path.exists(vocals_wav):
        if use_demucs:
            separate_vocals(mp3, vocals_wav, device=demucs_device)
        else:
            wav, _ = load_audio(mp3, 16000)
            Path(vocals_wav).parent.mkdir(parents=True, exist_ok=True)
            sf.write(vocals_wav, wav.squeeze(0).cpu().numpy(), 16000)

    with open(words_txt) as f:
        words = [w.strip() for w in f if w.strip()]
    waveform, _ = load_audio(vocals_wav, 16000)

    # VAD policy: silero is reliable at detecting *start* of vocals (instrumental
    # intros are unambiguous) but unreliable at the end of sung audio (sustained
    # outros fade out below silero's speech-tuned threshold). Inspected per-song
    # bounds vs ground truth: VAD starts within 1s of first GT word, but VAD ends
    # were 60-75s short on songs with sung outros (Embers, Pure Mids).
    #
    # So: use VAD only to clip the intro. Leave the song end alone. This both
    # eliminates the M2-regression risk and still wins on songs with long
    # instrumental intros (Embers GT first word at 32.4s).
    vad_intervals = None
    vad_used = False
    boundary = None
    if use_vad:
        vad_intervals = detect_voice_intervals(waveform)
        with open(vad_json, "w") as f:
            json.dump([{"start": iv.start_sec, "end": iv.end_sec} for iv in vad_intervals], f)
        song_sec = waveform.shape[-1] / 16000
        if vad_intervals:
            pad = 0.5
            vad_start = vad_intervals[0].start_sec
            # Only clip if (a) there's a meaningful intro to clip (>3s), and
            # (b) the detected start isn't suspiciously late (>50% of song,
            #     which would indicate silero failed entirely).
            # Trigger threshold: 5s intro. Anything shorter is too small a win
            # to outweigh the tiny boundary-arithmetic shift in re-emission.
            if vad_start > 5.0 and vad_start < 0.50 * song_sec:
                boundary = (max(0.0, vad_start - pad), song_sec)
                vad_used = True

    if vad_used:
        spans = aligner.align_with_boundary_window(waveform, words, boundary[0], boundary[1])
    else:
        spans = aligner.align(waveform, words)
    with open(align_json, "w") as f:
        json.dump([asdict(s) for s in spans], f, indent=2)

    # Word-accuracy via greedy decode (character head only; phoneme-head models
    # have no '|' delimiter to split tokens into words).
    decoded_text = ""
    if supports_greedy_decode:
        try:
            log_probs = aligner._emissions(waveform)
            decoded_text = greedy_ctc_decode_chars(
                log_probs, aligner.id_to_token, aligner.pad_id, aligner.word_delim_id)
        except Exception as e:
            print(f"  [decode] failed: {e}")

    gt = load_ground_truth(gt_csv)
    m = evaluate_m3(spans, gt,
                    decoded_text=decoded_text if decoded_text else None,
                    lyric_words=words if decoded_text else None)
    row = {"name": base, "vad_applied": vad_used, **asdict(m)}
    print(f"  [{base}] MAE={m.MAE:.3f} PCO0.3={m.PCO_03:.3f} "
          f"IntAcc0.5={m.IntervalAcc_05:.3f} "
          f"WordAcc={'N/A' if m.WordAcc is None else f'{m.WordAcc:.3f}'}"
          f"  vad={'on' if vad_used else 'off'}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jamendo", default="jamendolyrics")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--multilang-out", default="outputs_multilang")
    ap.add_argument("--no-demucs", action="store_true")
    ap.add_argument("--no-vad", action="store_true")
    ap.add_argument("--no-multilingual", action="store_true")
    ap.add_argument("--songs", nargs="*", default=None,
                    help="restrict English run to these basenames (smoke)")
    ap.add_argument("--limit-multilang", type=int, default=None,
                    help="songs per language cap, for fast multilingual smoke")
    ap.add_argument("--checkpoint", default=None,
                    help="explicit checkpoint path (else auto-pick latest in checkpoints/)")
    ap.add_argument("--results-out", default="results_m3.json",
                    help="filename for results JSON in --out dir")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    device = get_device()
    demucs_device = device
    print(f"[run_all] device: {device}  use_vad={not args.no_vad}  "
          f"multilingual={not args.no_multilingual}")

    # ----- English -----
    if args.checkpoint:
        ckpt_used = args.checkpoint if os.path.exists(args.checkpoint) else None
    else:
        ckpts = sorted(glob.glob("checkpoints/checkpoint_epoch_*.pt")) \
                + sorted(glob.glob("checkpoints/m3_epoch*.pt"))
        ckpt_used = ckpts[-1] if ckpts else None
    aligner, is_finetuned, supports_greedy = build_aligner(device, ckpt_used)

    songs = args.songs or english_songs(args.jamendo)
    print(f"[run_all] English songs: {len(songs)}")
    eng_per_song = []
    t0 = time.time()
    for base in songs:
        try:
            t = time.time()
            row = run_song_m3(base, args.jamendo, args.out, aligner,
                              use_demucs=not args.no_demucs,
                              use_vad=not args.no_vad,
                              demucs_device=demucs_device,
                              supports_greedy_decode=supports_greedy)
            row["seconds"] = round(time.time() - t, 1)
            eng_per_song.append(row)
        except Exception as e:
            print(f"  FAILED {base}: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
    eng_agg = aggregate(eng_per_song)
    print(f"[run_all] English aggregate: {eng_agg}")
    print(f"[run_all] English elapsed: {time.time()-t0:.1f}s")

    # ----- Multilingual -----
    multilang = {}
    if not args.no_multilingual:
        by_lang = songs_by_language(args.jamendo)
        for lang in ("French", "Spanish", "German"):
            ml_songs = by_lang.get(lang, [])
            if args.limit_multilang:
                ml_songs = ml_songs[: args.limit_multilang]
            if not ml_songs:
                continue
            multilang[lang] = run_language(
                lang, DEFAULT_MODELS[lang], ml_songs, args.jamendo,
                os.path.join(args.multilang_out, lang.lower()),
                demucs=not args.no_demucs, device=device,
            )

    out_payload = {
        "settings": {
            "device": device,
            "demucs": not args.no_demucs,
            "vad": not args.no_vad,
            "checkpoint": ckpt_used,
        },
        "english": {
            "per_song": eng_per_song,
            "aggregate": eng_agg,
        },
        "multilingual": {
            lang: {"aggregate": r.get("aggregate", {}), "model_id": r.get("model_id"),
                   "per_song": r.get("per_song", [])}
            for lang, r in multilang.items()
        },
    }
    out_path = os.path.join(args.out, args.results_out)
    with open(out_path, "w") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\n[run_all] wrote {out_path}")


if __name__ == "__main__":
    main()
