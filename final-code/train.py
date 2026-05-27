"""Fine-tune the wav2vec 2.0 CTC head on DALI.

Encoder frozen; only the released character lm_head trains, against CTC
loss on sung audio. Validation: 5 of the 20 JamendoLyrics English songs
(deterministic split seed=42); the other 15 are held out for final test.

Usage:
    python train.py --epochs 12 --batch-size 4 --lr 5e-6
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from pipeline import (
    Wav2Vec2Aligner, get_device, load_audio, load_ground_truth, normalize_words,
)
from evaluate import evaluate_m3
from vad import detect_voice_intervals

CKPT_DIR = Path("checkpoints")
SAMPLE_RATE = 16000
WAV2VEC2_FRAME_HOP = 320
FRAME_DUR_SEC = WAV2VEC2_FRAME_HOP / SAMPLE_RATE


# ----------------------------------------------------------------------------
# Dataset: DALI vocals + character-level lyrics
# ----------------------------------------------------------------------------

class DALICharDataset(Dataset):
    """One random 20s chunk per song per __getitem__.
    Targets are character-token ids using the wav2vec2-base-960h vocab."""

    def __init__(self, manifest: List[dict], tokenizer,
                 chunk_sec: float = 20.0, augment: bool = True):
        self.entries = manifest
        self.tokenizer = tokenizer
        self.chunk_sec = chunk_sec
        self.augment = augment

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        wav, _ = load_audio(e["vocals_path"], SAMPLE_RATE)
        wav = wav.squeeze(0)
        total = wav.shape[0]
        chunk_samples = int(self.chunk_sec * SAMPLE_RATE)
        if total <= chunk_samples:
            start_s, end_s = 0, total
        else:
            max_start = total - chunk_samples
            start_s = random.randint(0, max_start) if self.augment else 0
            end_s = start_s + chunk_samples
        wav_chunk = wav[start_s:end_s]
        t0 = start_s / SAMPLE_RATE
        t1 = end_s / SAMPLE_RATE

        # Words whose interval overlaps the chunk
        kept_words = []
        for w, ws, we in zip(e["words"], e["word_starts"], e["word_ends"]):
            if we <= t0 or ws >= t1:
                continue
            kept_words.append(w)
        if not kept_words:
            kept_words = e["words"][:3]  # fallback so we never produce empty targets

        norm = [n for n in normalize_words(kept_words) if n]
        if not norm:
            norm = ["A"]  # degenerate fallback
        transcript = " ".join(norm)
        token_ids = self.tokenizer(transcript, return_tensors="pt").input_ids[0]

        return {
            "wav": wav_chunk,
            "ctc_target": token_ids.to(torch.long),
        }


def collate(batch):
    max_wav = max(b["wav"].shape[0] for b in batch)
    B = len(batch)
    wavs = torch.zeros(B, max_wav, dtype=torch.float32)
    wav_lens = torch.zeros(B, dtype=torch.long)
    ctc_targets = []
    ctc_lens = torch.zeros(B, dtype=torch.long)
    for i, b in enumerate(batch):
        T = b["wav"].shape[0]
        wavs[i, :T] = b["wav"]
        wav_lens[i] = T
        ctc_targets.append(b["ctc_target"])
        ctc_lens[i] = b["ctc_target"].shape[0]
    ctc_flat = torch.cat(ctc_targets) if ctc_targets else torch.zeros(0, dtype=torch.long)
    return {
        "wavs": wavs, "wav_lens": wav_lens,
        "ctc_flat": ctc_flat, "ctc_lens": ctc_lens,
    }


# ----------------------------------------------------------------------------
# JL val/test split
# ----------------------------------------------------------------------------

def jl_val_test_split(jamendo_root: str, n_val: int = 5, seed: int = 42):
    from multilingual_eval import songs_by_language
    songs = sorted(songs_by_language(jamendo_root).get("English", []))
    rng = random.Random(seed)
    val = sorted(rng.sample(songs, n_val))
    test = [s for s in songs if s not in val]
    return val, test


def preload_jl_val(val_songs: List[str], jamendo_root: str, out_dir: str = "outputs"):
    """Load vocals + words + GT + VAD start for each val song into memory."""
    data = []
    for base in val_songs:
        vocals_wav = os.path.join(out_dir, base + "_vocals.wav")
        if not os.path.exists(vocals_wav):
            from pipeline import separate_vocals
            mp3 = os.path.join(jamendo_root, "mp3", base + ".mp3")
            separate_vocals(mp3, vocals_wav, device="cpu")
        wav, _ = load_audio(vocals_wav, SAMPLE_RATE)
        with open(os.path.join(jamendo_root, "lyrics", base + ".words.txt")) as f:
            words = [w.strip() for w in f if w.strip()]
        gt = load_ground_truth(os.path.join(jamendo_root, "annotations", "words", base + ".csv"))
        # VAD intro clip (matches run_all.py policy)
        intervals = detect_voice_intervals(wav)
        song_sec = wav.shape[-1] / SAMPLE_RATE
        vad_start = None
        if intervals:
            vs = intervals[0].start_sec
            if vs > 5.0 and vs < 0.50 * song_sec:
                vad_start = max(0.0, vs - 0.5)
        data.append({
            "name": base, "waveform": wav, "words": words, "gt": gt,
            "vad_start": vad_start, "song_sec": song_sec,
        })
    return data


@torch.inference_mode()
def jl_val_mae(eval_aligner: Wav2Vec2Aligner, val_data) -> dict:
    """Run the (live) model through each val song; report mean MAE + per-song.
       Returns {'mean': float, 'per_song': {name: mae}}."""
    eval_aligner.model.eval()
    aes = []
    per_song = {}
    for d in val_data:
        if d["vad_start"] is not None:
            spans = eval_aligner.align_with_boundary_window(
                d["waveform"], d["words"], d["vad_start"], d["song_sec"])
        else:
            spans = eval_aligner.align(d["waveform"], d["words"])
        m = evaluate_m3(spans, d["gt"])
        per_song[d["name"]] = m.MAE
        if not math.isnan(m.MAE):
            aes.append(m.MAE)
    eval_aligner.model.train()
    return {"mean": float(np.mean(aes)) if aes else float("nan"), "per_song": per_song}


# ----------------------------------------------------------------------------
# Train loop
# ----------------------------------------------------------------------------

def ctc_loss_fn(log_probs_bft, frame_lens, ctc_flat, ctc_lens, blank):
    """MPS has no native CTC kernel (PyTorch 2.11); compute on CPU."""
    lp = log_probs_bft.transpose(0, 1).cpu()
    return F.ctc_loss(lp, ctc_flat.cpu(), frame_lens.cpu(), ctc_lens.cpu(),
                      blank=blank, zero_infinity=True, reduction="mean")


def linear_warmup(step: int, warmup_steps: int):
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, (step + 1) / warmup_steps)


def train(args):
    device = get_device()
    print(f"[train] device: {device}", flush=True)

    train_manifest = json.load(open(args.manifest))
    print(f"[train] {len(train_manifest)} train songs", flush=True)
    if args.smoke:
        train_manifest = train_manifest[:8]
        args.epochs = 2
        print(f"[train] SMOKE: {len(train_manifest)} train, {args.epochs} epochs", flush=True)

    # Wav2Vec2Aligner gives us: pretrained model (encoder + character head),
    # processor (tokenizer), and the align() / align_with_boundary_window()
    # methods we need for val MAE.
    aligner = Wav2Vec2Aligner(model_id=args.base_id, device=device)
    model = aligner.model           # Wav2Vec2ForCTC
    processor = aligner.processor
    tokenizer = processor.tokenizer
    blank_id = tokenizer.pad_token_id

    # Freeze the encoder. Only lm_head trains.
    for p in model.wav2vec2.parameters():
        p.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[train] trainable params: {n_train:,} / {n_total:,} "
          f"({100*n_train/n_total:.3f}%)", flush=True)

    train_ds = DALICharDataset(train_manifest, tokenizer,
                               chunk_sec=args.chunk_sec, augment=True)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate, num_workers=0)
    steps_per_epoch = len(train_dl)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_frac))
    print(f"[train] {steps_per_epoch} steps/epoch x {args.epochs} epochs = {total_steps} steps "
          f"(warmup {warmup_steps})", flush=True)

    optim = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.98),
                              weight_decay=args.weight_decay)

    # JL val data preload
    val_songs, test_songs = jl_val_test_split(args.jamendo, n_val=args.n_jl_val)
    print(f"[train] JL val ({len(val_songs)}): {val_songs}", flush=True)
    print(f"[train] JL test ({len(test_songs)}): {test_songs[:3]}... (held out)", flush=True)
    val_data = preload_jl_val(val_songs, args.jamendo)
    print(f"[train] preloaded {len(val_data)} JL val songs", flush=True)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    # Save split so eval_checkpoints picks up the same songs
    with open(CKPT_DIR / "jl_split.json", "w") as f:
        json.dump({"val": val_songs, "test": test_songs, "seed": 42}, f, indent=2)

    history = []
    best_val = float("inf")
    autocast_dtype = torch.bfloat16 if device == "mps" else (torch.float16 if device == "cuda" else None)
    global_step = 0

    # Baseline JL-val MAE BEFORE any training (sanity check)
    base_val = jl_val_mae(aligner, val_data)
    print(f"[train] baseline JL-val MAE (no training): {base_val['mean']:.4f}s "
          f"per-song={[(k[:12], round(v,3)) for k,v in base_val['per_song'].items()]}",
          flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        # Re-freeze encoder after model.train() (BN/dropout state, but params still frozen)
        for p in model.wav2vec2.parameters():
            p.requires_grad_(False)

        t0 = time.time()
        losses = []
        for batch_i, batch in enumerate(train_dl):
            optim.zero_grad(set_to_none=True)
            lr_mult = linear_warmup(global_step, warmup_steps)
            for g in optim.param_groups:
                g["lr"] = args.lr * lr_mult

            # Normalize per-clip (mean/std) as Wav2Vec2FeatureExtractor would
            wavs = batch["wavs"]
            m = wavs.mean(dim=1, keepdim=True)
            s = wavs.std(dim=1, keepdim=True).clamp_min(1e-7)
            wavs_n = ((wavs - m) / s).to(device)

            ctx = (torch.autocast(device_type=device, dtype=autocast_dtype)
                   if autocast_dtype is not None and device in ("mps", "cuda")
                   else torch.amp.autocast(device_type="cpu", enabled=False))
            with ctx:
                # Encoder is frozen — no need for no_grad (gradients won't be
                # computed for frozen params), but using no_grad here saves
                # activations memory. lm_head needs grad, so we run it outside.
                with torch.no_grad():
                    hidden = model.wav2vec2(wavs_n).last_hidden_state
                logits = model.lm_head(hidden)
                log_probs = F.log_softmax(logits.float(), dim=-1)
                frame_lens = (batch["wav_lens"] // WAV2VEC2_FRAME_HOP).clamp_max(logits.shape[1])
                loss = ctc_loss_fn(log_probs, frame_lens,
                                   batch["ctc_flat"], batch["ctc_lens"], blank=blank_id)

            if not torch.isfinite(loss):
                print(f"  step {global_step}: non-finite loss, skipping", flush=True)
                global_step += 1
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optim.step()
            losses.append(loss.item())
            global_step += 1
            if global_step % 50 == 0:
                print(f"  e{epoch} step {global_step:5d}/{total_steps} "
                      f"lr={args.lr*lr_mult:.2e} loss={loss.item():.4f}", flush=True)

        train_loss = float(np.mean(losses)) if losses else float("nan")
        val = jl_val_mae(aligner, val_data)
        elapsed = time.time() - t0
        print(f"[train] epoch {epoch}: train_loss={train_loss:.4f}  "
              f"jl_val_mae={val['mean']:.4f}s  ({elapsed:.1f}s)", flush=True)

        # Save head-only checkpoint
        ckpt_path = CKPT_DIR / f"checkpoint_epoch_{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "lm_head_state": {k: v.cpu() for k, v in model.lm_head.state_dict().items()},
            "base_id": args.base_id,
            "kind": "character_head_finetune",
            "train_loss": train_loss,
            "jl_val_mae_mean": val["mean"],
            "jl_val_mae_per_song": val["per_song"],
            "jl_val_songs": val_songs,
            "jl_test_songs": test_songs,
            "config": {
                "lr": args.lr, "batch_size": args.batch_size,
                "warmup_frac": args.warmup_frac, "warmup_steps": warmup_steps,
                "epochs": args.epochs, "chunk_sec": args.chunk_sec,
                "weight_decay": args.weight_decay,
            },
        }, ckpt_path)
        print(f"[train] saved {ckpt_path}", flush=True)

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "jl_val_mae_mean": val["mean"],
            "jl_val_mae_per_song": val["per_song"],
            "ckpt": str(ckpt_path),
        })
        with open(CKPT_DIR / "history.json", "w") as f:
            json.dump({
                "baseline_jl_val_mae": base_val,
                "epochs": history,
            }, f, indent=2)

        if val["mean"] < best_val:
            best_val = val["mean"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifests/train.json")
    ap.add_argument("--base-id", default="facebook/wav2vec2-base-960h")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--warmup-frac", type=float, default=0.10,
                    help="fraction of total steps for linear warmup")
    ap.add_argument("--chunk-sec", type=float, default=20.0)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--jamendo", default="jamendolyrics")
    ap.add_argument("--n-jl-val", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
