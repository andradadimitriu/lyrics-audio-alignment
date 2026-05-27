"""LoRA fine-tune of the wav2vec 2.0 aligner.

LoRA adapters (r=16, alpha=32) on the transformer attention's q_proj and
v_proj. The encoder stays frozen; only the adapters and the lm_head train.
Same DALI training set, VAD policy, and 5/15 JamendoLyrics split as train.py.

Usage:
    python train_lora.py --epochs 10 --batch-size 4 --lr 1e-4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from peft import LoraConfig, get_peft_model

from pipeline import Wav2Vec2Aligner, get_device, load_audio
from train import (
    CKPT_DIR, SAMPLE_RATE, WAV2VEC2_FRAME_HOP,
    DALICharDataset, collate, ctc_loss_fn, linear_warmup,
    jl_val_test_split, preload_jl_val, jl_val_mae,
)


def adapter_state_dict(model):
    """Subset of state_dict containing LoRA params and lm_head only."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if ("lora_" in k) or ("lm_head" in k)}


def train(args):
    device = get_device()
    print(f"[train_lora] device: {device}", flush=True)

    train_manifest = json.load(open(args.manifest))
    print(f"[train_lora] {len(train_manifest)} train songs", flush=True)
    if args.smoke:
        train_manifest = train_manifest[:8]
        args.epochs = 2
        print(f"[train_lora] SMOKE: {len(train_manifest)} train, {args.epochs} epochs", flush=True)

    aligner = Wav2Vec2Aligner(model_id=args.base_id, device=device)
    processor = aligner.processor
    tokenizer = processor.tokenizer
    blank_id = tokenizer.pad_token_id

    # Build the model on CPU first so peft wrapping doesn't interact with MPS.
    raw_model = aligner.model.cpu()
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(raw_model, lora_cfg)
    # peft freezes everything except LoRA; manually unfreeze lm_head.
    for p in model.base_model.model.lm_head.parameters():
        p.requires_grad_(True)
    model.to(device)
    aligner.model = model  # so the validation aligner uses our LoRA-wrapped model

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[train_lora] trainable params: {n_train:,} / {n_total:,} "
          f"({100*n_train/n_total:.3f}%)", flush=True)

    train_ds = DALICharDataset(train_manifest, tokenizer,
                               chunk_sec=args.chunk_sec, augment=True)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate, num_workers=0)
    steps_per_epoch = len(train_dl)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_frac))
    print(f"[train_lora] {steps_per_epoch} steps/epoch x {args.epochs} epochs = "
          f"{total_steps} steps (warmup {warmup_steps})", flush=True)

    optim = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.98),
                              weight_decay=args.weight_decay)

    val_songs, test_songs = jl_val_test_split(args.jamendo, n_val=args.n_jl_val)
    print(f"[train_lora] JL val ({len(val_songs)}): {val_songs}", flush=True)
    print(f"[train_lora] JL test ({len(test_songs)}) held out", flush=True)
    val_data = preload_jl_val(val_songs, args.jamendo)
    print(f"[train_lora] preloaded {len(val_data)} JL val songs", flush=True)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CKPT_DIR / "jl_split.json", "w") as f:
        json.dump({"val": val_songs, "test": test_songs, "seed": 42}, f, indent=2)

    history = []
    autocast_dtype = (torch.bfloat16 if device == "mps"
                      else torch.float16 if device == "cuda" else None)
    global_step = 0
    best_val = float("inf")
    no_improve = 0

    # Baseline (pre-training) JL-val MAE — for a LoRA-wrapped model at init,
    # lora_B starts at zeros so the encoder behaves identically to the base.
    base_val = jl_val_mae(aligner, val_data)
    print(f"[train_lora] baseline JL-val MAE (no training): {base_val['mean']:.4f}s "
          f"per-song={[(k[:12], round(v,3)) for k,v in base_val['per_song'].items()]}",
          flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        for batch_i, batch in enumerate(train_dl):
            optim.zero_grad(set_to_none=True)
            lr_mult = linear_warmup(global_step, warmup_steps)
            for g in optim.param_groups:
                g["lr"] = args.lr * lr_mult

            wavs = batch["wavs"]
            m = wavs.mean(dim=1, keepdim=True)
            s = wavs.std(dim=1, keepdim=True).clamp_min(1e-7)
            wavs_n = ((wavs - m) / s).to(device)

            ctx = (torch.autocast(device_type=device, dtype=autocast_dtype)
                   if autocast_dtype is not None and device in ("mps", "cuda")
                   else torch.amp.autocast(device_type="cpu", enabled=False))
            with ctx:
                # Note: cannot wrap encoder in torch.no_grad() here — LoRA needs
                # gradients to flow through the encoder to the LoRA branches.
                out = model(wavs_n)  # PeftModel.forward delegates to Wav2Vec2ForCTC
                logits = out.logits
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
        print(f"[train_lora] epoch {epoch}: train_loss={train_loss:.4f}  "
              f"jl_val_mae={val['mean']:.4f}s  ({elapsed:.1f}s)", flush=True)

        ckpt_path = CKPT_DIR / f"{args.ckpt_prefix}_epoch_{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "adapter_state": adapter_state_dict(model),
            "base_id": args.base_id,
            "kind": "lora_finetune",
            "lora_config": {
                "r": args.lora_r, "lora_alpha": args.lora_alpha,
                "target_modules": ["q_proj", "v_proj"],
                "lora_dropout": 0.05, "bias": "none",
            },
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
        print(f"[train_lora] saved {ckpt_path}", flush=True)

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "jl_val_mae_mean": val["mean"],
            "jl_val_mae_per_song": val["per_song"],
            "ckpt": str(ckpt_path),
        })
        with open(CKPT_DIR / f"{args.ckpt_prefix}_history.json", "w") as f:
            json.dump({"baseline_jl_val_mae": base_val, "epochs": history}, f, indent=2)

        # Early-stop
        if val["mean"] < best_val - 1e-6:
            best_val = val["mean"]
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= args.early_stop_patience:
            print(f"[train_lora] STOPPING EARLY: no improvement in jl_val_mae for "
                  f"{no_improve} consecutive epochs (best={best_val:.4f})", flush=True)
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifests/train.json")
    ap.add_argument("--base-id", default="facebook/wav2vec2-base-960h")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-frac", type=float, default=0.10)
    ap.add_argument("--chunk-sec", type=float, default=20.0)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--jamendo", default="jamendolyrics")
    ap.add_argument("--n-jl-val", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--ckpt-prefix", default="checkpoint",
                    help="checkpoint filename prefix, e.g. 'checkpoint_v4'")
    ap.add_argument("--early-stop-patience", type=int, default=3,
                    help="stop after this many consecutive epochs without val improvement")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
