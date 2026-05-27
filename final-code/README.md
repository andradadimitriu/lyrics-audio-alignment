# Lyrics-to-Audio Alignment

University POLITEHNICA of Bucharest — Hasan Khadra, Dimitriu Andrada-Elena.

## Pipeline

```
song.mp3
   │
   ▼
HT Demucs (pretrained)          ── source separation
   │ vocals.wav (16 kHz mono)
   ▼
silero-vad (intro clip only)    ── per-song boundary cleanup
   │
   ▼
wav2vec 2.0 (CTC)               ── facebook/wav2vec2-base-960h  (+ LoRA r=32 for the v5 fine-tune)
                                   or XLSR-53 fine-tunes (FR/ES/DE)
   │ frame-level CTC log-probs
   ▼
torchaudio forced_align         ── Viterbi over CTC logits
   │
   ▼
word-level timestamps → MAE / MedAE / PCO 0.2/0.3/0.5/1.0/2.0
                        IntervalAcc 0.3s / 0.5s / WordAcc
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

System tools (macOS):

```bash
brew install ffmpeg deno
```

`ffmpeg` is required by Demucs and yt-dlp. `deno` is required by yt-dlp's
YouTube JavaScript-challenge solver.

## Data

Place the DALI v2 annotations at `dali/annot_tismir/` and the JamendoLyrics
dataset at `jamendolyrics/`. Neither is committed.

## Download DALI audio (two-batch workflow)

The full v4/v5 fine-tune uses two download batches (303 + 712 = 1,015 songs).

```bash
# Batch 1 — v1
cd dali_audio_downloader
python select_songs.py
DALI_COOKIES_FROM_BROWSER=chrome python download.py
cd ..
python extract_vocals_dali.py \
    --manifest dali_audio_downloader/manifest.json \
    --out-dir  dali_audio_downloader/vocals

# Batch 2 — v2 (skips IDs already in v1)
cd dali_audio_downloader_v2
python select_songs.py
DALI_COOKIES_FROM_BROWSER=chrome python download.py
python extract_vocals.py
python merge_with_v1.py   # optional: writes manifest_combined.json
cd ..
```

`DALI_COOKIES_FROM_BROWSER` may be `chrome`, `safari`, `firefox`, or `brave` —
yt-dlp reads YouTube cookies from a logged-in browser to clear the bot check.

## Build training manifest

Single batch (v1 only):

```bash
python data_prep.py
```

Combined v1 + v2 (used for v4 / v5):

```bash
python data_prep_combined.py
```

Both write `manifests/train.json` and `manifests/val.json`. Split is
deterministic (md5(dali_id), 90 / 10).

## Train

```bash
# Frozen-encoder character-head baseline (v2)
python train.py --epochs 12 --batch-size 4 --lr 5e-6

# LoRA r=16 (v3 / v4)
python train_lora.py --epochs 10 --batch-size 4 --lr 1e-4

# LoRA r=32, longer patience (v5 — the recommended fine-tune)
python train_lora.py --epochs 25 --batch-size 4 --lr 1e-4 \
    --lora-r 32 --lora-alpha 32 \
    --ckpt-prefix checkpoint_v5 --early-stop-patience 7

# Resume from an existing adapter checkpoint
python train_lora.py --resume-from checkpoints/checkpoint_v5_epoch_8.pt \
    --epochs 25 --ckpt-prefix checkpoint_v6
```

The LoRA checkpoint schema is `{"adapter_state", "base_id", "kind",
"lora_config", ...}` — only LoRA + lm_head weights are saved (~7 MB at r=32),
the base encoder is reconstructed from `base_id` at load time.

## Evaluate

```bash
python run.py                                # 3 default songs
python run.py --all                          # all 20 English JamendoLyrics songs

python run_all.py                            # English + FR/ES/DE multilingual
python run_all.py --checkpoint checkpoints/checkpoint_v5_epoch_8.pt --no-multilingual
python run_all.py --checkpoint <ckpt.pt> --songs <name> [<name> ...]   # subset

python eval_checkpoints.py                   # sweep every checkpoints/*.pt
```

`run_all.py` auto-detects the checkpoint kind: pretrained (no `--checkpoint`),
char-head fine-tune (v2 style), or LoRA adapter (v3 – v5). LoRA adapters are
merged into the base weights for inference.

## Visualize

```bash
python visualize.py --song <basename> --out-dir outputs/
```

Three-row plot (ground truth / predicted onsets / VAD intervals) for the
selected song's time window.

## Results

Headline (15-song JamendoLyrics English test set):

| Variant | Test MAE | All-20 MAE |
| --- | ---: | ---: |
| M2 baseline (Demucs + pretrained CTC) | — | 0.413 s |
| M3 pretrained (intro-only VAD) | 0.416 s | 0.391 s |
| v3 LoRA r=16, 303 songs, 10 epochs | 0.353 s | — |
| v4 LoRA r=16, 1015 songs, patience 3 | 0.373 s | 0.342 s |
| **v5 LoRA r=32, 1015 songs, patience 7** | **0.295 s** | **0.279 s** |
