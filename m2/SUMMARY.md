# Lyrics-to-Audio Alignment — Milestone 2 Summary

**Authors:** Hasan Khadra, Dimitriu Andrada-Elena
**Course:** University POLITEHNICA of Bucharest
**Date:** 2026-04-29

## Architecture

```
song.mp3 → HT Demucs (vocals stem) → wav2vec 2.0 base 960h (CTC logits)
        → torchaudio.functional.forced_align → word-level (start, end)
```

- **SSL component**: `facebook/wav2vec2-base-960h`. The base model was self-supervised on 960h of unlabeled LibriSpeech via contrastive predictive coding; the released checkpoint is then CTC fine-tuned on the labeled subset. We use it off-the-shelf as a frozen acoustic model — transfer learning from speech to singing.
- **CTC component**: the model's per-frame token log-probabilities are fed to `torchaudio.functional.forced_align`, a Viterbi pass over the CTC lattice given the lyrics as the target sequence.
- **Source separation**: HT Demucs (`htdemucs`) extracts the vocals stem before alignment so the CTC model sees vocals only — this proved essential.

Alternative we did **not** pick: a contrastive learning approach (e.g. dual-encoder lyrics vs. audio embeddings with InfoNCE). Discarded for M2 because forced-alignment from a pretrained CTC head is a more direct signal for *frame-accurate* timestamps and required no training.

## What was built

| File | Purpose |
| --- | --- |
| `pipeline.py` | `Wav2Vec2Aligner`, HT Demucs wrapper, CTC + forced-align, evaluation metrics. |
| `run.py` | CLI: `--all`, `--no-demucs`, `--songs`, `--demucs-device`. |
| `transcribe_demo.py` | Greedy CTC decode of vocals stem — visible artifact that the SSL+CTC backbone is doing real work. |
| `visualize.py` | Two-row timeline: predicted onsets vs. ground truth, colored by error. |

## Aggregate results — 20 English JamendoLyrics songs

| Variant | MAE (s) | MedAE (s) | PCO 0.3 | PCO 0.2 | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: |
| **wav2vec 2.0 + Demucs** (mps) | **0.413** | **0.064** | **91.6 %** | **87.0 %** | 211 s |
| wav2vec 2.0 raw mix (mps) | 6.175 | 3.524 | 60.2 % | 56.8 % | 30 s |

Source separation is decisive: MAE drops 15× and PCO 0.3 jumps 31 percentage points. All 20 songs aligned end-to-end (no hard failures); the median-error of 64 ms with Demucs means most onsets are well within a syllable.

## Per-song results (with Demucs)

| Song | n words | MAE (s) | MedAE (s) | PCO 0.3 | PCO 0.2 | sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Wordsmith — The Statement | 581 | 0.050 | 0.043 | 99.8 % | 99.5 % | 7.6 |
| JASON MILLER — CROWD PLEASER | 521 | 0.052 | 0.041 | 99.0 % | 98.3 % | 8.7 |
| Songwriterz — Back In Time | 238 | 0.065 | 0.037 | 98.3 % | 92.9 % | 10.7 |
| LUNABLIND — Vision (Radio Edit) | 288 | 0.077 | 0.047 | 98.6 % | 97.6 % | 10.7 |
| Slingshot Miracle — Whistler | 162 | 0.082 | 0.052 | 98.1 % | 90.7 % | 10.6 |
| Cortez — Feel (Stripped) | 355 | 0.091 | 0.068 | 97.5 % | 91.0 % | 13.3 |
| Moon I Mean — Wrong Concept | 272 | 0.095 | 0.048 | 97.4 % | 93.0 % | 10.3 |
| Rxbyn — Bad Side | 440 | 0.108 | 0.058 | 94.1 % | 90.5 % | 10.5 |
| HILA — Give Me the Same | 322 | 0.110 | 0.056 | 92.9 % | 88.8 % | 14.6 |
| Lower Loveday — Is It Right | 212 | 0.154 | 0.064 | 96.2 % | 92.9 % | 8.6 |
| Quentin Hannappe — Keep On | 175 | 0.162 | 0.036 | 97.1 % | 96.6 % | 13.0 |
| The Rinn — Voices (2017) | 203 | 0.164 | 0.044 | 90.1 % | 86.7 % | 10.8 |
| Color Out — Falling Star | 224 | 0.209 | 0.148 | 91.1 % | 72.8 % | 9.8 |
| The.madpix.project — One Way Street | 183 | 0.228 | 0.039 | 98.4 % | 92.9 % | 7.6 |
| Explosive Ear Candy — Like The Sun | 265 | 0.423 | 0.047 | 97.0 % | 95.1 % | 9.2 |
| Ridgway — Fire Inside | 304 | 0.605 | 0.054 | 88.2 % | 83.2 % | 13.1 |
| Kinematic — Peyote | 147 | 0.625 | 0.051 | 89.1 % | 86.4 % | 8.2 |
| Avercage — Embers | 189 | 0.818 | 0.159 | 67.2 % | 58.7 % | 10.7 |
| Tom Orlando — The One (feat. Tina G) | 498 | 1.294 | 0.046 | 82.1 % | 80.7 % | 11.5 |
| Pure Mids — The Leader | 114 | 2.846 | 0.142 | 59.6 % | 52.6 % | 11.6 |

## Sample artifacts

- `outputs/sample_alignment_with_demucs.png` — predicted-vs-GT timeline for HILA, 20–50 s window.
- `outputs/sample_alignment_raw.png` — same window, no Demucs (visibly worse).
- `outputs/ctc_transcription.txt` — first 30 s of the vocals stem decoded by greedy CTC. Output is recognisably lyrics-like ("DE YOU WAKEN... WONDERT HOW CAN..."), demonstrating that wav2vec 2.0 transfers from speech to singing acoustically — just imperfectly enough that we still need the lyrics as the alignment target.

## Failures and known limitations

- **Avercage — Embers** still has the largest tail error (MAE 0.82 s) even with Demucs. The song has a 30 s instrumental intro; Demucs reduces the music bleed but a few residual percussive ticks at ~25 s push some early forced-align frames to the wrong word. The median (159 ms) is fine, the mean is what suffers.
- **Pure Mids — The Leader** (MAE 2.85 s) and **Tom Orlando — The One** (MAE 1.29 s) — both are heavily processed productions where Demucs leaves audible music bleed; the CTC head latches onto false phonemes between sung lines. These are the dominant outliers in the with-Demucs aggregate.
- **No fine-tuning** of wav2vec 2.0 on singing yet. The model is a frozen speech ASR head used for transfer. M3 plan: CTC + masked frame-wise CE auxiliary loss (Cheng et al.) using JamendoLyrics word-level annotations as supervision.
- **English only**. Non-English songs in JamendoLyrics were skipped because the wav2vec 2.0 checkpoint is English-only.
- **DALI is skipped**. Its YouTube-derived audio is unreliable in the time budget; planned for M3 with a vetted audio source.

## Reproducibility

```bash
git clone --depth 1 https://github.com/f90/jamendolyrics.git
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt && pip install torchcodec  # mp3 backend
python run.py --jamendo jamendolyrics --out outputs --all          # with Demucs
python run.py --jamendo jamendolyrics --out outputs_raw --all --no-demucs  # baseline
```

Hardware: M4 Max MacBook, MPS device for both Demucs and wav2vec 2.0. Total wall-clock for both full runs: **241 s**.
