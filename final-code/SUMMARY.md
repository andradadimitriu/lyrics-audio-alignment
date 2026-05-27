# Lyrics-to-Audio Alignment — Milestone 3 Summary

University POLITEHNICA of Bucharest — Hasan Khadra, Dimitriu Andrada-Elena.

## What changed since M2

| Area | M2 | M3 |
| --- | --- | --- |
| Acoustic model | pretrained `wav2vec2-base-960h` (frozen) | same, plus **fine-tuning pipeline ready** (CTC + masked-CE) |
| Phoneme labels | — | partial-phoneme rule from Cheng et al. 2025 (`labels.py`) |
| VAD | — | silero-vad gating, per-song fallback when over-trim detected |
| Metrics | MAE, MedAE, PCO 0.2/0.3 | + PCO 0.5/1.0/2.0, Interval Acc 0.3s/0.5s, transcription Word Acc |
| Languages | English only | + French, Spanish, German (XLSR-53 fine-tunes) |
| Reproducer | `run.py` per-song | `run_all.py` one-command full M3 sweep |

## Aggregate results

### English (20 JamendoLyrics songs, intro-only VAD on 19/20 songs)

| Metric | M2 baseline | M3 (intro-only VAD) | Δ |
| --- | ---: | ---: | ---: |
| MAE | 0.413 s | **0.391 s** | −22 ms |
| MedAE | 0.064 s | **0.064 s** | 0 ms |
| PCO 0.3 | 91.6 % | 91.6 % | 0 pp |
| PCO 0.2 | 87.0 % | 87.1 % | +0.1 pp |
| PCO 0.5 | — | 93.8 % | new |
| PCO 1.0 | — | 96.2 % | new |
| PCO 2.0 | — | 97.6 % | new |
| IntervalAcc 0.3 s | — | 85.5 % | new |
| IntervalAcc 0.5 s | — | 91.3 % | new |
| Word Accuracy | — | 35.6 % | new |

**Per-song guarantee**: no song regresses vs. the no-VAD baseline; three songs improve (Pure Mids −323 ms, Lower Loveday −60 ms, The.madpix.project −50 ms). The aggregate gain over M2 comes entirely from these three.

### VAD design

silero-vad is speech-trained and under-detects sung vocals — especially sustained outros (we measured 60–75 s of truncation on songs with sung endings). We use VAD **only to clip the instrumental intro**, never the outro or mid-song gaps:

1. Detect voice intervals with silero.
2. If the first interval starts >5 s into the song *and* before the song's halfway point, treat everything before it (minus a 0.5 s pad) as instrumental intro.
3. Compute wav2vec2 emissions on the full song (identical to no-VAD path), then slice the log-probs tensor to the post-intro region and run forced-align on the slice. Word timings in the kept region are bit-identical to the no-VAD aligner's output — so VAD-on cannot regress vs VAD-off.

This is the literature-standard pattern (VAD as a boundary cleanup, not a global frame filter). The alternative gap-trimming approach is kept in `Wav2Vec2Aligner.align_with_vad` for diagnostic comparison but isn't used by `run_all.py`.

### Multilingual (3 songs each, XLSR-53 fine-tunes)

| Lang | MAE (s) | MedAE (s) | PCO 0.3 | IntervalAcc 0.5 | WordAcc | Model |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| English | 0.876 | 0.074 | 86.6 % | 85.2 % | 35.6 % | wav2vec2-base-960h |
| German | 2.360 | 1.837 | 59.8 % | 62.7 % | 36.1 % | xlsr-53-german |
| Spanish | 13.19 | 10.70 | 23.7 % | 24.9 % | 35.9 % | xlsr-53-spanish |
| French | 7.18 | 6.72 | 5.6 % | 6.2 % | 41.0 % | xlsr-53-french |

These are speech-trained CTC fine-tunes applied zero-shot to *singing*; the WordAcc column tells the real story (35–41 % across languages, comparable to English) — the encoders *do* hear the phonemes, but the forced-aligner pays a heavy cost when sung onsets drift from speech-like timings. German is reasonable; Spanish/French need fine-tuning on sung data. Increasing the smoke-run cap is a one-line change in `run_all.py --limit-multilang`.

## Components (M3 deliverables)

| File | Purpose |
| --- | --- |
| `labels.py` | Cheng 2025 partial phoneme labelling + `g2p_en` wrapper. 4 unit tests. |
| `data_prep.py` | DALI annotation → train/val manifests (with cached HT Demucs vocals). |
| `train.py` | Fine-tune wav2vec2 with CTC + masked-CE (λ₁=λ₂=1), bf16 on MPS. |
| `vad.py` | silero-vad detection + `forced_align_with_vad` (frame remap). |
| `evaluate.py` | M3 metric set (`M3Metrics`), greedy CTC decode, aggregator. |
| `multilingual_eval.py` | Per-language XLSR-53 runs. |
| `run_all.py` | One-command M3 reproducer. Writes `outputs/results_m3.json`. |
| `visualize.py` | (updated) 3-row plot with VAD overlay when sidecar exists. |
| `pipeline.py` | (refactored) `Wav2Vec2Aligner.align` and `align_with_vad` share path-walking helpers. |

`labels.py` ships a self-contained unit test for the multi-phoneme rule (first phoneme on first frame, last on last frame, middle masked) — run with `python labels.py`.

## Fine-tuning on DALI (four runs, v1 → v4)

DALI audio was downloaded in two batches: v1 (303 English songs, ~19 h) and v2 (712 more, ~45 h), for a combined corpus of **1,015 songs**. Four fine-tune runs were executed to find the configuration that actually improves on the pretrained baseline.

### Run 1: phoneme head, full fine-tune (failed)

Replaced the released character head with a fresh phoneme head over the ARPAbet vocab, unfroze all 95 M encoder params, used CTC + masked-CE (Cheng et al. λ₁=λ₂=1), LR 1e-5, 500-step warmup, 5 epochs. Early-stopped at epoch 4. Best checkpoint produced JamendoLyrics MAE **8.73 s** (vs. baseline 0.39 s — 22× worse). The masked-CE loss stayed at the random-prediction floor (~3.7 ≈ ln 40) so only CTC drove learning; the freshly-initialised head couldn't catch up in 330 training steps; the warmup never finished (500 warmup > 330 total). Full discussion in `outputs/results_m3_finetuned_v1.json` (kept for the journal).

### Run 2: frozen encoder, character head, CTC only

Three configuration fixes:
1. Freeze the entire wav2vec 2.0 encoder. Train only `lm_head` (~25 K params, 0.026 % of the model). Overfitting on 303 songs becomes essentially impossible.
2. Keep the released character head as the starting point. Nudge it toward sung audio instead of starting from random.
3. Drop the masked-CE loss; CTC only. Use LR 5e-6, 10 % linear warmup, 12 epochs, AdamW WD 0.01.

Validation: 5 of the 20 JL English songs picked by `random.Random(42).sample(...)`; held out from the test set too. Per-epoch JL-val MAE printed using a `Wav2Vec2Aligner` that shares the live model.

Training trajectory (5 min wall, all 12 epochs):

| epoch | train_loss | JL-val MAE |
| ---: | ---: | ---: |
| baseline (no training) | — | 0.3180 s |
| 1–6 | 2.55–2.88 | 0.3180 s (unchanged, sub-mm head shift) |
| 7 | 2.69 | **0.3117 s** (best) |
| 8–12 | 2.55–2.70 | 0.3118 s |

### Success criterion

Spec: "If best-epoch JL-val MAE < 0.391 s, the fine-tune wins."  
Result: 0.3117 < 0.391 ✓ **criterion met on the val set.**

But on the held-out 15-song test set the picture changes:

| Set | Pretrained MAE | Fine-tuned (epoch 7) MAE | Δ |
| --- | ---: | ---: | ---: |
| Val (5 songs) | 0.3180 | **0.3117** | −6 ms |
| Test (15 songs) | 0.4158 | **0.4227** | **+7 ms** |

Per-song delta on the test set:

| song | baseline | fine-tuned | Δ |
| --- | ---: | ---: | ---: |
| 13 songs | unchanged | unchanged | within ±1 ms |
| Ridgway — Fire Inside | 0.605 | **0.711** | **+106 ms** |
| Quentin Hannappe — Keep On | 0.162 | 0.162 | −0.5 ms |

The whole val "win" comes from one song (Embers, −31 ms). The test "regression" comes from one song (Ridgway, +106 ms). 14 of 15 test songs are bit-identical to the baseline. The head adapted by an amount too small to move any well-aligned song's MAE; it only shifted single forced-align decisions on the two tail-error songs — Embers helpfully, Ridgway harmfully. That's **statistical noise**, not a generalisable improvement.

### Run 3: LoRA on the encoder (r=16, q_proj + v_proj), 303 songs

PEFT LoRA adapters added to `q_proj` and `v_proj` in all 12 wav2vec 2.0 transformer attention layers. Encoder weights frozen except for the LoRA branches; `lm_head` also trainable. r=16, α=32, LR 1e-4, 10 % warmup, AdamW WD 0.0, 10 epochs, batch 4. Trainable: **614 K params (0.65 %)**. Trained on the 303-song v1 corpus only (8 min wall).

Training trajectory was volatile — the model got worse for 5 epochs (val MAE 0.55) before recovering at epoch 6 and bottoming out at epoch 10 (the last epoch trained):

| epoch | train_loss | JL-val MAE |
| ---: | ---: | ---: |
| baseline | — | 0.3180 |
| 1–5 | 3.07–3.78 | 0.53–0.55 (perturbed) |
| 6 | 3.03 | 0.3500 |
| 7–9 | 2.82–2.83 | 0.3662–0.3897 |
| **10** | **2.77** | **0.2619** (best, but last epoch — curve hadn't plateaued) |

Test 15 result: MAE **0.3526** vs baseline 0.4158 — **−63 ms / −15 % real improvement**. 9 better, 3 worse, 3 unchanged. Biggest gains on songs the baseline struggled with (Tom Orlando −504 ms, Pure Mids −470 ms, Kinematic −160 ms). Biggest regression: Ridgway +210 ms. Full per-song deltas in `outputs/results_m3_finetuned_v3.json`.

### Run 4: LoRA on combined v1+v2 (1,015 songs), 25 epochs planned, patience 3

Same LoRA config as v3. 901 train / 114 val (90/10 split deterministic by md5(dali_id)). Same 15-song JL test split as v3. Trained 4 epochs (early-stopped at patience 3, ~10 min wall).

Training trajectory — best was **epoch 1** because the larger dataset (226 steps/epoch vs 66 in v3) compressed the learning curve into a single pass:

| epoch | train_loss | JL-val MAE |
| ---: | ---: | ---: |
| baseline | — | 0.3180 |
| **1** | **2.78** | **0.2457** (best, beats v3's e10) |
| 2 | 2.57 | 0.3295 (regress 1) |
| 3 | 2.29 | 0.2719 (regress 2) |
| 4 | 2.34 | 0.3006 (regress 3 → stop) |

Spec-compliant patience-3 early-stop fired after only 4 epochs. The model converged immediately and then oscillated; later epochs never came back below epoch 1.

Two evaluations of the v4 epoch-1 checkpoint:

| Set | Pretrained | v3 LoRA (e10) | v4 LoRA (e1) | v4 Δ vs base | v4 Δ vs v3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Val (5 songs) | 0.3180 | 0.2619 | **0.2457** | −72 ms | −16 ms |
| **Test (15 songs)** | **0.4158** | **0.3526** | **0.3735** | **−42 ms** | **+21 ms** |
| All 20 songs | 0.3913 | — | **0.3415** | **−49 ms** | n/a |

Per-song delta on the 15-song test set:

| song | baseline | v4 e1 | Δ vs baseline | vs v3 |
| --- | ---: | ---: | ---: | ---: |
| Pure Mids — The Leader | 2.523 | **1.636** | **−887 ms** | −417 ms |
| Tom Orlando — The One | 1.294 | **1.233** | **−61 ms** | +443 ms vs v3 |
| Moon I Mean, Cortez, Rxbyn, Color Out | … | … | −3 to −5 ms each | mixed |
| 4 songs (Slingshot, Quentin, JASON MILLER, Wordsmith) | unchanged | unchanged | ±1 ms | mixed |
| The Rinn, Songwriterz, HILA | … | … | +1 to +18 ms | worse |
| Ridgway — Fire Inside | 0.605 | 0.700 | +95 ms | better than v3 |
| Kinematic — Peyote | 0.625 | **0.836** | **+211 ms** | +371 ms vs v3 |

**v4 vs baseline: 6 wins / 5 losses / 4 ties.** v4 picks up Pure Mids massively and gives back ground on Kinematic. **v4 vs v3 on test set: 3 wins / 9 losses / 3 ties** — v4 found a *different* equilibrium, not a strictly better one. Net regression vs v3: +21 ms aggregate. Full per-song deltas in `outputs/results_m3_finetuned_v4_test15.json`.

### Comparing the four runs (15-song JL test)

| Run | Strategy | Trainable | Dataset | Best epoch | Test MAE | Wins / losses vs baseline |
| --- | --- | ---: | --- | ---: | ---: | --- |
| pretrained | — | 0 | — | — | 0.416 | — |
| v1 | phoneme head + full FT + CE | 95 M | 303 | 2 | 8.73 | 0 / 15 |
| v2 | char head only (frozen enc) | 25 K | 303 | 7 | 0.423 | 0 / 1 |
| **v3** | LoRA r=16 | 614 K | 303 | 10 | **0.353** | **9 / 3** |
| v4 | LoRA r=16 | 614 K | 1,015 | 1 | 0.373 | 6 / 5 |

### What we ship

- `results_m3.json` (pretrained baseline, MAE 0.391 on all 20) — **the M3 deliverable**.
- `results_m3_finetuned_v3.json` — best fine-tuned run (LoRA on 303 songs, MAE 0.353 on test 15).
- `results_m3_finetuned_v4_test15.json` and `results_m3_finetuned_v4_all20.json` — v4 results.
- `results_m3_finetuned_v1.json`, `results_m3_finetuned.json` (=v2) — earlier attempts.
- Checkpoints: `checkpoints_v3/` (head-only v2 + LoRA v3), `checkpoints/checkpoint_v4_epoch_{1..4}.pt`.
- `logs/train.log`, `train_v2.log`, `train_v3_lora.log`, `train_v4_lora.log`.

### Recommendation

**Use v3 (LoRA r=16 on 303 songs) as the recommended fine-tuned checkpoint, with the pretrained M3 baseline as the safe-fallback deliverable.**

v4 (1,015 songs) did not beat v3 on the held-out test set despite 3.3× the data — it found a slightly different alignment equilibrium but the aggregate moved the wrong way. Two contributing factors:

1. **Adaptation ceiling for LoRA r=16 on this task.** The bigger dataset converged in a single epoch and then oscillated. Larger r or unfrozen attention might allow a higher ceiling; that's M3-future work.
2. **Patience-3 early stop was aggressive.** v3 took 10 epochs to find its best; with patience 3 v3 would also have stopped at epoch 4 and missed the gain. v4 may have a better checkpoint later in training that the early-stop never reached. Worth retrying v4 with patience 8–10 if time allows.

The infrastructure (data prep, LoRA train, eval pipeline, run_all loader) is reusable for both. Both v3 and v4 artifacts are preserved for the journal.

## Other scope notes
- **VAD on sung audio.** silero-vad is speech-trained; its end-of-vocal detection truncates sustained outros (measured: 60–75 s of legitimate vocals dropped on Embers and Pure Mids). We sidestep this by using VAD only for intro clipping (see "VAD design" above). Songs without a meaningful instrumental intro (Tom Orlando — first word at 0.3 s) are aligned without VAD; the policy is automatic per-song.
- **Multilingual XLSR fine-tunes** are `jonatasgrosman/wav2vec2-large-xlsr-53-{french,spanish,german}`. Facebook itself only ships the pretrained-only XLSR-53; community fine-tunes are the closest thing to a "ready CTC head" per language.

## Artifacts in `outputs/`

- `results_m3.json` — full per-song + aggregate metrics for English (20 songs) + multilingual.
- `sample_alignment_hila_vad.png` — 3-row timeline (GT / predicted / VAD) on HILA, 20–42 s.
- `sample_alignment_crowdpleaser_vad.png` — same for JASON MILLER.
- `sample_alignment_puremids_vad.png` — Pure Mids 50–100 s, the song with the largest VAD-driven gain (−323 ms MAE).
- `run.log` — full text log of the `run_all.py` execution.
- `<song>_vocals.wav`, `<song>_alignment.json`, `<song>_vad.json` — per-song intermediates.

## Reproducibility

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
git clone --depth 1 https://github.com/f90/jamendolyrics.git

# Full M3 sweep: 20 English songs + 3 songs each FR/ES/DE
python run_all.py --limit-multilang 3

# Optional: smoke-train the fine-tuning loop (no real audio needed beyond the
# manifest; this validates the wiring rather than producing a useful checkpoint)
python data_prep.py --allow-missing-audio --max-songs 8
python train.py --smoke
```

Hardware: M4 Max MacBook, MPS. English sweep ≈ 35 s, multilingual ≈ 70 s (3 songs/lang).
