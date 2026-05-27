# Lyrics-to-Audio Alignment — Milestone 2

University POLITEHNICA of Bucharest. Authors: Hasan Khadra, Dimitriu Andrada-Elena.

## Pipeline

```
song.mp3
   │
   ▼
HT Demucs (pretrained)        ── source separation, public model
   │ vocals.wav (16kHz mono)
   ▼
wav2vec 2.0 (SSL pretrained)  ── facebook/wav2vec2-base-960h
   │ frame-level CTC log-probs
   ▼
torchaudio forced_align       ── Viterbi alignment over CTC logits
   │
   ▼
word-level timestamps  →  MAE / MedAE / PCO0.3 / PCO0.2
```

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
git clone --depth 1 https://github.com/f90/jamendolyrics.git
```

## Running

```bash
# 3 validation songs with HT Demucs (default device autodetect: mps/cuda/cpu)
python run.py --jamendo jamendolyrics --out outputs

# All 20 English songs
python run.py --jamendo jamendolyrics --out outputs --all

# Raw-audio baseline (skip HT Demucs)
python run.py --jamendo jamendolyrics --out outputs_raw --all --no-demucs

# Specific songs
python run.py --jamendo jamendolyrics --out outputs --songs HILA_-_Give_Me_the_Same Avercage_-_Embers
```

Outputs:
- `outputs/<song>_vocals.wav` — Demucs vocals stem
- `outputs/<song>_alignment.json` — predicted (word, start, end) per word
- `outputs/results.json` — per-song and aggregate metrics

## Files

- `pipeline.py` — `Wav2Vec2Aligner`, `separate_vocals`, `evaluate`, `run_on_song`
- `run.py` — CLI orchestrator
- `requirements.txt` — pinned versions
