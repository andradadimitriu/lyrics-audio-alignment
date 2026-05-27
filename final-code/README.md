# Lyrics-to-Audio Alignment

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

Place the DALI v2 annotations at `dali/annot_tismir/`. Place the
JamendoLyrics dataset at `jamendolyrics/`. Neither is committed.

## Download DALI audio

```bash
cd dali_audio_downloader
python select_songs.py
DALI_COOKIES_FROM_BROWSER=chrome python download.py
cd ..
python extract_vocals_dali.py \
    --manifest dali_audio_downloader/manifest.json \
    --out-dir  dali_audio_downloader/vocals
```

`DALI_COOKIES_FROM_BROWSER` may be `chrome`, `safari`, `firefox`, or `brave` —
yt-dlp reads YouTube cookies from a logged-in browser to clear the bot check.

## Train

```bash
python data_prep.py
python train.py --epochs 12 --batch-size 4 --lr 5e-6
# or:
python train_lora.py --epochs 10 --batch-size 4 --lr 1e-4
```

## Evaluate

```bash
python run.py                 # 3 validation songs
python run.py --all           # all 20 English JamendoLyrics songs
python run_all.py             # English + FR/ES/DE multilingual
python eval_checkpoints.py    # sweep over checkpoints/, write winner
```

## Visualize

```bash
python visualize.py --song <basename> --out-dir outputs/
```
