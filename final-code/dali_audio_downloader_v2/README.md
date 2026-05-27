# DALI Audio Downloader v2

Second download batch for the English DALI subset. Extends v1 (in
`../dali_audio_downloader/`) with ~700 more songs and pre-extracted HT
Demucs vocals stems.

**This folder is independent of v1.** The training agent kept using v1's
manifest while this batch was being built; nothing here ever touches v1.

## What's here

```
dali_audio_downloader_v2/
  venv/                  isolated Python 3.10 venv
  select_songs.py        picks 1000 English candidates, EXCLUDING v1's attempts
  download.py            yt-dlp downloader: 8 workers, native m4a/webm, no MP3
  extract_vocals.py      HT Demucs → 16 kHz mono vocals/<dali_id>.wav
  merge_with_v1.py       combine v1 + v2 manifests for the training agent
  candidates.json        1000 chosen songs (dali_id, youtube_id, title, artist)
  audio/                 native bestaudio files (mostly .m4a)
  vocals/                HT Demucs vocals stems (16 kHz mono .wav)
  manifest.json          schema_version=2, includes vocals_path + source_batch
  logs/
    download.log         per-event log
    extract_vocals.log   per-song Demucs timings
    run.out              stdout of the latest download run
```

## What's different from v1

| | v1 | v2 |
|---|---|---|
| Output container | MP3 192 kbps (postprocessed) | native m4a / webm (yt-dlp bestaudio) |
| Validator | `soundfile.info` | `ffprobe` (handles any container) |
| Workers | 5 (later 3) | 8 |
| `concurrent_fragment_downloads` | unset | 4 |
| Success target | 300 | 700 |
| Time budget | 2 h | 3 h |
| Vocals pre-extracted | no | yes (HT Demucs) |
| Cookies | added late (browser=chrome) | required from the start |
| EJS challenge solver | added late (`remote_components=ejs:github`) | required from the start |

Native containers shave ~50% off wall-clock vs. the v1 MP3 transcode path,
and Demucs ingests them just as happily.

## How to re-run

```bash
cd dali_audio_downloader_v2/
source venv/bin/activate
export PATH="/opt/homebrew/bin:$PATH"      # ffmpeg, ffprobe, deno

# (Re)select 1000 English candidates excluding v1's attempts
python3 select_songs.py

# Download. Resumes by skipping any audio/<id>.<ext> already on disk.
DALI_COOKIES_FROM_BROWSER=chrome python3 download.py

# Extract vocals. Idempotent: skips any vocals/<id>.wav already there.
DALI_DEMUCS_DEVICE=mps python3 extract_vocals.py
```

Environment knobs (all optional):

| var | default | purpose |
|---|---|---|
| `DALI_COOKIES_FROM_BROWSER` | `chrome` | browser to pull cookies from (`safari`/`firefox`/`brave` accepted) |
| `DALI_WORKERS` | `8` | download thread-pool size |
| `DALI_SUCCESS_TARGET` | `700` | stop when this many successes |
| `DALI_ATTEMPT_TARGET` | `1000` | stop when this many attempts |
| `DALI_TIME_BUDGET` | `10800` (3 h) | stop after this many seconds |
| `DALI_DEMUCS_DEVICE` | auto (MPS / CUDA / CPU) | force Demucs device |
| `DALI_DEMUCS_WORKERS` | `1` | parallel Demucs processes — *only safe on CPU* |
| `DALI_VOCALS_BUDGET` | `9000` (2.5 h) | hard cap on vocal extraction |

## System dependencies

* `ffmpeg` 8.x + `ffprobe` (`brew install ffmpeg`)
* `deno` (`brew install deno`) — yt-dlp's JS challenge solver
* Chrome (or another supported browser) logged in to youtube.com

Python deps are pinned by `pip install` calls in `select_songs.py`'s setup
notes: `DALI-dataset`, `yt-dlp`, `tqdm`, `soundfile`, `torchaudio`, `demucs`,
`librosa`. Demucs pulls torch as a transitive dependency.

## Manifest schema (v2)

```json
{
  "downloaded": [
    {
      "dali_id": "...",
      "audio_path": "/abs/path/audio/<dali_id>.m4a",
      "vocals_path": "/abs/path/vocals/<dali_id>.wav",
      "duration_seconds": 240.5,
      "title": "...",
      "artist": "...",
      "language": "english",
      "source_batch": "v2"
    }
  ],
  "failed": [{"dali_id": "...", "youtube_id": "...", "reason": "..."}],
  "statistics": {
    "total_attempted": ...,
    "successful": ...,
    "failed": ...,
    "skipped_existing": ...,
    "vocals_extracted": ...,
    "started_at": "...",
    "finished_at": "...",
    "elapsed_seconds": ...
  },
  "schema_version": 2,
  "v1_manifest_reference": "../dali_audio_downloader/manifest.json"
}
```

`vocals_path` is added only after `extract_vocals.py` succeeds for that song;
if extraction was cut short by the time budget some entries may lack it.

## Merging with v1

Once the training agent wants the full v1+v2 set:

```bash
python3 merge_with_v1.py                  # JSON list on stdout + stats
python3 merge_with_v1.py --stats-only     # summary only
python3 merge_with_v1.py --out merged.json  # also write to disk
```

The script de-duplicates by `dali_id` (v1 wins on collisions, though v1 and
v2 are constructed to be disjoint). Each entry carries `source_batch`
("v1" or "v2") so the trainer can stratify or sample if it wants to.
