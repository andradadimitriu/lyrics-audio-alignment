"""Run HT Demucs on every v2 audio file and save 16 kHz mono vocals WAVs.

Mirrors the parent project's pipeline.separate_vocals (segment=7, shifts=0)
but without importing pipeline.py (which would drag in transformers).

Idempotent: skips songs whose vocals/<dali_id>.wav already exists. Updates
manifest.json in place after each success, adding `vocals_path` per entry
and incrementing statistics.vocals_extracted.

Stops when:
  * every successful download has vocals, OR
  * 2.5 hours elapsed (DALI_VOCALS_BUDGET to override).

Default device: MPS (Apple Silicon). Set DALI_DEMUCS_DEVICE=cpu to override.
Default workers: 1 (serial — concurrent MPS access is unsafe per the parent
project's own comment). Set DALI_DEMUCS_WORKERS=2 to attempt 2-parallel via
multiprocessing (one model per child process).
"""
from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from demucs.apply import apply_model
from demucs.pretrained import get_model

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "manifest.json"
VOCALS_DIR = ROOT / "vocals"
LOG_DIR = ROOT / "logs"
EXTRACT_LOG = LOG_DIR / "extract_vocals.log"

DEMUCS_DEVICE = os.environ.get("DALI_DEMUCS_DEVICE", "").strip() or None
WORKERS = int(os.environ.get("DALI_DEMUCS_WORKERS", "1"))
TIME_BUDGET = int(os.environ.get("DALI_VOCALS_BUDGET", str(int(2.5 * 60 * 60))))
PROGRESS_EVERY = 1  # print after every successful song

VOCALS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(EXTRACT_LOG, mode="a")],
)
log = logging.getLogger("extract_vocals")


def pick_device(override: str | None) -> str:
    if override:
        return override
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


_MODEL = None


def get_demucs(device: str):
    global _MODEL
    if _MODEL is None:
        _MODEL = get_model("htdemucs")
        _MODEL.eval()
    return _MODEL


def separate_vocals(audio_path: str, out_wav: str, device: str) -> float:
    """Extract vocals stem to a 16 kHz mono WAV. Returns duration_seconds."""
    model = get_demucs(device)
    sr_model = model.samplerate  # 44100 for htdemucs

    # Load source audio at native rate, stereo if available.
    data, sr = librosa.load(audio_path, sr=None, mono=False)
    if data.ndim == 1:
        wav = torch.from_numpy(data).unsqueeze(0)
    else:
        wav = torch.from_numpy(data)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)  # demucs expects stereo
    if sr != sr_model:
        wav = torchaudio.transforms.Resample(sr, sr_model)(wav)

    # Normalize per the demucs convention.
    ref = wav.mean(0)
    mean = ref.mean().item()
    std = ref.std().item()
    if std < 1e-8:
        std = 1.0
    wav_norm = (wav - mean) / std

    sources = apply_model(
        model, wav_norm.unsqueeze(0),
        device=device, progress=False, segment=7,
        shifts=0, num_workers=0,
    )
    sources = sources * std + mean
    vocals_idx = model.sources.index("vocals")
    vocals = sources[0, vocals_idx]  # [channels, T]
    if vocals.dim() == 2 and vocals.shape[0] > 1:
        vocals = vocals.mean(dim=0, keepdim=True)
    if sr_model != 16000:
        vocals = torchaudio.transforms.Resample(sr_model, 16000)(vocals)

    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
    arr = vocals.squeeze(0).cpu().numpy().astype(np.float32)
    sf.write(out_wav, arr, 16000)
    return float(arr.shape[0]) / 16000.0


def worker_loop(task_q: "mp.Queue", result_q: "mp.Queue", device: str) -> None:
    """Subprocess worker: pull tasks until None, write vocals, post results."""
    while True:
        item = task_q.get()
        if item is None:
            return
        dali_id, audio_path, out_wav = item
        t0 = time.time()
        try:
            dur = separate_vocals(audio_path, out_wav, device)
            result_q.put(("ok", dali_id, out_wav, dur, time.time() - t0, ""))
        except Exception as e:  # noqa: BLE001
            tb = "".join(traceback.format_exception_only(type(e), e))[:200]
            result_q.put(("err", dali_id, out_wav, 0.0, time.time() - t0, tb))


def fmt_hms(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


def main() -> int:
    if not MANIFEST_PATH.is_file():
        print(f"ERROR: missing {MANIFEST_PATH}", file=sys.stderr)
        return 1

    device = pick_device(DEMUCS_DEVICE)
    workers = max(1, WORKERS)

    manifest = json.load(MANIFEST_PATH.open())
    entries = manifest.get("downloaded", [])
    print(f"[extract] device={device} workers={workers} budget={TIME_BUDGET}s")
    print(f"[extract] {len(entries)} downloaded entries to consider")
    log.info("RUN START device=%s workers=%d budget=%ds n=%d",
             device, workers, TIME_BUDGET, len(entries))

    # Build the task list, skipping any entry whose vocals.wav already exists.
    pending: list[tuple[str, str, str]] = []
    already = 0
    for e in entries:
        dst = VOCALS_DIR / f"{e['dali_id']}.wav"
        if dst.is_file() and dst.stat().st_size > 1024:
            e["vocals_path"] = str(dst.resolve())
            already += 1
            continue
        pending.append((e["dali_id"], e["audio_path"], str(dst)))
    print(f"[extract] pending={len(pending)}  already-have-vocals={already}")

    t_start = time.time()
    done = 0
    failed = 0

    def commit_entry(dali_id: str, vocals_abs: str) -> None:
        for e in entries:
            if e["dali_id"] == dali_id:
                e["vocals_path"] = vocals_abs
                return

    def write_manifest():
        stats = manifest.setdefault("statistics", {})
        stats["vocals_extracted"] = sum(1 for e in entries if e.get("vocals_path"))
        stats["vocals_finished_at"] = datetime.now(tz=timezone.utc).isoformat()
        tmp = MANIFEST_PATH.with_suffix(".json.tmp")
        with tmp.open("w") as f:
            json.dump(manifest, f, indent=2)
        tmp.replace(MANIFEST_PATH)

    if workers == 1:
        # Same-process serial — fastest single-MPS path.
        for i, (dali_id, audio_path, out_wav) in enumerate(pending):
            elapsed = time.time() - t_start
            if elapsed >= TIME_BUDGET:
                print(f"[extract] time budget hit ({fmt_hms(elapsed)}); stopping")
                log.info("BUDGET HIT elapsed=%.1fs done=%d failed=%d", elapsed, done, failed)
                break
            t = time.time()
            try:
                separate_vocals(audio_path, out_wav, device)
                done += 1
                commit_entry(dali_id, str(Path(out_wav).resolve()))
                if done % 5 == 0:
                    write_manifest()
                dt = time.time() - t
                avg = elapsed / max(done, 1)
                eta = (len(pending) - done - failed) * avg
                print(f"[{done+failed:4d}/{len(pending)}] ok {dali_id[:12]} "
                      f"{dt:5.1f}s   elapsed {fmt_hms(elapsed)}   eta {fmt_hms(eta)}",
                      flush=True)
                log.info("vocals-ok %s %.1fs", dali_id, dt)
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"[{done+failed:4d}/{len(pending)}] FAIL {dali_id}: {type(e).__name__}: {e}")
                log.warning("vocals-fail %s :: %s", dali_id, e)
    else:
        # mp parallel: spawn N workers, each with its own Demucs.
        mp.set_start_method("spawn", force=True)
        task_q: mp.Queue = mp.Queue()
        result_q: mp.Queue = mp.Queue()
        for item in pending:
            task_q.put(item)
        for _ in range(workers):
            task_q.put(None)
        procs = [
            mp.Process(target=worker_loop, args=(task_q, result_q, device), daemon=True)
            for _ in range(workers)
        ]
        for p in procs:
            p.start()
        total = len(pending)
        seen = 0
        while seen < total:
            elapsed = time.time() - t_start
            if elapsed >= TIME_BUDGET:
                print(f"[extract] time budget hit ({fmt_hms(elapsed)}); terminating workers")
                for p in procs:
                    p.terminate()
                break
            try:
                status, dali_id, out_wav, dur, dt, err = result_q.get(timeout=5)
            except Exception:
                continue
            seen += 1
            if status == "ok":
                done += 1
                commit_entry(dali_id, str(Path(out_wav).resolve()))
                if done % 5 == 0:
                    write_manifest()
                avg = elapsed / max(done, 1)
                eta = (total - seen) * avg
                print(f"[{seen:4d}/{total}] ok {dali_id[:12]} "
                      f"{dt:5.1f}s   elapsed {fmt_hms(elapsed)}   eta {fmt_hms(eta)}",
                      flush=True)
                log.info("vocals-ok %s %.1fs", dali_id, dt)
            else:
                failed += 1
                print(f"[{seen:4d}/{total}] FAIL {dali_id}: {err}")
                log.warning("vocals-fail %s :: %s", dali_id, err)
        for p in procs:
            p.join(timeout=2)

    write_manifest()
    elapsed = time.time() - t_start
    print()
    print("V2 VOCALS EXTRACTION COMPLETE")
    print(f"Extracted now:    {done}")
    print(f"Already on disk:  {already}")
    print(f"Failed:           {failed}")
    print(f"Total with vocals: {already + done}/{len(entries)}")
    print(f"Elapsed:          {fmt_hms(elapsed)}")
    log.info("RUN END done=%d failed=%d already=%d elapsed=%.1fs",
             done, failed, already, elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
