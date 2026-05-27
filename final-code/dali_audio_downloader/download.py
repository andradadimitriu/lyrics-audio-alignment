"""YouTube audio downloader for the DALI English subset.

Reads candidates.json, downloads each song's audio via yt-dlp, and writes
manifest.json. Resumes by skipping any audio file already on disk.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import yt_dlp

ROOT = Path(__file__).resolve().parent
CANDIDATES_PATH = ROOT / "candidates.json"
AUDIO_DIR = ROOT / "audio"
LOG_DIR = ROOT / "logs"
MANIFEST_PATH = ROOT / "manifest.json"
DOWNLOAD_LOG = LOG_DIR / "download.log"

COOKIES_PATH = Path.home() / ".config" / "yt-dlp" / "cookies.txt"
COOKIES_FROM_BROWSER = os.environ.get("DALI_COOKIES_FROM_BROWSER", "chrome").strip() or None
FFMPEG_BIN = "/opt/homebrew/bin/ffmpeg"
FFPROBE_BIN = "/opt/homebrew/bin/ffprobe"

ATTEMPT_TARGET = int(os.environ.get("DALI_ATTEMPT_TARGET", "1000"))
SUCCESS_TARGET = int(os.environ.get("DALI_SUCCESS_TARGET", "700"))
TIME_BUDGET_SECONDS = int(os.environ.get("DALI_TIME_BUDGET", str(3 * 60 * 60)))
N_WORKERS = int(os.environ.get("DALI_WORKERS", "8"))
PER_SONG_SOCKET_TIMEOUT = 30
PROGRESS_INTERVAL = 30

YOUTUBE_BASE = "https://www.youtube.com/watch?v="

LOG_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(DOWNLOAD_LOG, mode="a")],
)
log = logging.getLogger("dali_dl")


@dataclass
class State:
    attempted: int = 0
    successful: int = 0
    skipped: int = 0
    failed: int = 0
    started_at: float | None = None
    downloaded: list[dict] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)
    stop_flag: bool = False
    lock: Lock = field(default_factory=Lock)


def yt_dl_opts(out_path_stem: Path) -> dict:
    opts: dict = {
        "format": "bestaudio[ext=m4a]/bestaudio",
        "outtmpl": str(out_path_stem) + ".%(ext)s",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": PER_SONG_SOCKET_TIMEOUT,
        "retries": 2,
        "fragment_retries": 2,
        "concurrent_fragment_downloads": 4,
        "ffmpeg_location": FFMPEG_BIN,
        "prefer_ffmpeg": True,
        # EJS challenge solver is required for signed-in YouTube requests.
        "remote_components": ["ejs:github"],
    }
    if COOKIES_FROM_BROWSER:
        opts["cookiesfrombrowser"] = (COOKIES_FROM_BROWSER,)
    elif COOKIES_PATH.is_file():
        opts["cookiefile"] = str(COOKIES_PATH)
    return opts


def find_existing(dali_id: str) -> Path | None:
    for ext in ("m4a", "webm", "opus", "ogg", "mp3"):
        p = AUDIO_DIR / f"{dali_id}.{ext}"
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def validate_audio(path: Path) -> tuple[bool, float]:
    if path.stat().st_size < 1024:
        return False, 0.0
    try:
        out = subprocess.run(
            [
                FFPROBE_BIN, "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True, text=True, timeout=15,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        log.warning("validate-fail %s :: %s", path.name, e)
        return False, 0.0
    if out.returncode != 0:
        return False, 0.0
    try:
        dur = float(out.stdout.strip())
    except ValueError:
        return False, 0.0
    return (dur > 0), dur


def download_one(candidate: dict) -> dict:
    dali_id = candidate["dali_id"]
    youtube_id = candidate["youtube_id"]
    title = candidate.get("title", "")
    artist = candidate.get("artist", "")
    stem = AUDIO_DIR / dali_id

    existing = find_existing(dali_id)
    if existing is not None:
        ok, dur = validate_audio(existing)
        if ok:
            log.info("skip-existing %s %s", dali_id, existing.name)
            return {
                "status": "skipped",
                "dali_id": dali_id,
                "audio_path": str(existing.resolve()),
                "duration_seconds": dur,
                "title": title,
                "artist": artist,
                "language": "english",
            }
        try:
            existing.unlink()
        except OSError:
            pass

    url = YOUTUBE_BASE + youtube_id

    def _attempt() -> Exception | None:
        try:
            with yt_dlp.YoutubeDL(yt_dl_opts(stem)) as ydl:
                ydl.download([url])
            return None
        except Exception as e:
            return e

    err = _attempt()
    if err is not None:
        msg = str(err).lower()
        if "429" in msg or "403" in msg or "rate" in msg or "forbidden" in msg:
            log.info("backoff-retry %s :: %s", dali_id, err)
            time.sleep(30)
            err = _attempt()

    written = find_existing(dali_id)
    if err is not None or written is None:
        if written is not None:
            try:
                written.unlink()
            except OSError:
                pass
        reason = str(err).splitlines()[-1][:200] if err else "no audio file written"
        log.warning("failed %s yt=%s :: %s", dali_id, youtube_id, reason)
        return {"status": "failed", "dali_id": dali_id, "youtube_id": youtube_id, "reason": reason}

    ok, dur = validate_audio(written)
    if not ok:
        try:
            written.unlink()
        except OSError:
            pass
        return {
            "status": "failed",
            "dali_id": dali_id,
            "youtube_id": youtube_id,
            "reason": "validation: unreadable / empty",
        }

    log.info("success %s %s dur=%.1fs", dali_id, written.name, dur)
    return {
        "status": "success",
        "dali_id": dali_id,
        "audio_path": str(written.resolve()),
        "duration_seconds": dur,
        "title": title,
        "artist": artist,
        "language": "english",
    }


def fmt_hms(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


def write_manifest(state: State, started_iso: str, finished_iso: str, elapsed: float) -> None:
    payload = {
        "downloaded": state.downloaded,
        "failed": state.failures,
        "statistics": {
            "total_attempted": state.attempted,
            "successful": state.successful,
            "failed": state.failed,
            "skipped_existing": state.skipped,
            "started_at": started_iso,
            "finished_at": finished_iso,
            "elapsed_seconds": round(elapsed, 1),
        },
    }
    tmp = MANIFEST_PATH.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(MANIFEST_PATH)


def main() -> int:
    if not CANDIDATES_PATH.is_file():
        print(f"ERROR: missing {CANDIDATES_PATH}", file=sys.stderr)
        return 1

    with CANDIDATES_PATH.open() as f:
        candidates = json.load(f)["candidates"][:ATTEMPT_TARGET]

    print(f"Loaded {len(candidates)} candidates")
    print(f"Stop when: success>={SUCCESS_TARGET} or attempts>={ATTEMPT_TARGET} "
          f"or elapsed>={TIME_BUDGET_SECONDS}s")
    if COOKIES_FROM_BROWSER:
        print(f"Cookies: browser={COOKIES_FROM_BROWSER}")
    elif COOKIES_PATH.is_file():
        print(f"Cookies: file={COOKIES_PATH}")
    else:
        print("Cookies: none")
    print(f"Workers: {N_WORKERS}")

    state = State()
    state.started_at = time.time()
    started_iso = datetime.now(tz=timezone.utc).isoformat()
    last_progress_line = state.started_at

    def should_stop() -> bool:
        if state.stop_flag:
            return True
        elapsed = time.time() - (state.started_at or time.time())
        return (state.successful >= SUCCESS_TARGET
                or state.attempted >= ATTEMPT_TARGET
                or elapsed >= TIME_BUDGET_SECONDS)

    def record(result: dict) -> None:
        nonlocal last_progress_line
        with state.lock:
            state.attempted += 1
            if result["status"] == "success":
                state.successful += 1
                state.downloaded.append({k: v for k, v in result.items() if k != "status"})
            elif result["status"] == "skipped":
                state.skipped += 1
                state.successful += 1
                state.downloaded.append({k: v for k, v in result.items() if k != "status"})
            else:
                state.failed += 1
                state.failures.append({k: v for k, v in result.items() if k != "status"})

            now = time.time()
            if now - last_progress_line >= PROGRESS_INTERVAL:
                last_progress_line = now
                elapsed = now - (state.started_at or now)
                print(f"Elapsed {fmt_hms(elapsed)} — attempted {state.attempted} — "
                      f"successful {state.successful} (target {SUCCESS_TARGET}) — "
                      f"failed {state.failed}", flush=True)

            if state.attempted % 25 == 0:
                write_manifest(state, started_iso,
                               datetime.now(tz=timezone.utc).isoformat(),
                               time.time() - (state.started_at or time.time()))

    pending = list(candidates)
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        in_flight: dict = {}

        def fill():
            while len(in_flight) < N_WORKERS * 2 and pending and not should_stop():
                cand = pending.pop(0)
                fut = pool.submit(download_one, cand)
                in_flight[fut] = cand

        fill()
        while in_flight:
            fut = next(as_completed(in_flight))
            cand = in_flight.pop(fut)
            try:
                result = fut.result()
            except Exception as e:
                result = {
                    "status": "failed",
                    "dali_id": cand["dali_id"],
                    "youtube_id": cand["youtube_id"],
                    "reason": f"unhandled: {e!r}",
                }
            record(result)
            if should_stop():
                with state.lock:
                    state.stop_flag = True
                continue
            fill()

    finished_iso = datetime.now(tz=timezone.utc).isoformat()
    elapsed = time.time() - (state.started_at or time.time())
    write_manifest(state, started_iso, finished_iso, elapsed)

    print()
    print(f"Attempted:   {state.attempted}")
    print(f"Successful:  {state.successful} (skip-existing: {state.skipped})")
    print(f"Failed:      {state.failed}")
    print(f"Manifest:    {MANIFEST_PATH.resolve()}")
    print(f"Audio dir:   {AUDIO_DIR.resolve()}")
    print(f"Elapsed:     {fmt_hms(elapsed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
