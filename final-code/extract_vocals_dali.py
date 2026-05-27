"""Pre-extract HT Demucs vocals for every DALI song listed in the downloader manifest.

Idempotent: skips songs whose vocals .wav already exists. Serial on MPS;
concurrent MPS access against a shared Demucs model is unsafe.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from pipeline import separate_vocals, get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="dali_audio_downloader/manifest.json")
    ap.add_argument("--out-dir", default="dali_audio_downloader/vocals")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or get_device()
    print(f"[extract] device={device}  out={out_dir}")

    manifest = json.load(open(args.manifest))
    entries = manifest["downloaded"]
    if args.limit:
        entries = entries[: args.limit]
    print(f"[extract] {len(entries)} songs in manifest")

    n_done = n_skip = n_fail = 0
    t_start = time.time()
    for i, e in enumerate(entries):
        dst = out_dir / f"{e['dali_id']}.wav"
        if dst.exists() and dst.stat().st_size > 1024:
            n_skip += 1
            continue
        t = time.time()
        try:
            separate_vocals(e["audio_path"], str(dst), device=device)
            n_done += 1
            dt = time.time() - t
            elapsed = time.time() - t_start
            done_total = n_done + n_skip
            est_remaining = (len(entries) - i - 1) * (elapsed / max(done_total, 1))
            print(f"[{i+1:3d}/{len(entries)}] {e['dali_id'][:12]} "
                  f"{e['title'][:40]:40s} {dt:5.1f}s  "
                  f"(elapsed {elapsed/60:.1f}m, eta {est_remaining/60:.1f}m)")
        except Exception as ex:
            print(f"[{i+1:3d}] FAIL {e['dali_id']}: {type(ex).__name__}: {ex}")
            n_fail += 1

    print(f"\n[extract] done: {n_done} new, {n_skip} skipped, {n_fail} failed; "
          f"total wall {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
