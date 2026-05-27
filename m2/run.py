"""
Run the lyrics-alignment pipeline over a list of JamendoLyrics songs.

Usage:
    python run.py                # default: 3 validation songs
    python run.py --all          # all 20 English songs
    python run.py --no-demucs    # skip source separation (raw audio)
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

from pipeline import (
    Wav2Vec2Aligner, get_device, run_on_song,
)

DEFAULT_SONGS = [
    "HILA_-_Give_Me_the_Same",
    "Avercage_-_Embers",
    "Quentin_Hannappe_-_Keep_On",
]


def english_songs(jamendo_root: str):
    """Read the metadata csv and return all English-language song basenames."""
    out = []
    csv_path = os.path.join(jamendo_root, "JamendoLyrics.csv")
    with open(csv_path, "r") as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) < 7:
                continue
            filepath = parts[1]
            language = parts[6]
            if language.strip() == "English":
                base = filepath.replace(".mp3", "")
                out.append(base)
    return out


def aggregate(per_song):
    keys = ("MAE", "MedAE", "PCO_03", "PCO_02")
    out = {}
    for k in keys:
        vals = [s[k] for s in per_song if s[k] == s[k]]  # drop nans
        out[k] = float(statistics.mean(vals)) if vals else float("nan")
    out["n_songs"] = len(per_song)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jamendo", default="jamendolyrics",
                    help="path to cloned jamendolyrics repo")
    ap.add_argument("--out", default="outputs",
                    help="output directory")
    ap.add_argument("--all", action="store_true",
                    help="run on all 20 English songs")
    ap.add_argument("--no-demucs", action="store_true",
                    help="skip HT Demucs (raw audio baseline)")
    ap.add_argument("--demucs-device", default=None,
                    help="device for demucs (default: same as alignment)")
    ap.add_argument("--songs", nargs="*", default=None,
                    help="specific song basenames to run on")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"alignment device: {device}")
    demucs_device = args.demucs_device or device

    if args.songs:
        songs = args.songs
    elif args.all:
        songs = english_songs(args.jamendo)
    else:
        songs = DEFAULT_SONGS
    print(f"running on {len(songs)} songs: {songs}")

    aligner = Wav2Vec2Aligner(device=device)

    per_song = []
    t0 = time.time()
    for s in songs:
        try:
            t_song = time.time()
            res = run_on_song(s, args.jamendo, args.out, aligner,
                              use_demucs=not args.no_demucs,
                              demucs_device=demucs_device)
            res["seconds"] = round(time.time() - t_song, 1)
            per_song.append(res)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    agg = aggregate(per_song)
    total = round(time.time() - t0, 1)
    print(f"\nDone in {total}s. Aggregate: {agg}")

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({"per_song": per_song, "aggregate": agg,
                   "settings": {"demucs": not args.no_demucs,
                                "device": device}}, f, indent=2)
    print(f"wrote {os.path.join(args.out, 'results.json')}")


if __name__ == "__main__":
    main()
