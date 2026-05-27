"""Pick the first N English DALI songs with a YouTube URL → candidates.json."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import DALI

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent.parent
ANNOT_DIR = PROJECT_ROOT / "dali" / "annot_tismir"
OUT_PATH = ROOT / "candidates.json"

TARGET_COUNT = int(os.environ.get("DALI_CANDIDATE_COUNT", "1000"))


def main() -> int:
    if not ANNOT_DIR.is_dir():
        print(f"ERROR: annotations dir missing: {ANNOT_DIR}", file=sys.stderr)
        return 1

    print("Loading all DALI annotations...")
    data = DALI.get_the_DALI_dataset(str(ANNOT_DIR))
    print(f"Loaded {len(data)} entries")

    candidates: list[dict] = []
    not_english = no_url = 0
    for dali_id, entry in data.items():
        info = entry.info
        lang = (info.get("metadata", {}) or {}).get("language", "")
        if not isinstance(lang, str) or lang.lower() != "english":
            not_english += 1
            continue
        yt_id = (info.get("audio", {}) or {}).get("url")
        if not yt_id or not isinstance(yt_id, str):
            no_url += 1
            continue
        candidates.append({
            "dali_id": dali_id,
            "youtube_id": yt_id,
            "title": info.get("title", ""),
            "artist": info.get("artist", ""),
            "language": "english",
        })

    candidates.sort(key=lambda r: r["dali_id"])
    chosen = candidates[:TARGET_COUNT]
    print(f"english pool: {len(candidates)}  not_english={not_english} no_url={no_url}")

    with OUT_PATH.open("w") as f:
        json.dump({"count": len(chosen), "target": TARGET_COUNT, "candidates": chosen}, f, indent=2)
    print(f"Wrote {len(chosen)} candidates → {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
