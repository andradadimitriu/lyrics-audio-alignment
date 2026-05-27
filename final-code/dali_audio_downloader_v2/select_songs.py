"""Pick the next 1000 English DALI songs that v1 has NOT already attempted.

We read ../dali_audio_downloader/manifest.json and union downloaded[*].dali_id
with failed[*].dali_id into an exclusion set. Anything in that set is off
limits for v2 (no point re-attempting v1's confirmed dead links, and no point
re-downloading songs the LoRA agent is already training on).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import DALI

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
ANNOT_DIR = PROJECT_ROOT / "dali" / "annot_tismir"
V1_MANIFEST = PROJECT_ROOT / "dali_audio_downloader" / "manifest.json"
OUT_PATH = ROOT / "candidates.json"

TARGET_COUNT = 1000


def main() -> int:
    if not ANNOT_DIR.is_dir():
        print(f"ERROR: annotations dir missing: {ANNOT_DIR}", file=sys.stderr)
        return 1
    if not V1_MANIFEST.is_file():
        print(f"ERROR: v1 manifest missing: {V1_MANIFEST}", file=sys.stderr)
        return 1

    v1 = json.load(V1_MANIFEST.open())
    already_attempted: set[str] = set()
    for entry in v1.get("downloaded", []):
        already_attempted.add(entry["dali_id"])
    for entry in v1.get("failed", []):
        already_attempted.add(entry["dali_id"])
    print(f"v1 attempted: {len(already_attempted)} dali_ids (will be excluded)")

    print("Loading all DALI annotations...")
    data = DALI.get_the_DALI_dataset(str(ANNOT_DIR))
    print(f"Loaded {len(data)} entries")

    candidates: list[dict] = []
    excluded = 0
    not_english = 0
    no_url = 0
    for dali_id, entry in data.items():
        if dali_id in already_attempted:
            excluded += 1
            continue
        info = entry.info
        lang = (info.get("metadata", {}) or {}).get("language", "")
        if not isinstance(lang, str) or lang.lower() != "english":
            not_english += 1
            continue
        audio = info.get("audio", {}) or {}
        yt_id = audio.get("url")
        if not yt_id or not isinstance(yt_id, str):
            no_url += 1
            continue
        candidates.append(
            {
                "dali_id": dali_id,
                "youtube_id": yt_id,
                "title": info.get("title", ""),
                "artist": info.get("artist", ""),
                "language": "english",
            }
        )

    candidates.sort(key=lambda r: r["dali_id"])
    chosen = candidates[:TARGET_COUNT]

    print(
        f"english pool (after v1 exclusion): {len(candidates)}  "
        f"excluded={excluded} not_english={not_english} no_url={no_url}"
    )

    with OUT_PATH.open("w") as f:
        json.dump(
            {
                "count": len(chosen),
                "target": TARGET_COUNT,
                "v1_manifest": str(V1_MANIFEST),
                "v1_excluded_count": len(already_attempted),
                "candidates": chosen,
            },
            f,
            indent=2,
        )
    print(f"Wrote {len(chosen)} candidates → {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
