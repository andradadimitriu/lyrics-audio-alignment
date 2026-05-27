"""Combine v1 + v2 manifests into a single in-memory list and print stats.

Reads ../dali_audio_downloader/manifest.json (v1) and ./manifest.json (v2),
emits a unified list of {dali_id, audio_path, vocals_path, language,
source_batch, title, artist, duration_seconds} on stdout (as JSON), and
prints a per-batch summary.

Usage:
    python3 merge_with_v1.py                  # print JSON list + stats
    python3 merge_with_v1.py --stats-only     # only the summary
    python3 merge_with_v1.py --out merged.json  # ALSO write JSON to disk

By default this does NOT write a merged manifest file — it's just a helper
for the training agent to import or invoke ad-hoc.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
V1_MANIFEST = ROOT.parent / "dali_audio_downloader" / "manifest.json"
V2_MANIFEST = ROOT / "manifest.json"


def load(path: Path) -> dict:
    if not path.is_file():
        print(f"WARNING: manifest missing: {path}", file=sys.stderr)
        return {"downloaded": [], "failed": [], "statistics": {}}
    return json.load(path.open())


def build_merged(v1: dict, v2: dict) -> list[dict]:
    merged: list[dict] = []
    seen: set[str] = set()
    for batch_tag, m in (("v1", v1), ("v2", v2)):
        for e in m.get("downloaded", []):
            if e["dali_id"] in seen:
                continue
            seen.add(e["dali_id"])
            merged.append({
                "dali_id": e["dali_id"],
                "audio_path": e["audio_path"],
                "vocals_path": e.get("vocals_path"),
                "language": e.get("language", "english"),
                "source_batch": e.get("source_batch", batch_tag),
                "title": e.get("title", ""),
                "artist": e.get("artist", ""),
                "duration_seconds": e.get("duration_seconds", 0.0),
            })
    return merged


def summary(merged: list[dict], v1: dict, v2: dict) -> None:
    v1_n = len(v1.get("downloaded", []))
    v2_n = len(v2.get("downloaded", []))
    v1_fail = len(v1.get("failed", []))
    v2_fail = len(v2.get("failed", []))
    total_dur = sum(e.get("duration_seconds", 0.0) or 0.0 for e in merged)
    with_vocals = sum(1 for e in merged if e.get("vocals_path"))
    vocals_dur = sum(
        e.get("duration_seconds", 0.0) or 0.0 for e in merged if e.get("vocals_path")
    )
    print(f"v1: {v1_n} downloaded / {v1_n + v1_fail} attempted = "
          f"{(v1_n / max(v1_n + v1_fail, 1)) * 100:.1f}% success")
    print(f"v2: {v2_n} downloaded / {v2_n + v2_fail} attempted = "
          f"{(v2_n / max(v2_n + v2_fail, 1)) * 100:.1f}% success")
    print(f"combined unique songs:   {len(merged)}")
    print(f"with vocals_path:        {with_vocals}")
    print(f"total audio hours:       {total_dur/3600:.2f}")
    print(f"hours w/ vocals stems:   {vocals_dur/3600:.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--out", help="optional path to write merged JSON list")
    args = ap.parse_args()

    v1 = load(V1_MANIFEST)
    v2 = load(V2_MANIFEST)
    merged = build_merged(v1, v2)
    summary(merged, v1, v2)

    if args.out:
        Path(args.out).write_text(json.dumps(merged, indent=2))
        print(f"wrote: {args.out}", file=sys.stderr)

    if not args.stats_only:
        print()
        print(json.dumps(merged, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
