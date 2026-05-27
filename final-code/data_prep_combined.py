"""
Build a combined training manifest from v1 + v2 DALI downloads.

Inputs:
    dali_audio_downloader/manifest.json     (v1, 303 songs, vocals at dali_audio_downloader/vocals/<id>.wav)
    dali_audio_downloader_v2/manifest.json  (v2, 712 songs, vocals_path already in each entry)
    dali/annot_tismir/<id>.gz               (extracted by data_prep.py earlier)

Outputs:
    dali_audio_downloader_v2/manifest_combined.json   (normalized merge for reference)
    manifests/train.json, manifests/val.json          (training entries, 90/10 by md5 hash)

Drops entries whose vocals.wav is missing or empty (<1 KB).
Deterministic split: hash(dali_id) % 10 == 0 -> val, else train.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import List, Optional

V1_MANIFEST = "dali_audio_downloader/manifest.json"
V1_VOCALS_DIR = "dali_audio_downloader/vocals"
V2_MANIFEST = "dali_audio_downloader_v2/manifest.json"
COMBINED_MANIFEST = "dali_audio_downloader_v2/manifest_combined.json"
ANNOT_DIR = Path("dali/annot_tismir")
MANIFEST_DIR = Path("manifests")
MIN_VOCALS_BYTES = 1024


def vocals_path_for(entry: dict, batch: str) -> str:
    if "vocals_path" in entry:
        return entry["vocals_path"]
    # v1 convention
    return os.path.join(V1_VOCALS_DIR, f"{entry['dali_id']}.wav")


def normalize(entries: List[dict], batch: str) -> List[dict]:
    out = []
    for e in entries:
        vp = vocals_path_for(e, batch)
        out.append({
            "dali_id": e["dali_id"],
            "audio_path": e.get("audio_path", ""),
            "vocals_path": vp,
            "duration_seconds": e.get("duration_seconds"),
            "title": e.get("title", ""),
            "artist": e.get("artist", ""),
            "language": e.get("language", ""),
            "source_batch": e.get("source_batch", batch),
        })
    return out


def vocals_ok(vp: str) -> bool:
    if not os.path.exists(vp):
        return False
    try:
        return os.path.getsize(vp) >= MIN_VOCALS_BYTES
    except OSError:
        return False


def load_annotation(dali_id: str):
    p = ANNOT_DIR / f"{dali_id}.gz"
    with gzip.open(p, "rb") as f:
        return pickle.load(f)


def build_training_entry(e: dict) -> Optional[dict]:
    """Returns the schema train_lora.py / DALICharDataset expects (or None)."""
    try:
        ann = load_annotation(e["dali_id"])
    except Exception:
        return None
    word_items = ann.annotations["annot"]["words"]
    phoneme_items = ann.annotations["annot"].get("phonemes", [])
    words, starts, ends, phons = [], [], [], []
    for i, w in enumerate(word_items):
        text = (w.get("text") or "").strip()
        t = w.get("time")
        if not text or not t:
            continue
        s, en = float(t[0]), float(t[1])
        if en <= s:
            continue
        ph = phoneme_items[i].get("text") if i < len(phoneme_items) else []
        ph = [p for p in (ph or []) if p]
        if not ph:
            continue
        words.append(text)
        starts.append(s)
        ends.append(en)
        phons.append(ph)
    if not words:
        return None
    return {
        "id": e["dali_id"],
        "vocals_path": e["vocals_path"],
        "source_batch": e["source_batch"],
        "words": words,
        "word_starts": starts,
        "word_ends": ends,
        "word_phonemes": phons,
    }


def split_train_val(entries: List[dict], val_frac: float = 0.10):
    """Deterministic by md5(dali_id). val if (hash % bucket_size) < (val_frac * bucket_size)."""
    train, val = [], []
    for e in entries:
        h = int(hashlib.md5(e["id"].encode()).hexdigest(), 16) % 10000
        if h < int(val_frac * 10000):
            val.append(e)
        else:
            train.append(e)
    return train, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.10)
    args = ap.parse_args()

    v1 = json.load(open(V1_MANIFEST))["downloaded"]
    v2 = json.load(open(V2_MANIFEST))["downloaded"]
    print(f"[combined] v1={len(v1)} v2={len(v2)} total raw={len(v1)+len(v2)}")

    norm = normalize(v1, "v1") + normalize(v2, "v2")
    # Dedupe by dali_id, prefer v2 (newer extraction)
    by_id = {}
    for e in norm:
        # If duplicate: keep the one with existing vocals
        prev = by_id.get(e["dali_id"])
        if prev is None or (not vocals_ok(prev["vocals_path"]) and vocals_ok(e["vocals_path"])):
            by_id[e["dali_id"]] = e
    deduped = list(by_id.values())
    print(f"[combined] deduped: {len(deduped)}")

    # Vocals filter
    usable = [e for e in deduped if vocals_ok(e["vocals_path"])]
    print(f"[combined] vocals ok: {len(usable)} (dropped {len(deduped)-len(usable)} missing/empty)")

    # Write combined reference manifest
    Path(os.path.dirname(COMBINED_MANIFEST)).mkdir(parents=True, exist_ok=True)
    with open(COMBINED_MANIFEST, "w") as f:
        json.dump({"downloaded": usable,
                   "statistics": {
                       "v1_count": sum(1 for e in usable if e["source_batch"]=="v1"),
                       "v2_count": sum(1 for e in usable if e["source_batch"]=="v2"),
                       "total": len(usable),
                   }}, f, indent=2)
    print(f"[combined] wrote {COMBINED_MANIFEST}")

    # Build training entries (needs DALI annotation pickle)
    train_entries = []
    n_no_ann = 0
    for e in usable:
        te = build_training_entry(e)
        if te is None:
            n_no_ann += 1
            continue
        train_entries.append(te)
    print(f"[combined] training entries: {len(train_entries)} "
          f"(skipped {n_no_ann} with no usable annotation)")

    train, val = split_train_val(train_entries, args.val_frac)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_DIR / "train.json", "w") as f:
        json.dump(train, f)
    with open(MANIFEST_DIR / "val.json", "w") as f:
        json.dump(val, f)
    print(f"[combined] wrote {len(train)} train / {len(val)} val to {MANIFEST_DIR}/")

    # Quick coverage summary
    from collections import Counter
    c_v = Counter(e.get("source_batch", "?") for e in train_entries)
    print(f"[combined] source batch counts: {dict(c_v)}")


if __name__ == "__main__":
    main()
