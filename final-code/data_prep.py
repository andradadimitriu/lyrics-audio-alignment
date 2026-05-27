"""
Build a DALI-based training manifest for M3 fine-tuning.

Inputs:
    dali_audio_downloader/manifest.json   — list of downloaded songs
    dali_audio_downloader/vocals/<id>.wav — HT Demucs vocals (from extract_vocals_dali.py)
    dali/DALI_v2.0.zip                    — DALI annotation pickles
    dali/annot_tismir/<id>.gz             — cached extracted annotations

Outputs:
    manifests/train.json, manifests/val.json
        Each entry:
            { "id": ..., "vocals_path": ..., "words": [...],
              "word_starts": [...], "word_ends": [...],
              "word_phonemes": [[...], ...] }

90/10 split deterministically by song-id hash.

Usage:
    python data_prep.py
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import pickle
import sys
import zipfile
from pathlib import Path
from typing import List, Optional

DALI_ZIP = "dali/DALI_v2.0.zip"
ANNOT_DIR = Path("dali/annot_tismir")
DOWNLOADER_MANIFEST = "dali_audio_downloader/manifest.json"
VOCALS_DIR = Path("dali_audio_downloader/vocals")
MANIFEST_DIR = Path("manifests")


def extract_annotations(zip_path: str = DALI_ZIP, out_dir: Path = ANNOT_DIR) -> None:
    """Unpack annot_tismir/*.gz from the zip if not already on disk."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if list(out_dir.glob("*.gz")):
        return
    print(f"[data_prep] extracting annotations from {zip_path} ...")
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.startswith("annot_tismir/") and name.endswith(".gz"):
                dst = out_dir / Path(name).name
                with z.open(name) as src, open(dst, "wb") as dstf:
                    dstf.write(src.read())


def load_annotation(dali_id: str):
    p = ANNOT_DIR / f"{dali_id}.gz"
    with gzip.open(p, "rb") as f:
        return pickle.load(f)


def build_entry(dali_id: str, vocals_path: str) -> Optional[dict]:
    try:
        ann = load_annotation(dali_id)
    except Exception as e:
        print(f"  [skip {dali_id}] annotation load failed: {e}")
        return None
    word_items = ann.annotations["annot"]["words"]
    phoneme_items = ann.annotations["annot"].get("phonemes", [])
    words, starts, ends, phons = [], [], [], []
    for i, w in enumerate(word_items):
        text = (w.get("text") or "").strip()
        t = w.get("time")
        if not text or not t:
            continue
        s, e = float(t[0]), float(t[1])
        if e <= s:
            continue
        ph = phoneme_items[i].get("text") if i < len(phoneme_items) else []
        ph = [p for p in (ph or []) if p]
        if not ph:
            continue  # words without phonemes are useless for the CE loss
        words.append(text)
        starts.append(s)
        ends.append(e)
        phons.append(ph)
    if not words:
        return None
    return {
        "id": dali_id,
        "vocals_path": vocals_path,
        "words": words,
        "word_starts": starts,
        "word_ends": ends,
        "word_phonemes": phons,
    }


def split_train_val(entries: List[dict], val_frac: float = 0.10) -> (List[dict], List[dict]):
    """Deterministic split by song-id hash. Same id always → same bucket."""
    train, val = [], []
    for e in entries:
        h = int(hashlib.md5(e["id"].encode()).hexdigest(), 16)
        bucket = (h % 10000) / 10000.0
        if bucket < val_frac:
            val.append(e)
        else:
            train.append(e)
    return train, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=DOWNLOADER_MANIFEST)
    ap.add_argument("--vocals-dir", default=str(VOCALS_DIR))
    ap.add_argument("--val-frac", type=float, default=0.10)
    args = ap.parse_args()

    if not os.path.exists(DALI_ZIP):
        print(f"ERROR: expected {DALI_ZIP}", file=sys.stderr)
        sys.exit(1)
    extract_annotations()

    downloader = json.load(open(args.manifest))["downloaded"]
    print(f"[data_prep] {len(downloader)} songs in downloader manifest")

    vocals_dir = Path(args.vocals_dir)
    entries = []
    n_missing_vocals = n_bad_ann = 0
    for d in downloader:
        dali_id = d["dali_id"]
        vocals_path = vocals_dir / f"{dali_id}.wav"
        if not vocals_path.exists():
            n_missing_vocals += 1
            continue
        entry = build_entry(dali_id, str(vocals_path))
        if entry is None:
            n_bad_ann += 1
            continue
        entries.append(entry)

    print(f"[data_prep] {len(entries)} usable entries  "
          f"({n_missing_vocals} missing vocals, {n_bad_ann} bad annotation)")

    if not entries:
        print("[data_prep] nothing to write; run extract_vocals_dali.py first.")
        return

    train, val = split_train_val(entries, args.val_frac)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_DIR / "train.json", "w") as f:
        json.dump(train, f)
    with open(MANIFEST_DIR / "val.json", "w") as f:
        json.dump(val, f)
    print(f"[data_prep] wrote {len(train)} train / {len(val)} val to {MANIFEST_DIR}/")

    # Phoneme coverage summary
    from collections import Counter
    c = Counter()
    for e in train:
        for ph in e["word_phonemes"]:
            for p in ph:
                c[p] += 1
    print(f"[data_prep] {len(c)} distinct phonemes in train")
    print(f"            top 10: {c.most_common(10)}")


if __name__ == "__main__":
    main()
