"""
M3 evaluation: extends pipeline.evaluate with extra PCO thresholds, an interval
accuracy metric, and a soft transcription-based word accuracy.

Metrics
-------
MAE, MedAE          : mean/median absolute onset error (seconds).
PCO_0.{2,3,5,10,20} : % of words with |onset_err| <= T.
IntervalAcc_T       : % of words where |start_err| <= T AND |end_err| <= T.
WordAcc             : % of GT words present (after normalisation) in the model's
                      greedy-decoded transcript. Soft, recall-style — singing
                      makes a strict WER unhelpful.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, asdict
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from pipeline import WordSpan  # reuse the M2 dataclass


# ----------------------------------------------------------------------------
# Metrics container
# ----------------------------------------------------------------------------

@dataclass
class M3Metrics:
    n_words: int = 0
    n_aligned: int = 0
    MAE: float = float("nan")
    MedAE: float = float("nan")
    PCO_02: float = 0.0
    PCO_03: float = 0.0
    PCO_05: float = 0.0
    PCO_10: float = 0.0
    PCO_20: float = 0.0
    IntervalAcc_03: float = 0.0
    IntervalAcc_05: float = 0.0
    WordAcc: Optional[float] = None  # filled only if decoded_text provided


def _pco(errs: Sequence[float], t: float) -> float:
    return float(np.mean([1.0 if e <= t else 0.0 for e in errs])) if errs else 0.0


# ----------------------------------------------------------------------------
# Word accuracy via greedy decode
# ----------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Z']+")


def _norm_tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.upper())


def word_accuracy(decoded_text: str, lyric_words: Sequence[str]) -> float:
    """% of GT words (with duplicates) present in the decoded transcript.

    A word is considered "present" if it appears as a contiguous token in the
    decoded stream. We use a multiset intersection so a word repeated N times in
    the lyrics is credited N times only if it appears N times in the decode.
    """
    if not lyric_words:
        return 0.0
    from collections import Counter
    decoded_counts = Counter(_norm_tokens(decoded_text))
    lyric_tokens = []
    for w in lyric_words:
        toks = _norm_tokens(w)
        lyric_tokens.extend(toks)
    if not lyric_tokens:
        return 0.0
    matched = 0
    remaining = Counter(decoded_counts)
    for t in lyric_tokens:
        if remaining.get(t, 0) > 0:
            matched += 1
            remaining[t] -= 1
    return matched / len(lyric_tokens)


def greedy_ctc_decode_chars(log_probs: "np.ndarray | torch.Tensor",
                            id_to_token: dict, blank_id: int,
                            word_delim_id: int) -> str:
    """Greedy CTC decode for a character vocab (e.g. wav2vec2-base-960h)."""
    if hasattr(log_probs, "argmax"):
        ids = log_probs.argmax(dim=-1) if hasattr(log_probs, "dim") else log_probs.argmax(axis=-1)
        ids = ids.cpu().tolist() if hasattr(ids, "cpu") else list(ids)
    else:
        ids = list(log_probs)
    out = []
    prev = None
    for tid in ids:
        if tid == prev:
            continue
        prev = tid
        if tid == blank_id:
            continue
        if tid == word_delim_id:
            out.append(" ")
            continue
        out.append(id_to_token.get(tid, ""))
    return "".join(out).strip()


# ----------------------------------------------------------------------------
# Top-level evaluate
# ----------------------------------------------------------------------------

def evaluate_m3(predicted: Sequence[WordSpan],
                gt: Sequence[Tuple[float, float]],
                decoded_text: Optional[str] = None,
                lyric_words: Optional[Sequence[str]] = None) -> M3Metrics:
    n = min(len(predicted), len(gt))
    onset_errs: List[float] = []
    pair_errs: List[Tuple[float, float]] = []
    for i in range(n):
        ps, pe = predicted[i].start_sec, predicted[i].end_sec
        gs, ge = gt[i]
        if math.isnan(ps) or math.isnan(pe):
            continue
        onset_errs.append(abs(ps - gs))
        pair_errs.append((abs(ps - gs), abs(pe - ge)))

    m = M3Metrics(n_words=n, n_aligned=len(onset_errs))
    if onset_errs:
        m.MAE = float(np.mean(onset_errs))
        m.MedAE = float(np.median(onset_errs))
        m.PCO_02 = _pco(onset_errs, 0.2)
        m.PCO_03 = _pco(onset_errs, 0.3)
        m.PCO_05 = _pco(onset_errs, 0.5)
        m.PCO_10 = _pco(onset_errs, 1.0)
        m.PCO_20 = _pco(onset_errs, 2.0)
    if pair_errs:
        m.IntervalAcc_03 = float(np.mean([1.0 if (s <= 0.3 and e <= 0.3) else 0.0
                                          for s, e in pair_errs]))
        m.IntervalAcc_05 = float(np.mean([1.0 if (s <= 0.5 and e <= 0.5) else 0.0
                                          for s, e in pair_errs]))
    if decoded_text is not None and lyric_words is not None:
        m.WordAcc = word_accuracy(decoded_text, lyric_words)
    return m


def aggregate(per_song: Sequence[dict], keys: Iterable[str] = None) -> dict:
    """Mean across songs, ignoring NaNs and Nones per key."""
    if keys is None:
        keys = ("MAE", "MedAE", "PCO_02", "PCO_03", "PCO_05", "PCO_10", "PCO_20",
                "IntervalAcc_03", "IntervalAcc_05", "WordAcc")
    out = {}
    for k in keys:
        vals = []
        for s in per_song:
            v = s.get(k)
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if math.isnan(fv):
                continue
            vals.append(fv)
        out[k] = float(np.mean(vals)) if vals else float("nan")
    out["n_songs"] = len(per_song)
    return out
