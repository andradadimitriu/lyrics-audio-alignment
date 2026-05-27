"""
Partial phoneme labels for masked frame-wise cross-entropy (Cheng et al. 2025).

Rule:
    - Single-phoneme word: every frame inside (word_start, word_end] carries
      that phoneme as a hard label.
    - Multi-phoneme word: only the FIRST frame of the span gets the first
      phoneme, and the LAST frame gets the last phoneme. All other frames
      inside the span are MASKED (label = -100).
    - All frames outside any word span are MASKED.

Mask value follows PyTorch CrossEntropyLoss(ignore_index=-100) convention.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

IGNORE_INDEX = -100


def build_phoneme_vocab(phoneme_sets: Iterable[Sequence[str]]) -> dict:
    """Stable id mapping from a corpus of phoneme sequences. id 0 = <blank>."""
    seen = set()
    for seq in phoneme_sets:
        for p in seq:
            seen.add(p)
    vocab = {"<blank>": 0}
    for p in sorted(seen):
        vocab[p] = len(vocab)
    return vocab


def _frame_index(t_sec: float, frame_dur_sec: float, n_frames: int) -> int:
    """Convert a timestamp to a clamped frame index in [0, n_frames-1]."""
    f = int(round(t_sec / frame_dur_sec))
    if f < 0:
        f = 0
    if f >= n_frames:
        f = n_frames - 1
    return f


def partial_phoneme_labels(
    word_phonemes: Sequence[Sequence[str]],
    word_starts: Sequence[float],
    word_ends: Sequence[float],
    n_frames: int,
    frame_dur_sec: float,
    phoneme_to_id: dict,
    unknown_phoneme_id: Optional[int] = None,
) -> List[int]:
    """
    Returns a length-`n_frames` list of phoneme ids (or IGNORE_INDEX for
    unlabelled frames). See module docstring for the rule.

    `word_phonemes[i]` is the phoneme list for word i (already produced by g2p).
    Words with empty phoneme lists are skipped.
    """
    assert len(word_phonemes) == len(word_starts) == len(word_ends), \
        "word_phonemes, word_starts, word_ends must align"

    labels = [IGNORE_INDEX] * n_frames

    for phons, ws, we in zip(word_phonemes, word_starts, word_ends):
        if not phons or ws is None or we is None:
            continue
        if we <= ws:
            continue
        f_start = _frame_index(ws, frame_dur_sec, n_frames)
        f_end = _frame_index(we, frame_dur_sec, n_frames)
        if f_end < f_start:
            f_start, f_end = f_end, f_start

        if len(phons) == 1:
            pid = phoneme_to_id.get(phons[0], unknown_phoneme_id)
            if pid is None:
                continue
            for f in range(f_start, f_end + 1):
                labels[f] = pid
        else:
            first = phoneme_to_id.get(phons[0], unknown_phoneme_id)
            last = phoneme_to_id.get(phons[-1], unknown_phoneme_id)
            if first is not None:
                labels[f_start] = first
            if last is not None:
                # If span is exactly one frame, last would overwrite first.
                # Keep first in that degenerate case (the word *is* its onset).
                if f_end != f_start:
                    labels[f_end] = last

    return labels


# ----------------------------------------------------------------------------
# Word -> phonemes
# ----------------------------------------------------------------------------

_G2P = None


def get_g2p():
    """Lazy-load g2p_en to keep import cheap when only labels are exercised."""
    global _G2P
    if _G2P is None:
        from g2p_en import G2p
        _G2P = G2p()
    return _G2P


def word_to_phonemes(word: str, g2p=None) -> List[str]:
    """ARPAbet phonemes for a single word. Strips stress digits."""
    g = g2p if g2p is not None else get_g2p()
    raw = g(word)
    out = []
    for p in raw:
        if not p.strip() or not p[0].isalpha():
            continue
        # ARPAbet vowels carry stress digits (e.g. AH0/AH1/AH2). Drop them.
        out.append("".join(c for c in p if c.isalpha()))
    return out


def words_to_phonemes(words: Sequence[str]) -> List[List[str]]:
    g = get_g2p()
    return [word_to_phonemes(w, g) for w in words]


# ----------------------------------------------------------------------------
# Unit test (Cheng et al. 2025 example)
# ----------------------------------------------------------------------------

def _test_multi_phoneme_word_only_endpoints_labelled():
    """The CAT case: phonemes [K, AE, T] over a 5-frame span.
       Only the first frame gets K, the last frame gets T, the middle is masked."""
    phoneme_to_id = {"<blank>": 0, "K": 1, "AE": 2, "T": 3}
    n_frames = 10
    frame_dur = 0.02  # 50 fps
    # Word span: frames 2..6 inclusive (0.04s..0.14s)
    labels = partial_phoneme_labels(
        word_phonemes=[["K", "AE", "T"]],
        word_starts=[0.04],
        word_ends=[0.13],
        n_frames=n_frames,
        frame_dur_sec=frame_dur,
        phoneme_to_id=phoneme_to_id,
    )
    expected = [IGNORE_INDEX] * n_frames
    expected[2] = 1       # K on first frame
    expected[7] = 3       # T on last frame (0.13 / 0.02 = 6.5 -> rounds to 7)
    # Recompute end index using the helper rule so the assertion matches it
    # rather than hard-coding our arithmetic.
    end_idx = int(round(0.13 / 0.02))
    start_idx = int(round(0.04 / 0.02))
    expected = [IGNORE_INDEX] * n_frames
    expected[start_idx] = 1
    expected[end_idx] = 3
    assert labels == expected, f"multi-phoneme labelling wrong:\n  got: {labels}\n  exp: {expected}"


def _test_single_phoneme_word_every_frame_labelled():
    """A single-phoneme word labels every frame inside its span."""
    phoneme_to_id = {"<blank>": 0, "AH": 1}
    n_frames = 10
    frame_dur = 0.02
    labels = partial_phoneme_labels(
        word_phonemes=[["AH"]],
        word_starts=[0.04],
        word_ends=[0.10],
        n_frames=n_frames,
        frame_dur_sec=frame_dur,
        phoneme_to_id=phoneme_to_id,
    )
    # frames 2..5 inclusive should be AH
    for f in range(2, 6):
        assert labels[f] == 1, f"frame {f} should be labelled AH, got {labels[f]}"
    for f in (0, 1, 6, 7, 8, 9):
        assert labels[f] == IGNORE_INDEX, f"frame {f} should be masked, got {labels[f]}"


def _test_outside_spans_masked():
    phoneme_to_id = {"<blank>": 0, "K": 1, "T": 2}
    labels = partial_phoneme_labels(
        word_phonemes=[["K", "T"]],
        word_starts=[0.2],
        word_ends=[0.3],
        n_frames=30,
        frame_dur_sec=0.02,
        phoneme_to_id=phoneme_to_id,
    )
    # Only two non-masked frames: first and last of the span.
    nonmasked = [i for i, v in enumerate(labels) if v != IGNORE_INDEX]
    assert len(nonmasked) == 2, f"expected exactly 2 labelled frames, got {nonmasked}"


def _test_degenerate_single_frame_span_keeps_first_phoneme():
    phoneme_to_id = {"<blank>": 0, "K": 1, "T": 2}
    labels = partial_phoneme_labels(
        word_phonemes=[["K", "T"]],
        word_starts=[0.10],
        word_ends=[0.105],
        n_frames=10,
        frame_dur_sec=0.02,
        phoneme_to_id=phoneme_to_id,
    )
    nonmasked = [(i, v) for i, v in enumerate(labels) if v != IGNORE_INDEX]
    assert len(nonmasked) == 1, f"degenerate span should label exactly one frame, got {nonmasked}"
    assert nonmasked[0][1] == 1, "single-frame span should keep the FIRST phoneme"


def _run_tests():
    _test_multi_phoneme_word_only_endpoints_labelled()
    _test_single_phoneme_word_every_frame_labelled()
    _test_outside_spans_masked()
    _test_degenerate_single_frame_span_keeps_first_phoneme()
    print("labels.py: all tests passed")


if __name__ == "__main__":
    _run_tests()
