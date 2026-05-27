"""
Voice-activity detection on vocals stems (silero-vad), and a forced-alignment
helper that restricts alignment to voice-active frames only.

For lyrics alignment, gating frames by VAD removes long instrumental sections
so the aligner doesn't have to "spend" tokens on silence. We concatenate the
voice-active CTC log-probs, run forced_align on the concatenation, then map
each per-frame alignment back to its absolute time in the original timeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import torchaudio.functional as TF


# ----------------------------------------------------------------------------
# Frame rate of wav2vec2 (50 Hz at 16 kHz)
# ----------------------------------------------------------------------------
SAMPLE_RATE = 16000
FRAME_HOP = 320  # samples per output frame
FRAME_DUR_SEC = FRAME_HOP / SAMPLE_RATE


_VAD_MODEL = None


def _silero():
    global _VAD_MODEL
    if _VAD_MODEL is None:
        from silero_vad import load_silero_vad
        _VAD_MODEL = load_silero_vad()
    return _VAD_MODEL


@dataclass
class VoiceInterval:
    start_sec: float
    end_sec: float


def detect_voice_intervals(waveform: torch.Tensor, sr: int = SAMPLE_RATE,
                           threshold: float = 0.45,
                           min_speech_ms: int = 250,
                           min_silence_ms: int = 200,
                           pad_ms: int = 100) -> List[VoiceInterval]:
    """waveform: [1, T] or [T] at 16 kHz mono."""
    from silero_vad import get_speech_timestamps
    if waveform.dim() == 2:
        wav1d = waveform.mean(dim=0)
    else:
        wav1d = waveform
    if sr != SAMPLE_RATE:
        import torchaudio
        wav1d = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(wav1d.unsqueeze(0)).squeeze(0)
    ts = get_speech_timestamps(
        wav1d, _silero(),
        threshold=threshold,
        sampling_rate=SAMPLE_RATE,
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=min_silence_ms,
        speech_pad_ms=pad_ms,
        return_seconds=True,
    )
    return [VoiceInterval(start_sec=float(t["start"]), end_sec=float(t["end"])) for t in ts]


def intervals_to_frame_mask(intervals: Sequence[VoiceInterval],
                            n_frames: int,
                            frame_dur_sec: float = FRAME_DUR_SEC) -> torch.Tensor:
    """Boolean mask of length n_frames; True where the frame center lies in a voice interval."""
    mask = torch.zeros(n_frames, dtype=torch.bool)
    if not intervals:
        return mask
    centers = (torch.arange(n_frames, dtype=torch.float32) + 0.5) * frame_dur_sec
    for iv in intervals:
        lo, hi = iv.start_sec, iv.end_sec
        mask |= (centers >= lo) & (centers < hi)
    return mask


# ----------------------------------------------------------------------------
# VAD-gated forced alignment
# ----------------------------------------------------------------------------

def forced_align_with_vad(log_probs: torch.Tensor,
                          target_token_ids: Sequence[int],
                          blank_id: int,
                          vad_intervals: Sequence[VoiceInterval],
                          frame_dur_sec: float = FRAME_DUR_SEC,
                          ) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
    """
    Restrict forced_align to voice-active frames, then map back to absolute time.

    Args
    ----
    log_probs : [T, V] log-softmax over the full song.
    target_token_ids : flat target sequence (CTC vocabulary).
    blank_id : CTC blank id.
    vad_intervals : voice-active intervals (seconds).
    frame_dur_sec : duration of one wav2vec2 frame.

    Returns
    -------
    (frame_token_ids, frame_to_abs_idx, mask)
      frame_token_ids : per *kept* frame, the token id from forced_align.
      frame_to_abs_idx : long tensor mapping kept-frame index -> absolute frame
                         index in the original log_probs.
      mask : bool tensor of length T marking kept frames.
    """
    n_frames = log_probs.shape[0]
    mask = intervals_to_frame_mask(vad_intervals, n_frames, frame_dur_sec)
    if not mask.any():
        # Fall back to no VAD if the detector returns nothing useful
        mask = torch.ones(n_frames, dtype=torch.bool)
    kept_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    log_probs_kept = log_probs[mask]  # [T_kept, V]

    log_probs_b = log_probs_kept.unsqueeze(0)
    targets_b = torch.tensor([list(target_token_ids)], dtype=torch.int32)
    alignments, _ = TF.forced_align(log_probs_b, targets_b, blank=blank_id)
    return alignments.squeeze(0).tolist(), kept_indices, mask


# ----------------------------------------------------------------------------
# CLI for ad-hoc inspection
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, json, sys
    from pipeline import load_audio
    ap = argparse.ArgumentParser()
    ap.add_argument("vocals", help="path to vocals wav")
    ap.add_argument("--out", default=None, help="optional JSON sidecar to write")
    args = ap.parse_args()
    wav, _sr = load_audio(args.vocals, SAMPLE_RATE)
    intervals = detect_voice_intervals(wav)
    print(f"{len(intervals)} voice intervals; "
          f"total active = {sum(iv.end_sec-iv.start_sec for iv in intervals):.1f}s")
    if args.out:
        with open(args.out, "w") as f:
            json.dump([{"start": iv.start_sec, "end": iv.end_sec} for iv in intervals], f)
        print(f"wrote {args.out}")
