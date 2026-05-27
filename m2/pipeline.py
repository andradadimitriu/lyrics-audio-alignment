"""
Lyrics-to-Audio Alignment pipeline (Milestone 2).

Architecture:
    song.mp3 -> HT Demucs (vocals stem) -> wav2vec 2.0 (CTC logits)
            -> torchaudio.functional.forced_align -> word-level timestamps

Authors: Hasan Khadra, Dimitriu Andrada-Elena
University POLITEHNICA of Bucharest
"""

from __future__ import annotations

import json
import math
import os
import re
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as TF
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# ----------------------------------------------------------------------------
# Device
# ----------------------------------------------------------------------------

def get_device() -> str:
    """Prefer MPS on Apple Silicon, else CUDA if present, else CPU."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ----------------------------------------------------------------------------
# Audio loading
# ----------------------------------------------------------------------------

def load_audio(path: str, target_sr: int = 16000) -> Tuple[torch.Tensor, int]:
    """Load audio as mono float32 at target_sr. Returns (waveform [1, T], sr)."""
    waveform, sr = torchaudio.load(path)
    # to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
        sr = target_sr
    return waveform, sr


# ----------------------------------------------------------------------------
# Source separation (HT Demucs)
# ----------------------------------------------------------------------------

_DEMUCS_MODEL = None  # cache across calls


def _get_demucs(device: str):
    global _DEMUCS_MODEL
    if _DEMUCS_MODEL is None:
        from demucs.pretrained import get_model
        _DEMUCS_MODEL = get_model("htdemucs")
        _DEMUCS_MODEL.eval()
    return _DEMUCS_MODEL


def separate_vocals(mp3_path: str, out_wav_path: str, device: str = "cpu") -> str:
    """
    Run HT Demucs to extract vocals stem. Saves a 16 kHz mono wav to out_wav_path.
    Returns out_wav_path. Falls back to raw mixed audio if demucs fails.
    """
    try:
        from demucs.apply import apply_model

        model = _get_demucs(device)
        sr_model = model.samplerate

        # Load audio at the model's expected rate, keep stereo for HT Demucs.
        wav, sr = torchaudio.load(mp3_path)
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)  # demucs expects stereo
        if sr != sr_model:
            wav = torchaudio.transforms.Resample(sr, sr_model)(wav)
        # Normalize per the demucs convention (zero-mean, unit-variance over mono ref).
        ref = wav.mean(0)
        mean = ref.mean().item()
        std = ref.std().item()
        if std < 1e-8:
            std = 1.0
        wav_norm = (wav - mean) / std

        sources = apply_model(model, wav_norm.unsqueeze(0),
                              device=device, progress=False, segment=7,
                              shifts=0, num_workers=0)
        # sources: [1, 4, channels, T]
        sources = sources * std + mean
        vocals_idx = model.sources.index("vocals")
        vocals = sources[0, vocals_idx]  # [channels, T]
        # mono
        if vocals.dim() == 2 and vocals.shape[0] > 1:
            vocals = vocals.mean(dim=0, keepdim=True)
        # to 16kHz
        if sr_model != 16000:
            vocals = torchaudio.transforms.Resample(sr_model, 16000)(vocals)

        Path(out_wav_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_wav_path, vocals.squeeze(0).cpu().numpy(), 16000)
        return out_wav_path
    except Exception as e:
        print(f"  [separate_vocals] HT Demucs failed ({type(e).__name__}: {e}); "
              f"falling back to raw audio.")
        import traceback; traceback.print_exc()
        waveform, sr = load_audio(mp3_path, 16000)
        Path(out_wav_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_wav_path, waveform.squeeze(0).cpu().numpy(), sr)
        return out_wav_path


# ----------------------------------------------------------------------------
# Lyrics normalization
# ----------------------------------------------------------------------------

# wav2vec2-base-960h vocab is uppercase A-Z, "'", "|" (space), <pad>, <s>, </s>, <unk>
_KEEP = re.compile(r"[A-Z']")

def normalize_word(w: str) -> str:
    w = w.upper()
    w = "".join(c for c in w if _KEEP.match(c))
    return w

def normalize_words(words: List[str]) -> List[str]:
    """Uppercase, drop punctuation. Keeps empty entries to preserve indexing."""
    return [normalize_word(w) for w in words]


# ----------------------------------------------------------------------------
# Alignment
# ----------------------------------------------------------------------------

@dataclass
class WordSpan:
    word: str
    start_sec: float
    end_sec: float


class Wav2Vec2Aligner:
    """wav2vec 2.0 + forced alignment for one song at a time."""

    def __init__(self, model_id: str = "facebook/wav2vec2-base-960h",
                 device: Optional[str] = None,
                 chunk_seconds: float = 30.0,
                 chunk_overlap_seconds: float = 2.0):
        self.device = device or get_device()
        print(f"  [aligner] loading {model_id} on {self.device} ...")
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id).to(self.device).eval()
        self.sample_rate = 16000
        self.chunk_seconds = chunk_seconds
        self.chunk_overlap_seconds = chunk_overlap_seconds
        # Vocab: id -> token
        self.id_to_token = {v: k for k, v in self.processor.tokenizer.get_vocab().items()}
        self.pad_id = self.processor.tokenizer.pad_token_id  # blank for CTC
        self.word_delim_id = self.processor.tokenizer.word_delimiter_token_id  # '|'

    @torch.inference_mode()
    def _emissions(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Run model in chunks and concatenate frame log-probs.
        waveform: [1, T] at 16kHz. Returns log_probs [T_frames, V].
        """
        T = waveform.shape[1]
        chunk = int(self.chunk_seconds * self.sample_rate)
        all_logits: List[torch.Tensor] = []
        # Use non-overlapping chunks for simplicity; wav2vec2 receptive field is small.
        for start in range(0, T, chunk):
            end = min(T, start + chunk)
            seg = waveform[:, start:end].to(self.device)
            # processor expects 1D numpy; we pass tensor directly
            inputs = self.processor(seg.squeeze(0).cpu().numpy(),
                                    sampling_rate=self.sample_rate,
                                    return_tensors="pt")
            input_values = inputs.input_values.to(self.device)
            logits = self.model(input_values).logits  # [1, T_chunk_frames, V]
            all_logits.append(logits.squeeze(0).cpu())
        all_logits_tensor = torch.cat(all_logits, dim=0)  # [T_total_frames, V]
        log_probs = F.log_softmax(all_logits_tensor, dim=-1)
        return log_probs

    def align(self, waveform: torch.Tensor, words: List[str]) -> List[WordSpan]:
        """Return word-level timestamps."""
        norm = normalize_words(words)
        # Build the transcript as a single string with '|' separators (matches vocab).
        kept_indices = [i for i, w in enumerate(norm) if w]
        kept_words = [norm[i] for i in kept_indices]
        if not kept_words:
            return []
        transcript = " ".join(kept_words)
        # Encode characters via the tokenizer (operates char-by-char for wav2vec2).
        token_ids = self.processor.tokenizer(transcript,
                                             return_tensors="pt").input_ids[0]
        token_ids_list = token_ids.tolist()

        # Compute emissions
        log_probs = self._emissions(waveform)  # [T, V]
        T_frames = log_probs.shape[0]

        # forced_align expects [B, T, V] and targets [B, L]
        log_probs_b = log_probs.unsqueeze(0)
        targets_b = torch.tensor([token_ids_list], dtype=torch.int32)
        try:
            alignments, _scores = TF.forced_align(log_probs_b, targets_b, blank=self.pad_id)
        except Exception as e:
            print(f"  [align] forced_align error: {e}")
            return []
        # alignments: [1, T] of token *positions* in target (0..L-1) or blanks.
        # In torchaudio's forced_align, the returned values are token IDs from the target,
        # with blanks represented as `blank` (the pad id). To translate frame->target-position
        # we re-derive by walking and counting non-blank token transitions.

        frame_token_ids = alignments.squeeze(0).tolist()  # length T_frames

        # Compute frame duration in seconds.
        total_seconds = waveform.shape[1] / self.sample_rate
        frame_dur = total_seconds / T_frames

        # Derive target-position-per-frame from the forced-alignment path.
        # forced_align returns, per frame, either the blank id or the token id of the
        # target position currently being emitted. Repeats of the same non-blank id
        # in consecutive frames represent the same target position (a "held" emission).
        # When the non-blank id changes, we have transitioned to the *next* matching
        # position in the target sequence.
        target_pos_per_frame: List[int] = []
        ptr = -1  # index into token_ids_list of the current emission (-1 = none yet)
        prev_non_blank = None
        for tid in frame_token_ids:
            if tid == self.pad_id:
                target_pos_per_frame.append(ptr if ptr >= 0 else 0)
                prev_non_blank = None
                continue
            if prev_non_blank is not None and tid == prev_non_blank:
                # held emission, same target position
                target_pos_per_frame.append(ptr)
                continue
            # New emission: advance ptr to next target position whose token == tid
            next_ptr = ptr + 1
            while next_ptr < len(token_ids_list) and token_ids_list[next_ptr] != tid:
                next_ptr += 1
            if next_ptr < len(token_ids_list):
                ptr = next_ptr
            target_pos_per_frame.append(ptr if ptr >= 0 else 0)
            prev_non_blank = tid

        # Now group target positions into words using '|' delimiters.
        # Build map: target index -> word index
        word_idx_of_target: List[int] = []
        wi = 0
        for tid in token_ids_list:
            if tid == self.word_delim_id:
                word_idx_of_target.append(-1)  # delimiter, no word
                wi += 1
            else:
                word_idx_of_target.append(wi)

        # Per frame, derive word index from target position.
        # Blank frames are NOT included in word spans (they belong "between" emissions).
        spans: List[Tuple[int, int]] = [(-1, -1)] * len(kept_words)  # (start_frame, end_frame)
        for f, (pos, tid) in enumerate(zip(target_pos_per_frame, frame_token_ids)):
            if tid == self.pad_id:
                continue
            if pos < 0 or pos >= len(word_idx_of_target):
                continue
            w = word_idx_of_target[pos]
            if w < 0:
                continue
            s, e = spans[w]
            if s < 0:
                spans[w] = (f, f)
            else:
                spans[w] = (s, max(e, f))

        # Convert frame spans to seconds.
        out: List[WordSpan] = []
        for wi_keep, (s, e) in enumerate(spans):
            original_idx = kept_indices[wi_keep]
            word = words[original_idx]
            if s < 0:
                # Word never aligned — fill with NaN sentinels
                out.append(WordSpan(word=word, start_sec=float("nan"), end_sec=float("nan")))
            else:
                out.append(WordSpan(word=word, start_sec=s * frame_dur, end_sec=(e + 1) * frame_dur))
        return out


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------

@dataclass
class Metrics:
    n_words: int
    n_aligned: int
    MAE: float
    MedAE: float
    PCO_03: float
    PCO_02: float


def load_ground_truth(csv_path: str) -> List[Tuple[float, float]]:
    """Returns list of (word_start, word_end) from JamendoLyrics annotation csv."""
    out: List[Tuple[float, float]] = []
    with open(csv_path, "r") as f:
        header = f.readline()  # word_start,word_end,line_end
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            try:
                ws = float(parts[0])
                we = float(parts[1])
                out.append((ws, we))
            except ValueError:
                continue
    return out


def evaluate(predicted: List[WordSpan], gt: List[Tuple[float, float]]) -> Metrics:
    n = min(len(predicted), len(gt))
    errs: List[float] = []
    for i in range(n):
        ps = predicted[i].start_sec
        gs = gt[i][0]
        if math.isnan(ps):
            continue
        errs.append(abs(ps - gs))
    if not errs:
        return Metrics(n_words=n, n_aligned=0, MAE=float("nan"),
                       MedAE=float("nan"), PCO_03=0.0, PCO_02=0.0)
    mae = float(np.mean(errs))
    med = float(np.median(errs))
    pco3 = float(np.mean([1.0 if e <= 0.3 else 0.0 for e in errs]))
    pco2 = float(np.mean([1.0 if e <= 0.2 else 0.0 for e in errs]))
    return Metrics(n_words=n, n_aligned=len(errs), MAE=mae, MedAE=med, PCO_03=pco3, PCO_02=pco2)


# ----------------------------------------------------------------------------
# End-to-end per song
# ----------------------------------------------------------------------------

def run_on_song(basename: str,
                jamendo_root: str,
                out_dir: str,
                aligner: Wav2Vec2Aligner,
                use_demucs: bool = True,
                demucs_device: str = "cpu") -> dict:
    """Run the full pipeline on one song. Returns a dict with metrics."""
    mp3 = os.path.join(jamendo_root, "mp3", basename + ".mp3")
    words_txt = os.path.join(jamendo_root, "lyrics", basename + ".words.txt")
    gt_csv = os.path.join(jamendo_root, "annotations", "words", basename + ".csv")
    vocals_wav = os.path.join(out_dir, basename + "_vocals.wav")
    align_json = os.path.join(out_dir, basename + "_alignment.json")

    print(f"[{basename}]")

    # 1. Source separation (or pass-through)
    if use_demucs:
        separate_vocals(mp3, vocals_wav, device=demucs_device)
    else:
        wav, _ = load_audio(mp3, 16000)
        Path(vocals_wav).parent.mkdir(parents=True, exist_ok=True)
        sf.write(vocals_wav, wav.squeeze(0).cpu().numpy(), 16000)

    # 2. Load words
    with open(words_txt, "r") as f:
        words = [w.strip() for w in f if w.strip()]
    print(f"  {len(words)} words to align")

    # 3. Align
    waveform, _ = load_audio(vocals_wav, 16000)
    spans = aligner.align(waveform, words)

    # 4. Save alignment
    with open(align_json, "w") as f:
        json.dump([asdict(s) for s in spans], f, indent=2)

    # 5. Evaluate
    gt = load_ground_truth(gt_csv)
    metrics = evaluate(spans, gt)
    print(f"  MAE={metrics.MAE:.3f}s  MedAE={metrics.MedAE:.3f}s  "
          f"PCO0.3={metrics.PCO_03:.3f}  PCO0.2={metrics.PCO_02:.3f}  "
          f"({metrics.n_aligned}/{metrics.n_words} aligned)")
    return {"name": basename, **asdict(metrics)}
