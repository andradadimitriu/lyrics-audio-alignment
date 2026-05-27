"""Lyrics-to-audio alignment pipeline.

song -> HT Demucs (vocals stem) -> wav2vec 2.0 (CTC) -> forced_align
     -> word-level timestamps.

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
    if path.lower().endswith(".wav"):
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T)
    else:
        import librosa
        data, sr = librosa.load(path, sr=None, mono=False)
        if data.ndim == 1:
            waveform = torch.from_numpy(data).unsqueeze(0)
        else:
            waveform = torch.from_numpy(data)
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
        import librosa
        if mp3_path.lower().endswith(".wav"):
            data, sr = sf.read(mp3_path, dtype="float32", always_2d=True)
            wav = torch.from_numpy(data.T)
        else:
            data, sr = librosa.load(mp3_path, sr=None, mono=False)
            if data.ndim == 1:
                wav = torch.from_numpy(data).unsqueeze(0)
            else:
                wav = torch.from_numpy(data)
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

    def _prepare_targets(self, words: List[str]):
        norm = normalize_words(words)
        kept_indices = [i for i, w in enumerate(norm) if w]
        kept_words = [norm[i] for i in kept_indices]
        transcript = " ".join(kept_words)
        token_ids = self.processor.tokenizer(transcript, return_tensors="pt").input_ids[0]
        return kept_indices, kept_words, token_ids.tolist()

    def _path_to_spans(self, frame_token_ids: List[int], token_ids_list: List[int],
                       kept_words: List[str], kept_indices: List[int],
                       words: List[str], abs_frame_times: List[float],
                       frame_dur_default: float) -> List[WordSpan]:
        """Walk the forced-align path, build per-word frame spans, return WordSpans.
        abs_frame_times[i] is the absolute (original-timeline) seconds of kept frame i."""
        # Re-derive target-position-per-frame by walking the CTC path.
        target_pos_per_frame: List[int] = []
        ptr = -1
        prev_non_blank = None
        for tid in frame_token_ids:
            if tid == self.pad_id:
                target_pos_per_frame.append(ptr if ptr >= 0 else 0)
                prev_non_blank = None
                continue
            if prev_non_blank is not None and tid == prev_non_blank:
                target_pos_per_frame.append(ptr)
                continue
            next_ptr = ptr + 1
            while next_ptr < len(token_ids_list) and token_ids_list[next_ptr] != tid:
                next_ptr += 1
            if next_ptr < len(token_ids_list):
                ptr = next_ptr
            target_pos_per_frame.append(ptr if ptr >= 0 else 0)
            prev_non_blank = tid

        word_idx_of_target: List[int] = []
        wi = 0
        for tid in token_ids_list:
            if tid == self.word_delim_id:
                word_idx_of_target.append(-1)
                wi += 1
            else:
                word_idx_of_target.append(wi)

        spans: List[Tuple[int, int]] = [(-1, -1)] * len(kept_words)
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

        out: List[WordSpan] = []
        for wi_keep, (s, e) in enumerate(spans):
            original_idx = kept_indices[wi_keep]
            word = words[original_idx]
            if s < 0:
                out.append(WordSpan(word=word, start_sec=float("nan"), end_sec=float("nan")))
            else:
                # Frame s starts at abs_frame_times[s]; frame e ends at abs_frame_times[e]+frame_dur_default
                out.append(WordSpan(word=word,
                                    start_sec=abs_frame_times[s],
                                    end_sec=abs_frame_times[e] + frame_dur_default))
        return out

    def align(self, waveform: torch.Tensor, words: List[str]) -> List[WordSpan]:
        """Return word-level timestamps."""
        kept_indices, kept_words, token_ids_list = self._prepare_targets(words)
        if not kept_words:
            return []
        log_probs = self._emissions(waveform)
        T_frames = log_probs.shape[0]
        log_probs_b = log_probs.unsqueeze(0)
        targets_b = torch.tensor([token_ids_list], dtype=torch.int32)
        try:
            alignments, _scores = TF.forced_align(log_probs_b, targets_b, blank=self.pad_id)
        except Exception as e:
            print(f"  [align] forced_align error: {e}")
            return []
        frame_token_ids = alignments.squeeze(0).tolist()
        total_seconds = waveform.shape[1] / self.sample_rate
        frame_dur = total_seconds / T_frames
        abs_times = [f * frame_dur for f in range(T_frames)]
        return self._path_to_spans(frame_token_ids, token_ids_list, kept_words,
                                   kept_indices, words, abs_times, frame_dur)

    def align_with_vad(self, waveform: torch.Tensor, words: List[str],
                       vad_intervals) -> List[WordSpan]:
        """Forced-align over VAD-active frames only; spans in the original timeline.

        Diagnostic variant. Removing inter-vocal silences introduces frame-time
        discontinuities — prefer `align_with_boundary_window` in production.
        """
        from vad import forced_align_with_vad, FRAME_DUR_SEC
        kept_indices, kept_words, token_ids_list = self._prepare_targets(words)
        if not kept_words:
            return []
        log_probs = self._emissions(waveform)
        frame_token_ids, abs_frame_idx, _mask = forced_align_with_vad(
            log_probs, token_ids_list, self.pad_id, vad_intervals,
            frame_dur_sec=FRAME_DUR_SEC,
        )
        abs_times = [int(idx) * FRAME_DUR_SEC for idx in abs_frame_idx.tolist()]
        return self._path_to_spans(frame_token_ids, token_ids_list, kept_words,
                                   kept_indices, words, abs_times, FRAME_DUR_SEC)

    def align_with_boundary_window(self, waveform: torch.Tensor, words: List[str],
                                   t_start_sec: float, t_end_sec: float) -> List[WordSpan]:
        """Forced-align only on log-prob frames in [t_start_sec, t_end_sec].

        Emissions are computed on the full waveform (so timings inside the
        window match the no-VAD path exactly), then sliced before
        forced_align. Spans are returned in the original timeline.
        """
        kept_indices, kept_words, token_ids_list = self._prepare_targets(words)
        if not kept_words:
            return []
        # Emissions on the FULL song — same as no-VAD path.
        log_probs = self._emissions(waveform)
        T_total = log_probs.shape[0]
        total_seconds = waveform.shape[-1] / self.sample_rate
        frame_dur = total_seconds / T_total
        f_start = max(0, int(round(t_start_sec / frame_dur)))
        f_end = min(T_total, int(round(t_end_sec / frame_dur)))
        if f_end - f_start < 50:  # <1s of frames, fall back
            return self.align(waveform, words)
        log_probs_slice = log_probs[f_start:f_end]
        log_probs_b = log_probs_slice.unsqueeze(0)
        targets_b = torch.tensor([token_ids_list], dtype=torch.int32)
        try:
            alignments, _ = TF.forced_align(log_probs_b, targets_b, blank=self.pad_id)
        except Exception as ex:
            print(f"  [align_with_boundary_window] forced_align error: {ex}")
            return []
        frame_token_ids = alignments.squeeze(0).tolist()
        # Absolute time for kept-frame k = (f_start + k) * frame_dur.
        abs_times = [(f_start + f) * frame_dur for f in range(len(frame_token_ids))]
        return self._path_to_spans(frame_token_ids, token_ids_list, kept_words,
                                   kept_indices, words, abs_times, frame_dur)


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
