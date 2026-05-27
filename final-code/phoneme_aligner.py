"""
Inference path for the M3 fine-tuned phoneme-head model.

Loads a checkpoint produced by `train.py`:
    {"encoder_state", "head_state", "vocab", "base_id"}

For each input word the lyrics are phonemized via g2p_en (same ARPAbet vocab as
the DALI annotations used during training). forced_align runs over the
concatenated phoneme target sequence; per-word spans come from a parallel
word-index-per-token array (no '|' delimiter — phoneme adjacency is implicit).

API mirrors `pipeline.Wav2Vec2Aligner`:
    .align(waveform, words)
    .align_with_boundary_window(waveform, words, t_start, t_end)
    ._emissions(waveform)
    .pad_id, .id_to_token, .word_delim_id (None — kept for greedy-decode compat)
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as TF

from pipeline import WordSpan, get_device


class PhonemeAligner:
    def __init__(self, checkpoint_path: str,
                 base_id: Optional[str] = None,
                 device: Optional[str] = None,
                 chunk_seconds: float = 30.0):
        from transformers import Wav2Vec2Model
        from labels import get_g2p
        self.device = device or get_device()
        print(f"  [phoneme-aligner] loading {checkpoint_path} on {self.device} ...")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.vocab = ckpt["vocab"]
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.pad_id = self.vocab.get("<blank>", 0)
        self.word_delim_id = None  # phoneme path: no delimiter, words are
                                   # demarcated by a parallel array
        base = base_id or ckpt.get("base_id", "facebook/wav2vec2-base-960h")
        self.encoder = Wav2Vec2Model.from_pretrained(base)
        self.encoder.load_state_dict(ckpt["encoder_state"])
        H = self.encoder.config.hidden_size
        V = len(self.vocab)
        self.head = nn.Linear(H, V)
        self.head.load_state_dict(ckpt["head_state"])
        self.encoder.to(self.device).eval()
        self.head.to(self.device).eval()
        self.sample_rate = 16000
        self.chunk_seconds = chunk_seconds
        self.g2p = get_g2p()

    # ------------------------------------------------------------------
    # Lyrics -> phoneme target
    # ------------------------------------------------------------------

    def _words_to_phoneme_target(self, words: Sequence[str]
                                 ) -> Tuple[List[int], List[int], List[int]]:
        """Returns (token_ids, word_idx_per_token, kept_word_indices)."""
        from labels import word_to_phonemes
        token_ids, word_idx_per_token, kept = [], [], []
        for orig_i, w in enumerate(words):
            phons = word_to_phonemes(w, self.g2p)
            ids = [self.vocab[p] for p in phons if p in self.vocab]
            if not ids:
                continue
            kept.append(orig_i)
            for tid in ids:
                token_ids.append(tid)
                word_idx_per_token.append(len(kept) - 1)
        return token_ids, word_idx_per_token, kept

    # ------------------------------------------------------------------
    # Emissions
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _emissions(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform: [1, T] or [T]. Returns log_probs [T_frames, V] on CPU."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        T = waveform.shape[-1]
        chunk = int(self.chunk_seconds * self.sample_rate)
        all_logits = []
        for start in range(0, T, chunk):
            end = min(T, start + chunk)
            seg = waveform[..., start:end].to(self.device)
            m = seg.mean()
            s = seg.std().clamp_min(1e-7)
            seg_n = (seg - m) / s
            hidden = self.encoder(seg_n).last_hidden_state  # [1, F, H]
            logits = self.head(hidden).squeeze(0).cpu()  # [F, V]
            all_logits.append(logits)
        all_logits_t = torch.cat(all_logits, dim=0)
        return F.log_softmax(all_logits_t, dim=-1)

    # ------------------------------------------------------------------
    # Path -> word spans
    # ------------------------------------------------------------------

    def _path_to_spans(self, frame_token_ids: List[int],
                       token_ids_list: List[int],
                       word_idx_per_token: List[int],
                       kept_indices: List[int], words: Sequence[str],
                       abs_frame_times: List[float],
                       frame_dur_default: float) -> List[WordSpan]:
        # Walk the CTC path; identical algorithm to pipeline.Wav2Vec2Aligner.
        target_pos_per_frame = []
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

        spans: List[Tuple[int, int]] = [(-1, -1)] * len(kept_indices)
        for f, (pos, tid) in enumerate(zip(target_pos_per_frame, frame_token_ids)):
            if tid == self.pad_id:
                continue
            if pos < 0 or pos >= len(word_idx_per_token):
                continue
            w = word_idx_per_token[pos]
            s, e = spans[w]
            if s < 0:
                spans[w] = (f, f)
            else:
                spans[w] = (s, max(e, f))

        out: List[WordSpan] = []
        for wi_keep, (s, e) in enumerate(spans):
            original = kept_indices[wi_keep]
            word = words[original]
            if s < 0:
                out.append(WordSpan(word=word, start_sec=float("nan"),
                                    end_sec=float("nan")))
            else:
                out.append(WordSpan(word=word,
                                    start_sec=abs_frame_times[s],
                                    end_sec=abs_frame_times[e] + frame_dur_default))
        return out

    # ------------------------------------------------------------------
    # Public align methods
    # ------------------------------------------------------------------

    def align(self, waveform: torch.Tensor, words: Sequence[str]) -> List[WordSpan]:
        token_ids, word_idx, kept = self._words_to_phoneme_target(words)
        if not token_ids:
            return []
        log_probs = self._emissions(waveform)
        T = log_probs.shape[0]
        log_probs_b = log_probs.unsqueeze(0)
        targets_b = torch.tensor([token_ids], dtype=torch.int32)
        try:
            alignments, _ = TF.forced_align(log_probs_b, targets_b, blank=self.pad_id)
        except Exception as e:
            print(f"  [phoneme-aligner] forced_align error: {e}")
            return []
        frame_token_ids = alignments.squeeze(0).tolist()
        total_seconds = waveform.shape[-1] / self.sample_rate
        frame_dur = total_seconds / T
        abs_times = [f * frame_dur for f in range(T)]
        return self._path_to_spans(frame_token_ids, token_ids, word_idx,
                                   kept, words, abs_times, frame_dur)

    def align_with_boundary_window(self, waveform: torch.Tensor,
                                   words: Sequence[str],
                                   t_start_sec: float, t_end_sec: float
                                   ) -> List[WordSpan]:
        token_ids, word_idx, kept = self._words_to_phoneme_target(words)
        if not token_ids:
            return []
        log_probs = self._emissions(waveform)
        T_total = log_probs.shape[0]
        total_seconds = waveform.shape[-1] / self.sample_rate
        frame_dur = total_seconds / T_total
        f_start = max(0, int(round(t_start_sec / frame_dur)))
        f_end = min(T_total, int(round(t_end_sec / frame_dur)))
        if f_end - f_start < 50:
            return self.align(waveform, words)
        log_probs_slice = log_probs[f_start:f_end]
        log_probs_b = log_probs_slice.unsqueeze(0)
        targets_b = torch.tensor([token_ids], dtype=torch.int32)
        try:
            alignments, _ = TF.forced_align(log_probs_b, targets_b, blank=self.pad_id)
        except Exception as e:
            print(f"  [phoneme-aligner] forced_align error: {e}")
            return []
        frame_token_ids = alignments.squeeze(0).tolist()
        abs_times = [(f_start + f) * frame_dur for f in range(len(frame_token_ids))]
        return self._path_to_spans(frame_token_ids, token_ids, word_idx,
                                   kept, words, abs_times, frame_dur)
