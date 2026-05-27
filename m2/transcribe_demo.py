"""
Quick CTC greedy-decoding demo: prints what wav2vec 2.0 thinks the lyrics are
for the first N seconds of a vocals stem. Useful as a visible artifact to show
that the SSL+CTC backbone is doing real work.
"""

import argparse
import os
import sys

import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from pipeline import get_device


def main(wav_path: str, seconds: float, out_txt: str | None):
    device = get_device()
    proc = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device).eval()

    wav, sr = torchaudio.load(wav_path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    wav = wav[:, : int(16000 * seconds)]

    inp = proc(wav.squeeze(0).numpy(), sampling_rate=16000, return_tensors="pt").input_values.to(device)
    with torch.inference_mode():
        ids = model(inp).logits.argmax(-1)
    text = proc.batch_decode(ids)[0]
    print(f"CTC greedy decode (first {seconds:.0f} s of {os.path.basename(wav_path)}):\n")
    print(text)
    if out_txt:
        with open(out_txt, "w") as f:
            f.write(text + "\n")
        print(f"\nwrote {out_txt}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default="outputs/HILA_-_Give_Me_the_Same_vocals.wav")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--out", default="outputs/ctc_transcription.txt")
    args = ap.parse_args()
    main(args.wav, args.seconds, args.out)
