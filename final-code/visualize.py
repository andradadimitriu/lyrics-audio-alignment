"""
Visualize predicted vs ground-truth word onsets for one song.
Three-row plot (when VAD sidecar is present):
  bottom: ground-truth word boxes
  middle: predicted onsets
  top:    silero-vad voice-active intervals
For a chosen time window. Indices are matched per word (we plot only the words
whose ground-truth start lies inside the window — and use the same indices for
the predicted side, so apples-to-apples).

VAD overlay is read from `{out_dir}/{song}_vad.json` if present; otherwise the
top row is omitted (backwards-compatible with M2 outputs).
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def _load_vad(out_dir, song):
    p = os.path.join(out_dir, f"{song}_vad.json")
    if not os.path.exists(p):
        return None
    raw = json.load(open(p))
    return [(float(x["start"]), float(x["end"])) for x in raw]


def main(song, out_dir, jamendo_root, t_start, t_end):
    pred_path = os.path.join(out_dir, f"{song}_alignment.json")
    gt_path = os.path.join(jamendo_root, "annotations", "words", f"{song}.csv")
    pred = json.load(open(pred_path))
    gt = []
    with open(gt_path) as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                gt.append((float(parts[0]), float(parts[1])))
    vad = _load_vad(out_dir, song)

    # words whose GT onset lies in window
    indices = [i for i, (gs, _ge) in enumerate(gt[: len(pred)]) if t_start <= gs <= t_end]

    fig_h = 4.4 if vad else 3.6
    fig, ax = plt.subplots(figsize=(13, fig_h), dpi=130)

    if vad:
        for vs, ve in vad:
            if ve < t_start or vs > t_end:
                continue
            vs_c = max(vs, t_start)
            ve_c = min(ve, t_end)
            ax.add_patch(Rectangle((vs_c, 2.10), max(0.05, ve_c - vs_c), 0.35,
                                   facecolor="#9CA3AF", edgecolor="#4B5563",
                                   linewidth=0.5, alpha=0.75, zorder=1))

    for i in indices:
        gs, ge = gt[i]
        word = pred[i]["word"][:10] if i < len(pred) else "?"
        # Ground truth box, bottom row
        ax.add_patch(Rectangle((gs, -0.3), max(0.05, ge - gs), 0.6,
                               facecolor="#4A90D9", edgecolor="#1E2761", linewidth=0.6, zorder=2))
        if (ge - gs) > 0.18:
            ax.text((gs + ge) / 2, 0.0, word, ha="center", va="center",
                    fontsize=7.5, color="white", fontweight="bold", zorder=3)

        # Predicted onset, top row
        ps = pred[i]["start_sec"]
        if isinstance(ps, float) and ps == ps and t_start - 0.5 <= ps <= t_end + 0.5:
            err = ps - gs
            color = "#2E7D32" if abs(err) <= 0.3 else ("#B7791F" if abs(err) <= 0.6 else "#C9302C")
            ax.vlines(ps, 0.95, 1.55, colors=color, linewidth=2.2, zorder=2)
            ax.text(ps, 1.62, word, rotation=55, ha="left", va="bottom",
                    fontsize=7.5, color=color, zorder=3)
            # Connection line GT-start -> predicted
            ax.plot([gs, ps], [0.30, 0.95], color=color, linewidth=0.6,
                    alpha=0.55, zorder=1)

    if vad:
        ax.set_yticks([0, 1.25, 2.28])
        ax.set_yticklabels(["Ground truth", "Predicted onsets", "VAD"], fontsize=10)
        y_top = 2.95
    else:
        ax.set_yticks([0, 1.25])
        ax.set_yticklabels(["Ground truth", "Predicted onsets"], fontsize=10)
        y_top = 2.6
    ax.set_xlabel("time (s)", fontsize=10)
    ax.set_xlim(t_start, t_end)
    ax.set_ylim(-0.85, y_top)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.set_title(
        f"Word alignment — {song.replace('_-_', ' — ').replace('_', ' ')}  "
        f"({t_start:.0f}–{t_end:.0f} s, {len(indices)} words)",
        fontsize=11.5, color="#1E2761")

    # Legend
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="#2E7D32", lw=2.2, label="error ≤ 0.3 s"),
        Line2D([0], [0], color="#B7791F", lw=2.2, label="0.3 < err ≤ 0.6 s"),
        Line2D([0], [0], color="#C9302C", lw=2.2, label="error > 0.6 s")
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, frameon=False)

    fig.tight_layout()
    out_png = os.path.join(out_dir, "sample_alignment.png")
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"wrote {out_png}  ({len(indices)} words plotted)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--song", default="HILA_-_Give_Me_the_Same")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--jamendo", default="jamendolyrics")
    ap.add_argument("--start", type=float, default=20.0)
    ap.add_argument("--end", type=float, default=42.0)
    args = ap.parse_args()
    main(args.song, args.out, args.jamendo, args.start, args.end)
