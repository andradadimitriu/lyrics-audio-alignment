"""
Run run_all.py's English JamendoLyrics sweep against every saved checkpoint
in checkpoints/, record aggregate MAE per checkpoint, and write the winner
to outputs/results_m3_finetuned.json (in the same schema as results_m3.json).

KEEPS results_m3.json (pretrained baseline) intact.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-glob", default="checkpoints/checkpoint_epoch_*.pt")
    ap.add_argument("--jamendo", default="jamendolyrics")
    ap.add_argument("--scratch-out", default="outputs_ft_eval",
                    help="per-checkpoint scratch outputs (vocals/alignment files)")
    ap.add_argument("--final-out", default="outputs/results_m3_finetuned.json")
    args = ap.parse_args()

    ckpts = sorted(glob.glob(args.ckpt_glob))
    if not ckpts:
        print(f"ERROR: no checkpoints matching {args.ckpt_glob}", file=sys.stderr)
        sys.exit(1)
    print(f"[eval_ckpts] {len(ckpts)} checkpoints found")

    per_ckpt_summary = []
    best = None  # (mae, ckpt, results_dict)
    for ckpt in ckpts:
        scratch = Path(args.scratch_out) / Path(ckpt).stem
        scratch.mkdir(parents=True, exist_ok=True)
        results_name = "results.json"
        cmd = [
            sys.executable, "run_all.py",
            "--jamendo", args.jamendo,
            "--out", str(scratch),
            "--no-multilingual",
            "--checkpoint", ckpt,
            "--results-out", results_name,
        ]
        print(f"\n[eval_ckpts] running on {ckpt} ...")
        t0 = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        dt = time.time() - t0
        if result.returncode != 0:
            print(f"  failed (exit {result.returncode}): {result.stderr[-400:]}")
            per_ckpt_summary.append({"ckpt": ckpt, "error": result.stderr[-400:]})
            continue
        results_path = scratch / results_name
        if not results_path.exists():
            print(f"  no results file produced")
            per_ckpt_summary.append({"ckpt": ckpt, "error": "no results file"})
            continue
        with open(results_path) as f:
            res = json.load(f)
        agg = res.get("english", {}).get("aggregate", {})
        mae = agg.get("MAE", float("inf"))
        print(f"  MAE={mae:.4f}  PCO0.3={agg.get('PCO_03',0):.3f}  "
              f"IntAcc0.5={agg.get('IntervalAcc_05',0):.3f}  ({dt:.1f}s)")
        per_ckpt_summary.append({
            "ckpt": ckpt, "wall_sec": round(dt, 1),
            "MAE": mae, "MedAE": agg.get("MedAE"),
            "PCO_03": agg.get("PCO_03"), "PCO_05": agg.get("PCO_05"),
            "IntervalAcc_05": agg.get("IntervalAcc_05"),
        })
        if best is None or mae < best[0]:
            best = (mae, ckpt, res, str(results_path))

    if best is None:
        print("[eval_ckpts] no checkpoint produced usable results")
        sys.exit(1)

    print("\n[eval_ckpts] summary:")
    for s in per_ckpt_summary:
        if "error" in s:
            print(f"  {s['ckpt']:40s}  ERROR")
        else:
            print(f"  {s['ckpt']:40s}  MAE={s['MAE']:.4f}  PCO0.3={s['PCO_03']:.3f}")
    print(f"\n[eval_ckpts] BEST: {best[1]}  MAE={best[0]:.4f}")

    # Pack the chosen results with the per-checkpoint summary alongside
    out = best[2]
    out["checkpoint_selected"] = best[1]
    out["per_checkpoint_summary"] = per_ckpt_summary
    Path(os.path.dirname(args.final_out)).mkdir(parents=True, exist_ok=True)
    with open(args.final_out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[eval_ckpts] wrote {args.final_out}")


if __name__ == "__main__":
    main()
