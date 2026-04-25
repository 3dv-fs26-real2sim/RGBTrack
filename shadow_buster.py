"""Shadow Buster — CLAHE + gamma preprocessing for SAM2 dark-region contrast.

Boosts local contrast in the Lightness channel only (LAB space) so the black
hat stays distinguishable from dark finger shadows. Leaves the yellow body
colour largely untouched since hue/chroma channels are not modified.

Usage:
    python shadow_buster.py --in_dir <scene>/rgb --out_dir <scene>/rgb_sb
    python shadow_buster.py --in_dir <scene>/rgb --out_dir <scene>/rgb_sb \
        --clip_limit 4.0 --gamma 1.3 --tile 8
"""
import argparse
import glob
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np


def build_gamma_table(gamma: float) -> np.ndarray:
    inv = 1.0 / gamma
    return (np.arange(256, dtype=np.float32) / 255.0) ** inv * 255.0


def process_frame(src: str, dst: str, clahe, gamma_table: np.ndarray) -> str:
    img = cv2.imread(src)
    if img is None:
        return f"SKIP {src}"

    # LAB: separate lightness from colour so we don't shift the yellow hue
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE on L only — pulls detail out of shadows without touching saturation
    l_clahe = clahe.apply(l)

    out = cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2BGR)

    # Gamma lift: brightens mid-tones/darks; 1.0 = no change
    out = cv2.LUT(out, gamma_table.astype(np.uint8))

    cv2.imwrite(dst, out)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir",     required=True,
                    help="Input RGB frame directory (*.png or *.jpg)")
    ap.add_argument("--out_dir",    required=True,
                    help="Output directory for processed frames")
    ap.add_argument("--clip_limit", type=float, default=3.0,
                    help="CLAHE clip limit — higher = more aggressive contrast (default 3.0)")
    ap.add_argument("--tile",       type=int,   default=8,
                    help="CLAHE tile grid size (default 8)")
    ap.add_argument("--gamma",      type=float, default=1.2,
                    help="Gamma for dark-lift, >1 brightens shadows (default 1.2)")
    ap.add_argument("--workers",    type=int,   default=8,
                    help="Parallel worker threads (default 8)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    frames = sorted(glob.glob(os.path.join(args.in_dir, "*.png")) +
                    glob.glob(os.path.join(args.in_dir, "*.jpg")))
    if not frames:
        print(f"No frames found in {args.in_dir}")
        return

    clahe       = cv2.createCLAHE(clipLimit=args.clip_limit,
                                   tileGridSize=(args.tile, args.tile))
    gamma_table = build_gamma_table(args.gamma)

    dsts = [os.path.join(args.out_dir, os.path.basename(f)) for f in frames]

    print(f"Processing {len(frames)} frames  "
          f"(clip={args.clip_limit} tile={args.tile} gamma={args.gamma}) ...")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(process_frame, s, d, clahe, gamma_table): s
                for s, d in zip(frames, dsts)}
        done = 0
        for fut in as_completed(futs):
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(frames)}")
        _ = [f.result() for f in futs]  # surface any exceptions

    print(f"Done → {args.out_dir}")


if __name__ == "__main__":
    main()
