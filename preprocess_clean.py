"""Non-destructive image enhancer: unsharp-mask sharpening then CLAHE on L channel.

Safe for downstream NNs — no colour shift, no clipping of fine detail.
"""
import argparse
import glob
import os

import cv2
import numpy as np


def sharpen(img: np.ndarray, amount: float = 0.6, radius: int = 1) -> np.ndarray:
    """Unsharp mask: original + amount*(original - blurred)."""
    blur = cv2.GaussianBlur(img, (2 * radius + 1, 2 * radius + 1), 0)
    sharp = cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)
    return sharp


def clahe_lab(img: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    """Apply CLAHE on the L channel of LAB — leaves hue and saturation untouched."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir",      required=True)
    ap.add_argument("--out_dir",     required=True)
    ap.add_argument("--sharpen_amt", type=float, default=0.6,
                    help="Unsharp-mask strength (0=off, 1=strong)")
    ap.add_argument("--sharpen_rad", type=int,   default=1,
                    help="Gaussian blur radius for unsharp mask")
    ap.add_argument("--clahe_clip",  type=float, default=2.0,
                    help="CLAHE clip limit (higher = more contrast)")
    ap.add_argument("--clahe_tile",  type=int,   default=8,
                    help="CLAHE tile grid size (NxN)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.in_dir, "*.png")))
    print(f"Processing {len(paths)} frames")

    for i, p in enumerate(paths):
        img = cv2.imread(p)
        img = sharpen(img, args.sharpen_amt, args.sharpen_rad)
        img = clahe_lab(img, args.clahe_clip, args.clahe_tile)
        cv2.imwrite(os.path.join(args.out_dir, os.path.basename(p)), img)
        if i % 100 == 0:
            print(f"  frame {i}/{len(paths)}")

    print(f"Done → {args.out_dir}")
