"""Preprocess depth PNGs to amplify gradients (depth discontinuities).

Loads uint16 depth PNGs (mm), computes Sobel gradient magnitude,
amplifies and saves as uint8 edge map. Sudden depth jumps (object boundaries)
become very bright; flat surfaces become black.
"""
import argparse
import glob
import os

import cv2
import numpy as np

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth_dir", required=True)
    ap.add_argument("--out_dir",   required=True)
    ap.add_argument("--amplify",   type=float, default=5.0,
                    help="Multiply gradient magnitude by this factor before clipping")
    ap.add_argument("--blur",      type=int,   default=3,
                    help="Gaussian blur kernel before gradient (reduce noise)")
    ap.add_argument("--ksize",     type=int,   default=3,
                    help="Sobel kernel size")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.depth_dir, "*.png")))
    print(f"Processing {len(paths)} depth frames  amplify={args.amplify}x")

    for i, p in enumerate(paths):
        d = cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.float32)

        # Blur to reduce noise before gradient
        if args.blur > 1:
            d = cv2.GaussianBlur(d, (args.blur, args.blur), 0)

        # Sobel gradient magnitude
        gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=args.ksize)
        gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=args.ksize)
        mag = np.sqrt(gx**2 + gy**2)

        # Amplify and clip to uint8
        mag = mag * args.amplify
        mag = np.clip(mag, 0, 255).astype(np.uint8)

        cv2.imwrite(os.path.join(args.out_dir, os.path.basename(p)), mag)

        if i % 100 == 0:
            print(f"  frame {i}/{len(paths)}  max_grad={mag.max()}")

    print(f"Done → {args.out_dir}")
