"""Boost yellow hue in RGB frames to make duck stand out.

Converts to HSV, amplifies saturation for yellow hue range (H=15-35),
converts back. Everything else unchanged.
"""
import argparse
import glob
import os

import cv2
import numpy as np

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir",       required=True)
    ap.add_argument("--out_dir",      required=True)
    ap.add_argument("--hue_lo",       type=int,   default=15,
                    help="Lower yellow hue bound (OpenCV: 0-180)")
    ap.add_argument("--hue_hi",       type=int,   default=35,
                    help="Upper yellow hue bound")
    ap.add_argument("--sat_boost",    type=float, default=2.0,
                    help="Multiply saturation in yellow range by this factor")
    ap.add_argument("--val_boost",    type=float, default=1.2,
                    help="Multiply value in yellow range by this factor")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.in_dir, "*.png")))
    print(f"Processing {len(paths)} frames")

    for i, p in enumerate(paths):
        img = cv2.imread(p)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Yellow mask
        yellow = (hsv[:, :, 0] >= args.hue_lo) & (hsv[:, :, 0] <= args.hue_hi)

        # Boost saturation and value in yellow region
        hsv[:, :, 1][yellow] = np.clip(hsv[:, :, 1][yellow] * args.sat_boost, 0, 255)
        hsv[:, :, 2][yellow] = np.clip(hsv[:, :, 2][yellow] * args.val_boost, 0, 255)

        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        cv2.imwrite(os.path.join(args.out_dir, os.path.basename(p)), out)

        if i % 100 == 0:
            print(f"  frame {i}/{len(paths)}")

    print(f"Done → {args.out_dir}")
