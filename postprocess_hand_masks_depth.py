"""Zero hand mask pixels at or behind duck depth.

For each frame: compute mean depth of duck mask region,
then zero any hand mask pixel with depth >= duck_depth.
"""
import argparse
import glob
import os

import cv2
import numpy as np

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand_mask_dir", required=True)
    ap.add_argument("--duck_mask_dir", required=True)
    ap.add_argument("--depth_dir",     required=True)
    ap.add_argument("--out_dir",       required=True)
    ap.add_argument("--depth_scale",   type=float, default=1.0)
    ap.add_argument("--duck_margin",   type=float, default=0.05,
                    help="Zero pixels this many metres in front of duck depth too")
    ap.add_argument("--min_depth",     type=float, default=0.2,
                    help="Also zero pixels closer than this (metres)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    hand_paths = sorted(glob.glob(os.path.join(args.hand_mask_dir, "*.png")))
    duck_paths = sorted(glob.glob(os.path.join(args.duck_mask_dir, "*.png")))
    N = len(hand_paths)
    print(f"Processing {N} frames")

    for i, (hp, dp) in enumerate(zip(hand_paths, duck_paths)):
        hand  = cv2.imread(hp, cv2.IMREAD_GRAYSCALE)
        hand  = (hand > 127).astype(np.uint8) * 255
        duck  = cv2.imread(dp, cv2.IMREAD_GRAYSCALE)
        duck  = (duck > 127).astype(np.uint8)

        name       = os.path.basename(hp)
        depth_path = os.path.join(args.depth_dir, name)
        depth      = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        if depth is not None and duck.any():
            depth_m   = depth.astype(np.float32) / 1000.0 * args.depth_scale
            valid_px  = duck & (depth_m > 0.01)
            if valid_px.any():
                duck_depth = float(depth_m[valid_px].mean())
                cutoff = duck_depth - args.duck_margin
                hand[(depth_m >= cutoff) & (depth_m > 0.01)] = 0

        # Zero pixels closer than min_depth
        if depth is not None and depth_m is not None:
            hand[(depth_m > 0.01) & (depth_m < args.min_depth)] = 0

        cv2.imwrite(os.path.join(args.out_dir, name), hand)

        if i % 100 == 0:
            print(f"  frame {i}/{N}  duck_depth={duck_depth:.3f}  cutoff={cutoff:.3f}" if depth is not None else f"  frame {i}/{N}")

    print(f"Done → {args.out_dir}")
