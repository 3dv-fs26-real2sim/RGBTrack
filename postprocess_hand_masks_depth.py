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
    ap.add_argument("--duck_mask_dir", default=None,
                    help="Unused — duck depth comes from pose txt")
    ap.add_argument("--pose_dir",      required=True,
                    help="ob_in_cam directory with per-frame pose .txt files")
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
    N = len(hand_paths)
    print(f"Processing {N} frames")

    for i, hp in enumerate(hand_paths):
        hand = cv2.imread(hp, cv2.IMREAD_GRAYSCALE)
        hand = (hand > 127).astype(np.uint8) * 255

        name       = os.path.basename(hp)
        stem       = os.path.splitext(name)[0]
        depth_path = os.path.join(args.depth_dir, name)
        pose_path  = os.path.join(args.pose_dir, f"{stem}.txt")
        depth      = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        if depth is not None and os.path.exists(pose_path):
            depth_m    = depth.astype(np.float32) / 1000.0 * args.depth_scale
            duck_depth = float(np.loadtxt(pose_path).reshape(4, 4)[2, 3])
            cutoff     = duck_depth - args.duck_margin
            hand[(depth_m >= cutoff) & (depth_m > 0.01)] = 0
            hand[(depth_m > 0.01) & (depth_m < args.min_depth)] = 0

        cv2.imwrite(os.path.join(args.out_dir, name), hand)

        if i % 100 == 0:
            dc = f"{duck_depth:.3f}" if depth is not None and os.path.exists(pose_path) else "n/a"
            print(f"  frame {i}/{N}  duck_z={dc}")

    print(f"Done → {args.out_dir}")
