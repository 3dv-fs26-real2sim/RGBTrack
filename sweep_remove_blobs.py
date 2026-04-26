"""Sweep through mask sequence and remove sudden large blobs.

For each frame: find pixels that were black in prev frame but white now.
If those new pixels form a connected component >= min_blob_area, zero it out.
"""
import argparse
import glob
import os

import cv2
import numpy as np

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir",        required=True)
    ap.add_argument("--out_dir",       required=True)
    ap.add_argument("--min_blob_area", type=int, default=300)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.in_dir, "*.png")))
    print(f"Sweeping {len(paths)} frames  min_blob_area={args.min_blob_area}")

    prev = None
    removed = 0

    for i, p in enumerate(paths):
        mask = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8) * 255

        if prev is None:
            cv2.imwrite(os.path.join(args.out_dir, os.path.basename(p)), mask)
            prev = mask
            continue

        # New pixels
        new_px = cv2.bitwise_and(mask, cv2.bitwise_not(prev))

        # Find large blobs in new pixels
        n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(
            new_px, connectivity=8)

        refined = mask.copy()
        for lbl in range(1, n_lab):
            if stats[lbl, cv2.CC_STAT_AREA] >= args.min_blob_area:
                refined[labels == lbl] = 0
                removed += 1

        cv2.imwrite(os.path.join(args.out_dir, os.path.basename(p)), refined)
        prev = refined

        if i % 100 == 0:
            print(f"  frame {i}/{len(paths)}  blobs_removed={removed}")

    print(f"Done. Total blobs removed: {removed}")
    print(f"Output → {args.out_dir}")
