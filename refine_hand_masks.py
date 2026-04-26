"""Temporal post-processing for hand masks.

Strategy:
  - For each frame, decompose mask into connected components.
  - Track the hand component (centroid + velocity).
  - If a frame contains multiple disconnected blobs OR a sudden area jump,
    keep only the component whose centroid is closest to the predicted
    position (prev centroid + velocity). Reject all others — those are
    typically the duck, a false GroundingDINO detection, etc.
"""
import argparse
import glob
import os

import cv2
import numpy as np


def predicted_centroid(prev, vel):
    if prev is None:
        return None
    return prev + vel


def pick_component(mask, predicted, max_jump_px):
    n_lab, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_lab <= 1:
        return mask, None

    best_lbl  = -1
    best_dist = float("inf")
    for lbl in range(1, n_lab):
        c = centroids[lbl]
        if predicted is None:
            score = -stats[lbl, cv2.CC_STAT_AREA]  # first frame: pick largest
        else:
            score = np.linalg.norm(c - predicted)
            if score > max_jump_px:
                continue
        if score < best_dist:
            best_dist = score
            best_lbl  = lbl

    if best_lbl < 0:
        # No valid component close to prediction — return empty
        return np.zeros_like(mask), None

    out = (labels == best_lbl).astype(np.uint8) * 255
    return out, centroids[best_lbl]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir",  required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_jump_px",   type=float, default=80.0,
                    help="Max distance from predicted centroid to accept a component")
    ap.add_argument("--max_area_ratio", type=float, default=2.5,
                    help="Reject mask entirely if area > prev_area * this ratio")
    ap.add_argument("--velocity_alpha", type=float, default=0.6,
                    help="EMA factor for velocity smoothing")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.in_dir, "*.png")))
    print(f"Found {len(paths)} masks")

    prev_c   = None
    velocity = np.zeros(2)
    prev_area = None
    rejected_frames = 0
    cleaned_frames  = 0

    for i, p in enumerate(paths):
        mask = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8) * 255

        if not mask.any():
            cv2.imwrite(os.path.join(args.out_dir, os.path.basename(p)),
                        np.zeros_like(mask))
            continue

        predicted = predicted_centroid(prev_c, velocity)
        cleaned, c = pick_component(mask, predicted, args.max_jump_px)

        # Reject area explosions (e.g. mask suddenly fills the duck too)
        if prev_area is not None and cleaned.any():
            new_area = float(cleaned.sum() / 255)
            if new_area > prev_area * args.max_area_ratio:
                cleaned = np.zeros_like(mask)
                c = None
                rejected_frames += 1

        cv2.imwrite(os.path.join(args.out_dir, os.path.basename(p)), cleaned)

        if c is not None:
            if prev_c is not None:
                v = c - prev_c
                velocity = args.velocity_alpha * v + (1 - args.velocity_alpha) * velocity
            prev_c    = c
            prev_area = float(cleaned.sum() / 255)
            cleaned_frames += 1

        if i % 100 == 0:
            print(f"  frame {i}/{len(paths)}  prev_c={prev_c}  vel={velocity}")

    print(f"Done. Cleaned: {cleaned_frames}  Rejected (area explosion): {rejected_frames}")
    print(f"Output → {args.out_dir}")
