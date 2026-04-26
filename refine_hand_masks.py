"""Temporal hand mask refiner using optical flow prediction.

Algorithm per frame:
  1. Warp previous mask forward using sparse LK optical flow on mask corners
     -> predicted_mask
  2. Find NEW pixels: current_mask & ~prev_mask
  3. Label new-pixel connected components
  4. A component is "legitimate growth" if it touches the dilated border of
     the predicted mask (border-attached expansion of the hand)
  5. A component is a "blob" if it is disconnected from the predicted mask
     border (sudden isolated patch = duck / surface noise)
  6. Replace blob pixels in current_mask with predicted_mask pixels
     (i.e. those pixels get the prediction instead of the DINO noisy mask)
  7. Save cleaned mask
"""
import argparse
import glob
import os

import cv2
import numpy as np


# ── Optical flow warp ────────────────────────────────────────────────────────
def warp_mask_by_flow(prev_mask, prev_frame, curr_frame, n_corners=200):
    """Estimate motion from prev→curr via LK optical flow on prev_mask corners.
    Returns warped prev_mask as uint8 binary (0/255)."""
    corners = cv2.goodFeaturesToTrack(
        cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if prev_frame.ndim == 3
        else prev_frame,
        maxCorners=n_corners, qualityLevel=0.01, minDistance=5,
        mask=prev_mask,
    )
    if corners is None or len(corners) < 4:
        return prev_mask.copy()

    prev_gray = (cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                 if prev_frame.ndim == 3 else prev_frame)
    curr_gray = (cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
                 if curr_frame.ndim == 3 else curr_frame)

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray,
                                                    corners, None)
    good_prev = corners[status.ravel() == 1]
    good_next = next_pts[status.ravel() == 1]

    if len(good_prev) < 4:
        return prev_mask.copy()

    M, _ = cv2.estimateAffinePartial2D(good_prev, good_next,
                                        method=cv2.RANSAC, ransacReprojThreshold=3)
    if M is None:
        return prev_mask.copy()

    h, w = prev_mask.shape
    warped = cv2.warpAffine(prev_mask, M, (w, h),
                            flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped


# ── Blob rejection ────────────────────────────────────────────────────────────
def reject_overgrowth(current_mask, predicted_mask, prev_mask,
                      max_growth_px=15, min_blob_area=150):
    """
      1. Union with prev_mask — pixels that existed before NEVER disappear.
      2. Find blobs: new pixels outside growth zone with area >= min_blob_area.
      3. Replace ONLY blob pixels with predicted values — leave everything else.
    """
    # Never lose previous pixels
    base = cv2.bitwise_or(current_mask, prev_mask)

    # Allowed growth zone around predicted
    dist_from_pred = cv2.distanceTransform(
        cv2.bitwise_not(predicted_mask), cv2.DIST_L2, 5)
    allowed = (dist_from_pred <= max_growth_px).astype(np.uint8) * 255

    # New pixels (relative to predicted) outside allowed zone
    new_px  = cv2.bitwise_and(base, cv2.bitwise_not(predicted_mask))
    bad_new = cv2.bitwise_and(new_px, cv2.bitwise_not(allowed))

    n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(bad_new,
                                                                 connectivity=8)
    blob_mask = np.zeros_like(current_mask)
    for lbl in range(1, n_lab):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_blob_area:
            blob_mask[labels == lbl] = 255

    # Replace only blob pixels with predicted
    refined = base.copy()
    refined[blob_mask > 0] = predicted_mask[blob_mask > 0]

    return refined, bool(blob_mask.any())


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask_dir",  required=True,
                    help="Input hand masks directory")
    ap.add_argument("--rgb_dir",   required=True,
                    help="RGB frames directory (for optical flow)")
    ap.add_argument("--out_dir",   required=True)
    ap.add_argument("--border_dilation", type=int, default=15,
                    help="Max pixels a mask edge can grow from predicted per frame")
    ap.add_argument("--min_blob_area",   type=int, default=150,
                    help="Ignore new-pixel components smaller than this (noise)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    mask_paths = sorted(glob.glob(os.path.join(args.mask_dir, "*.png")))
    rgb_paths  = sorted(glob.glob(os.path.join(args.rgb_dir,  "*.png")))

    assert len(mask_paths) == len(rgb_paths), \
        f"Mask count ({len(mask_paths)}) != RGB count ({len(rgb_paths)})"
    N = len(mask_paths)
    print(f"Refining {N} frames...")

    total_blobs = 0
    for i, (mp, rp) in enumerate(zip(mask_paths, rgb_paths)):
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8) * 255
        rgb  = cv2.imread(rp)

        if i == 0:
            prev_mask = mask.copy()
            prev_rgb  = rgb.copy()
            cv2.imwrite(os.path.join(args.out_dir, os.path.basename(mp)), mask)
            continue

        predicted = warp_mask_by_flow(prev_mask, prev_rgb, rgb)
        refined, blob_found = reject_overgrowth(mask, predicted, prev_mask,
                                                args.border_dilation,
                                                args.min_blob_area)
        n_blobs = 1 if blob_found else 0
        total_blobs += n_blobs

        cv2.imwrite(os.path.join(args.out_dir, os.path.basename(mp)), refined)

        prev_mask = refined.copy()
        prev_rgb  = rgb.copy()

        if i % 100 == 0:
            print(f"  frame {i}/{N}  blob_px_replaced={n_blobs}")

    print(f"Done. Total blob pixels replaced: {total_blobs}")
    print(f"Output → {args.out_dir}")
