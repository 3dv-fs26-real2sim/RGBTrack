"""Temporal hand mask refiner — contour-anchored approach.

Frame 0: extract dense contour points of the hand mask.
Each frame: track those points via LK optical flow → reconstructed boundary.
Fill the tracked polygon → allowed region.
Anything in current_mask outside allowed region (+ dilation margin) is zeroed.

Blob replacement: detected blobs outside allowed zone are replaced with
the flow-predicted mask values at those pixels.
"""
import argparse
import glob
import os

import cv2
import numpy as np


def sample_contour_points(mask, n_points=200):
    """Sample n_points evenly from the largest contour of mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if len(c) <= 4:
        return None
    # Resample to n_points
    step = max(1, len(c) // n_points)
    pts  = c[::step, 0, :].astype(np.float32)
    return pts.reshape(-1, 1, 2)


def track_points(pts, prev_gray, curr_gray):
    """LK optical flow. Returns updated points (only good ones)."""
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray,
                                                    pts, None)
    good = status.ravel() == 1
    if good.sum() < 4:
        return pts  # fallback: keep old points
    return next_pts[good].reshape(-1, 1, 2)


def pts_to_mask(pts, h, w, dilation=20):
    """Fill convex hull of tracked points, dilated by `dilation` px."""
    hull  = cv2.convexHull(pts.astype(np.int32))
    m     = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(m, hull, 255)
    if dilation > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (dilation * 2 + 1,) * 2)
        m = cv2.dilate(m, k)
    return m


def warp_mask_by_flow(prev_mask, prev_gray, curr_gray, n_corners=200):
    corners = cv2.goodFeaturesToTrack(prev_gray, maxCorners=n_corners,
                                      qualityLevel=0.01, minDistance=5,
                                      mask=prev_mask)
    if corners is None or len(corners) < 4:
        return prev_mask.copy()
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray,
                                                    corners, None)
    good_p = corners[status.ravel() == 1]
    good_n = next_pts[status.ravel() == 1]
    if len(good_p) < 4:
        return prev_mask.copy()
    M, _ = cv2.estimateAffinePartial2D(good_p, good_n,
                                        method=cv2.RANSAC,
                                        ransacReprojThreshold=3)
    if M is None:
        return prev_mask.copy()
    h, w  = prev_mask.shape
    return cv2.warpAffine(prev_mask, M, (w, h),
                          flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask_dir",        required=True)
    ap.add_argument("--rgb_dir",         required=True)
    ap.add_argument("--out_dir",         required=True)
    ap.add_argument("--seed_frames",     type=int, default=5,
                    help="Number of early frames used to initialise contour")
    ap.add_argument("--dilation",        type=int, default=25,
                    help="Pixels to dilate the tracked polygon boundary")
    ap.add_argument("--min_blob_area",   type=int, default=150)
    ap.add_argument("--n_contour_pts",   type=int, default=300)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    mask_paths = sorted(glob.glob(os.path.join(args.mask_dir, "*.png")))
    rgb_paths  = sorted(glob.glob(os.path.join(args.rgb_dir,  "*.png")))
    assert len(mask_paths) == len(rgb_paths)
    N = len(mask_paths)
    print(f"Refining {N} frames  seed={args.seed_frames}  dilation={args.dilation}px")

    tracked_pts = None
    prev_gray   = None
    prev_mask   = None
    blobs_replaced = 0

    for i, (mp, rp) in enumerate(zip(mask_paths, rgb_paths)):
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8) * 255
        rgb  = cv2.imread(rp)
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

        if i < args.seed_frames:
            # Accumulate contour from first good masks
            pts = sample_contour_points(mask, args.n_contour_pts)
            if pts is not None:
                tracked_pts = pts if tracked_pts is None \
                    else np.concatenate([tracked_pts, pts], axis=0)
            cv2.imwrite(os.path.join(args.out_dir, os.path.basename(mp)), mask)
            prev_gray = gray
            prev_mask = mask
            continue

        h, w = mask.shape

        # Track contour points forward
        if tracked_pts is not None and prev_gray is not None:
            tracked_pts = track_points(tracked_pts, prev_gray, gray)
            allowed     = pts_to_mask(tracked_pts, h, w, args.dilation)
        else:
            allowed = np.ones((h, w), dtype=np.uint8) * 255

        # Flow-predicted mask for blob replacement
        predicted = warp_mask_by_flow(prev_mask, prev_gray, gray)

        # Find blobs: current pixels outside allowed zone
        outside  = cv2.bitwise_and(mask, cv2.bitwise_not(allowed))
        n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(
            outside, connectivity=8)

        blob_mask = np.zeros_like(mask)
        for lbl in range(1, n_lab):
            if stats[lbl, cv2.CC_STAT_AREA] >= args.min_blob_area:
                blob_mask[labels == lbl] = 255

        # Replace blob pixels with predicted
        refined = mask.copy()
        refined[blob_mask > 0] = predicted[blob_mask > 0]

        if blob_mask.any():
            blobs_replaced += 1

        cv2.imwrite(os.path.join(args.out_dir, os.path.basename(mp)), refined)

        prev_gray = gray
        prev_mask = refined

        if i % 100 == 0:
            print(f"  frame {i}/{N}  tracked_pts={len(tracked_pts) if tracked_pts is not None else 0}"
                  f"  blobs_replaced={blobs_replaced}")

    print(f"Done. Frames with blobs replaced: {blobs_replaced}")
    print(f"Output → {args.out_dir}")
