"""Track frame-0 hand contour via optical flow. Zero everything outside.

Frame 0: sample contour points from hand mask.
Each frame: track those points via LK optical flow.
Fill the tracked polygon -> allowed region (+ 1px dilation).
current_mask & allowed_region -> output.
"""
import argparse
import glob
import os

import cv2
import numpy as np


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask_dir",      required=True)
    ap.add_argument("--rgb_dir",       required=True)
    ap.add_argument("--out_dir",       required=True)
    ap.add_argument("--dilation",      type=int, default=1)
    ap.add_argument("--n_contour_pts", type=int, default=400)
    ap.add_argument("--min_blob_area", type=int, default=150)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    mask_paths = sorted(glob.glob(os.path.join(args.mask_dir, "*.png")))
    rgb_paths  = sorted(glob.glob(os.path.join(args.rgb_dir,  "*.png")))
    assert len(mask_paths) == len(rgb_paths)
    N = len(mask_paths)
    print(f"Refining {N} frames")

    tracked_pts = None
    prev_gray   = None

    for i, (mp, rp) in enumerate(zip(mask_paths, rgb_paths)):
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8) * 255
        rgb  = cv2.imread(rp)
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

        if i == 0:
            # Sample contour from frame 0
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_NONE)
            if contours:
                c    = max(contours, key=cv2.contourArea)
                step = max(1, len(c) // args.n_contour_pts)
                tracked_pts = c[::step, 0, :].astype(np.float32).reshape(-1, 1, 2)
            cv2.imwrite(os.path.join(args.out_dir, os.path.basename(mp)), mask)
            prev_gray = gray
            continue

        # Track contour points via LK
        if tracked_pts is not None and prev_gray is not None:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, tracked_pts, None)
            good = status.ravel() == 1
            if good.sum() >= 4:
                tracked_pts = next_pts[good].reshape(-1, 1, 2)

        # Build allowed region from tracked polygon
        h, w = mask.shape
        allowed = np.zeros((h, w), np.uint8)
        if tracked_pts is not None and len(tracked_pts) >= 3:
            hull = cv2.convexHull(tracked_pts.astype(np.int32))
            cv2.fillConvexPoly(allowed, hull, 255)
            if args.dilation > 0:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (args.dilation*2+1,)*2)
                allowed = cv2.dilate(allowed, k)

        refined = cv2.bitwise_and(mask, allowed)
        cv2.imwrite(os.path.join(args.out_dir, os.path.basename(mp)), refined)

        prev_gray = gray

        if i % 100 == 0:
            print(f"  frame {i}/{N}  pts={len(tracked_pts) if tracked_pts is not None else 0}")

    print(f"Done → {args.out_dir}")
