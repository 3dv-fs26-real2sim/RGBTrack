"""Edge visualization: depth edges + duck contour tracked via optical flow.

Output: black background, fine white edges from:
  - Depth Sobel (all object boundaries)
  - Duck mask contour (sharp, tracked via LK optical flow frame-to-frame)
"""
import argparse
import glob
import os

import cv2
import numpy as np


def extract_contour_edges(mask, thickness=1):
    """Draw mask contour as white lines on black image."""
    out = np.zeros_like(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, contours, -1, 255, thickness)
    return out


def depth_edges(depth_path, table_cutoff, amplify=100.0):
    d = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if d is None:
        return None
    d = d.astype(np.float32) / 1000.0
    d[d > table_cutoff] = 0.0
    d[d < 0.01]         = 0.0
    gx  = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
    gy  = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2) * amplify
    return np.clip(mag, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir",   required=True)
    ap.add_argument("--mask_dir",    required=True,
                    help="Duck mask directory")
    ap.add_argument("--out_video",   required=True)
    ap.add_argument("--table_depth", type=float, default=None)
    ap.add_argument("--amplify",     type=float, default=100.0)
    ap.add_argument("--fps",         type=int,   default=50)
    args = ap.parse_args()

    depth_dir = os.path.join(args.scene_dir, "depth")
    rgb_dir   = os.path.join(args.scene_dir, "rgb")

    depth_paths = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
    mask_paths  = sorted(glob.glob(os.path.join(args.mask_dir, "*.png")))
    rgb_paths   = sorted(glob.glob(os.path.join(rgb_dir,   "*.png")))
    N = len(depth_paths)

    # Auto table cutoff
    if args.table_depth is None:
        print("Auto-detecting table cutoff...")
        depths = []
        for p in depth_paths[:50]:
            d = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if d is not None:
                depths.append(d.astype(np.float32) / 1000.0)
        valid = np.concatenate([d[d > 0.1].ravel() for d in depths])
        d_min, d_max = float(valid.min()), float(valid.max())
        norm = ((valid - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        t, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        table_cutoff = d_min + (t / 255.0) * (d_max - d_min)
        print(f"Table cutoff: {table_cutoff:.3f}m")
    else:
        table_cutoff = args.table_depth

    sample = cv2.imread(rgb_paths[0])
    h, w   = sample.shape[:2]
    out    = cv2.VideoWriter(args.out_video, cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (w, h))

    prev_gray       = None
    tracked_pts     = None   # duck contour points tracked via LK

    for i in range(N):
        rgb   = cv2.imread(rgb_paths[i])
        gray  = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        mask  = cv2.imread(mask_paths[i], cv2.IMREAD_GRAYSCALE)
        mask  = (mask > 127).astype(np.uint8) * 255

        # ── Depth edges ──────────────────────────────────────────────────
        d_edges = depth_edges(depth_paths[i], table_cutoff, args.amplify)
        if d_edges is None:
            d_edges = np.zeros((h, w), np.uint8)

        # ── Duck contour: track via LK or re-sample from mask ────────────
        if i == 0 or tracked_pts is None:
            # Sample contour points from mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_NONE)
            if contours:
                c    = max(contours, key=cv2.contourArea)
                step = max(1, len(c) // 500)
                tracked_pts = c[::step, 0, :].astype(np.float32).reshape(-1, 1, 2)
        elif prev_gray is not None and tracked_pts is not None:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, tracked_pts, None)
            good = status.ravel() == 1
            if good.sum() >= 4:
                tracked_pts = next_pts[good].reshape(-1, 1, 2)

        # Draw tracked duck contour as fine white lines
        duck_edges = np.zeros((h, w), np.uint8)
        if tracked_pts is not None and len(tracked_pts) >= 3:
            pts_int = tracked_pts.reshape(-1, 2).astype(np.int32)
            cv2.polylines(duck_edges, [pts_int], isClosed=True,
                          color=255, thickness=1)

        # ── Combine: depth edges + duck contour ──────────────────────────
        combined = np.maximum(d_edges, duck_edges)
        frame    = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
        out.write(frame)

        prev_gray = gray

        if i % 100 == 0:
            print(f"  frame {i}/{N}")

    out.release()
    print(f"Done → {args.out_video}")
