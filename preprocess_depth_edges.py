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
    ap.add_argument("--depth_dir",   required=True)
    ap.add_argument("--out_dir",     required=True)
    ap.add_argument("--amplify",     type=float, default=100.0)
    ap.add_argument("--blur",        type=int,   default=0)
    ap.add_argument("--ksize",       type=int,   default=3)
    ap.add_argument("--table_depth", type=float, default=None,
                    help="Hard cutoff in metres. If not set, auto-detected via Otsu on first 50 frames.")
    ap.add_argument("--depth_scale", type=float, default=1.0)
    ap.add_argument("--seed_frames", type=int,   default=50)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.depth_dir, "*.png")))
    print(f"Processing {len(paths)} depth frames  amplify={args.amplify}x")

    # Auto-detect table cutoff from seed frames
    if args.table_depth is None:
        print(f"Auto-detecting table cutoff from first {args.seed_frames} frames...")
        depths = []
        for p in paths[:args.seed_frames]:
            d = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if d is not None:
                depths.append(d.astype(np.float32) / 1000.0 * args.depth_scale)
        valid = np.concatenate([d[d > 0.1].ravel() for d in depths])
        d_min, d_max = float(valid.min()), float(valid.max())
        norm = ((valid - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        t, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        table_cutoff = d_min + (t / 255.0) * (d_max - d_min)
        print(f"Auto table cutoff: {table_cutoff:.3f}m")
    else:
        table_cutoff = args.table_depth
        print(f"Using table cutoff: {table_cutoff:.3f}m")

    prev_gray    = None
    prev_edges   = None

    for i, p in enumerate(paths):
        d = cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0 * args.depth_scale
        d[d > table_cutoff] = 0.0
        d[d < 0.01]         = 0.0

        if args.blur > 1:
            d = cv2.GaussianBlur(d, (args.blur, args.blur), 0)

        gx  = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=args.ksize)
        gy  = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=args.ksize)
        mag = np.sqrt(gx**2 + gy**2) * args.amplify
        mag = np.clip(mag, 0, 255).astype(np.float32)

        # Load RGB for optical flow
        rgb_path = os.path.join(args.depth_dir.replace("depth", "rgb"),
                                os.path.basename(p))
        rgb = cv2.imread(rgb_path)
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY) if rgb is not None \
               else np.zeros(mag.shape, np.uint8)

        # Use flow prediction as spatial amplifier — only boosts current frame signal
        if prev_edges is not None and prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            h, w = mag.shape
            map_x = (np.arange(w) + flow[:, :, 0]).astype(np.float32)
            map_y = (np.arange(h).reshape(-1, 1) + flow[:, :, 1]).astype(np.float32)
            predicted = cv2.remap(prev_edges, map_x, map_y,
                                  interpolation=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            # Where prediction is strong, amplify current edges (2x boost)
            # Output is purely current frame — prediction only weights it
            weight = 1.0 + (predicted / 255.0)
            mag    = mag * weight

        mag_u8 = np.clip(mag, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(args.out_dir, os.path.basename(p)), mag_u8)

        prev_gray  = gray
        prev_edges = mag

        if i % 100 == 0:
            print(f"  frame {i}/{len(paths)}  max_grad={mag_u8.max()}")

    print(f"Done → {args.out_dir}")
