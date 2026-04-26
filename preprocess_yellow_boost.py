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
    ap.add_argument("--duck_mask_dir", default=None,
                    help="If set, only boost yellow inside the duck bbox + bbox_pad")
    ap.add_argument("--bbox_pad",     type=int,   default=30,
                    help="Pixels of padding around the duck bbox")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.in_dir, "*.png")))
    print(f"Processing {len(paths)} frames")

    for i, p in enumerate(paths):
        img = cv2.imread(p)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        h_img, w_img = img.shape[:2]

        # Soft yellow mask — gaussian-blurred for feathered edges
        yellow_hard = ((hsv[:, :, 0] >= args.hue_lo) &
                       (hsv[:, :, 0] <= args.hue_hi)).astype(np.float32)
        yellow = cv2.GaussianBlur(yellow_hard, (15, 15), 0)

        # Restrict to duck bbox region if mask dir provided
        if args.duck_mask_dir:
            mp = os.path.join(args.duck_mask_dir, os.path.basename(p))
            duck = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if duck is not None and duck.any():
                ys, xs = np.where(duck > 127)
                x1 = max(0, xs.min() - args.bbox_pad)
                y1 = max(0, ys.min() - args.bbox_pad)
                x2 = min(w_img, xs.max() + args.bbox_pad)
                y2 = min(h_img, ys.max() + args.bbox_pad)
                bbox_mask = np.zeros_like(yellow)
                bbox_mask[y1:y2, x1:x2] = 1.0
                bbox_mask = cv2.GaussianBlur(bbox_mask, (31, 31), 0)
                yellow = yellow * bbox_mask

        # Blend boosted vs original gradually
        s_boosted = np.clip(hsv[:, :, 1] * args.sat_boost, 0, 255)
        v_boosted = np.clip(hsv[:, :, 2] * args.val_boost, 0, 255)
        hsv[:, :, 1] = hsv[:, :, 1] * (1 - yellow) + s_boosted * yellow
        hsv[:, :, 2] = hsv[:, :, 2] * (1 - yellow) + v_boosted * yellow

        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        cv2.imwrite(os.path.join(args.out_dir, os.path.basename(p)), out)

        if i % 100 == 0:
            print(f"  frame {i}/{len(paths)}")

    print(f"Done → {args.out_dir}")
