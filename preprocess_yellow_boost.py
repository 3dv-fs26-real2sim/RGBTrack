"""Replace yellow pixels with fluorescent pink to make duck stand out.

Within the duck bbox: find pixels in yellow hue range with high saturation,
replace with fluorescent pink (BGR=(255, 0, 255)).
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
    ap.add_argument("--bbox_pad",     type=int,   default=10,
                    help="Pixels of padding around the duck bbox")
    ap.add_argument("--sat_min",      type=int,   default=100,
                    help="Min saturation to count as 'yellow' pixel")
    ap.add_argument("--val_min",      type=int,   default=80,
                    help="Min value to count as 'yellow' pixel")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.in_dir, "*.png")))
    print(f"Processing {len(paths)} frames")

    PINK_BGR = np.array([255, 0, 255], dtype=np.uint8)

    for i, p in enumerate(paths):
        img = cv2.imread(p)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_img, w_img = img.shape[:2]

        # Bbox restriction
        bbox_mask = np.ones((h_img, w_img), dtype=bool)
        if args.duck_mask_dir:
            mp = os.path.join(args.duck_mask_dir, os.path.basename(p))
            duck = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if duck is not None and duck.any():
                ys, xs = np.where(duck > 127)
                x1 = max(0, xs.min() - args.bbox_pad)
                y1 = max(0, ys.min() - args.bbox_pad)
                x2 = min(w_img, xs.max() + args.bbox_pad)
                y2 = min(h_img, ys.max() + args.bbox_pad)
                bbox_mask = np.zeros((h_img, w_img), dtype=bool)
                bbox_mask[y1:y2, x1:x2] = True

        # Yellow pixels: hue in range AND saturated AND bright
        yellow = ((hsv[:, :, 0] >= args.hue_lo) &
                  (hsv[:, :, 0] <= args.hue_hi) &
                  (hsv[:, :, 1] >= args.sat_min) &
                  (hsv[:, :, 2] >= args.val_min) &
                  bbox_mask)

        out = img.copy()
        out[yellow] = PINK_BGR
        cv2.imwrite(os.path.join(args.out_dir, os.path.basename(p)), out)

        if i % 100 == 0:
            print(f"  frame {i}/{len(paths)}")

    print(f"Done → {args.out_dir}")
