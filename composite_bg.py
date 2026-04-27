"""Composite AI background into black pixels of masked frames, with clean edges.

- Morphological opening removes small isolated black blobs near borders
- Gaussian-feathered alpha blend softens the hard foreground/background edge
"""
import argparse, glob, os
import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fg_dir",   required=True, help="rgb_masked frames dir")
    ap.add_argument("--bg_file",  required=True, help="background PNG")
    ap.add_argument("--out_dir",  required=True)
    ap.add_argument("--black_thr", type=int,   default=10,
                    help="Pixel max-channel below this = background")
    ap.add_argument("--open_px",   type=int,   default=3,
                    help="Morphological opening radius on black mask (removes blobs)")
    ap.add_argument("--feather_px",type=int,   default=5,
                    help="Gaussian feather radius for edge blend (0=off)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.fg_dir, "*.png")))
    bg = cv2.imread(args.bg_file)

    fg0 = cv2.imread(paths[0])
    H, W = fg0.shape[:2]
    bg_r = cv2.resize(bg, (W, H))

    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2*args.open_px+1, 2*args.open_px+1))

    for i, p in enumerate(paths):
        fg = cv2.imread(p)

        # Black mask: pixels where all channels are dark
        black = (fg.max(axis=2) < args.black_thr).astype(np.uint8) * 255

        # Remove small isolated blobs (open = erode then dilate)
        if args.open_px > 0:
            black = cv2.morphologyEx(black, cv2.MORPH_OPEN, open_k)

        # Feathered alpha: 1 where background, 0 where foreground
        if args.feather_px > 0:
            sigma = args.feather_px * 0.5
            alpha = cv2.GaussianBlur(black.astype(np.float32) / 255.0,
                                     (2*args.feather_px+1, 2*args.feather_px+1), sigma)
        else:
            alpha = (black / 255.0).astype(np.float32)

        alpha3 = alpha[:, :, np.newaxis]
        out = (fg.astype(np.float32) * (1 - alpha3) +
               bg_r.astype(np.float32) * alpha3).clip(0, 255).astype(np.uint8)

        cv2.imwrite(os.path.join(args.out_dir, os.path.basename(p)), out)
        if i % 100 == 0:
            print(f"  {i}/{len(paths)}")

    print(f"Done → {args.out_dir}")


if __name__ == "__main__":
    main()
