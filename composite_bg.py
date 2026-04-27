"""Composite AI background into black pixels of masked frames, with clean edges.

- Morphological opening removes small isolated black blobs near borders
- Gaussian-feathered alpha blend softens the hard foreground/background edge
- Optional: hand mask pixels stamped from original RGB — fully opaque, no blending
"""
import argparse, glob, os
import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fg_dir",        required=True, help="rgb_masked frames dir")
    ap.add_argument("--bg_file",       required=True, help="background PNG")
    ap.add_argument("--out_dir",       required=True)
    ap.add_argument("--orig_dir",      default=None,
                    help="Original rgb/ dir — arm pixels stamped from here when --hand_mask_dir set")
    ap.add_argument("--hand_mask_dir", default=None,
                    help="Dir of hand masks (white=arm); forces arm pixels fully opaque from orig_dir")
    ap.add_argument("--black_thr",     type=int, default=10,
                    help="Pixel max-channel below this = background")
    ap.add_argument("--open_px",       type=int, default=3,
                    help="Morphological opening radius on black mask (removes blobs)")
    ap.add_argument("--close_px",      type=int, default=4,
                    help="Morphological closing radius on foreground mask (fills surface-tension black gaps at tight boundaries)")
    ap.add_argument("--feather_px",    type=int, default=0,
                    help="Gaussian feather radius for edge blend (0=off, avoids arm translucency)")
    args = ap.parse_args()

    use_hand = args.hand_mask_dir and args.orig_dir

    os.makedirs(args.out_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.fg_dir, "*.png")))
    bg = cv2.imread(args.bg_file)

    fg0 = cv2.imread(paths[0])
    H, W = fg0.shape[:2]
    bg_r = cv2.resize(bg, (W, H))

    open_k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (2*args.open_px+1,  2*args.open_px+1))
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (2*args.close_px+1, 2*args.close_px+1))

    for i, p in enumerate(paths):
        name = os.path.basename(p)
        fg = cv2.imread(p)

        # Black mask: pixels where all channels are dark
        black = (fg.max(axis=2) < args.black_thr).astype(np.uint8) * 255

        # Remove small isolated blobs
        if args.open_px > 0:
            black = cv2.morphologyEx(black, cv2.MORPH_OPEN, open_k)

        # Fill narrow black gaps at tight foreground junctions (surface tension)
        if args.close_px > 0:
            fg_mask = 255 - black
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, close_k)
            black = 255 - fg_mask

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

        # Stamp arm pixels from original RGB — fully opaque, overrides any blending
        if use_hand:
            hm = cv2.imread(os.path.join(args.hand_mask_dir, name), cv2.IMREAD_GRAYSCALE)
            orig = cv2.imread(os.path.join(args.orig_dir, name))
            if hm is not None and orig is not None:
                out[hm > 127] = orig[hm > 127]

        # Final pass: any remaining near-black pixel → background (kills boundary remnants)
        still_black = out.max(axis=2) < args.black_thr
        out[still_black] = bg_r[still_black]

        cv2.imwrite(os.path.join(args.out_dir, name), out)
        if i % 100 == 0:
            print(f"  {i}/{len(paths)}")

    print(f"Done → {args.out_dir}")


if __name__ == "__main__":
    main()
