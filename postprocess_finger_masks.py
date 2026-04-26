"""Post-process finger masks: warp prev mask via optical flow,
reject new pixels that appear far from the predicted boundary.

Hand translates slowly → silhouette barely changes → tight dilation.
Any blob appearing outside the warped prediction gets removed.
"""
import argparse
import glob
import os

import cv2
import numpy as np


def warp_mask(prev_mask, prev_gray, curr_gray):
    flow  = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    h, w  = prev_mask.shape
    map_x = (np.arange(w) + flow[:, :, 0]).astype(np.float32)
    map_y = (np.arange(h).reshape(-1, 1) + flow[:, :, 1]).astype(np.float32)
    return cv2.remap(prev_mask, map_x, map_y,
                     interpolation=cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir",        required=True)
    ap.add_argument("--rgb_dir",       required=True)
    ap.add_argument("--out_dir",       required=True)
    ap.add_argument("--dilation",      type=int,   default=4,
                    help="Pixels around predicted mask that count as valid growth")
    ap.add_argument("--min_blob_area", type=int,   default=100,
                    help="New-pixel components larger than this outside zone are removed")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    mask_paths = sorted(glob.glob(os.path.join(args.in_dir,  "*.png")))
    rgb_paths  = sorted(glob.glob(os.path.join(args.rgb_dir, "*.png")))
    # Match rgb to available masks only
    mask_stems = {os.path.splitext(os.path.basename(p))[0] for p in mask_paths}
    rgb_paths  = [p for p in rgb_paths
                  if os.path.splitext(os.path.basename(p))[0] in mask_stems]
    N = len(mask_paths)
    print(f"Processing {N} frames  dilation={args.dilation}px  min_blob={args.min_blob_area}")

    k       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (args.dilation*2+1,)*2)
    prev_mask = None
    prev_gray = None
    removed   = 0

    for i, (mp, rp) in enumerate(zip(mask_paths, rgb_paths)):
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8) * 255
        rgb  = cv2.imread(rp)
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

        if prev_mask is None:
            cv2.imwrite(os.path.join(args.out_dir, os.path.basename(mp)), mask)
            prev_mask, prev_gray = mask, gray
            continue

        # Warp previous mask to current frame
        predicted = warp_mask(prev_mask, prev_gray, gray)

        # Allowed zone: predicted + dilation
        allowed = cv2.dilate(predicted, k)

        # New pixels outside allowed zone
        new_px  = cv2.bitwise_and(mask, cv2.bitwise_not(predicted))
        bad_new = cv2.bitwise_and(new_px, cv2.bitwise_not(allowed))

        # Remove large blobs in bad_new
        n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(
            bad_new, connectivity=8)
        refined = mask.copy()
        for lbl in range(1, n_lab):
            if stats[lbl, cv2.CC_STAT_AREA] >= args.min_blob_area:
                refined[labels == lbl] = 0
                removed += 1

        cv2.imwrite(os.path.join(args.out_dir, os.path.basename(mp)), refined)
        prev_mask = refined
        prev_gray = gray

        if i % 100 == 0:
            print(f"  frame {i}/{N}  blobs_removed_total={removed}")

    print(f"Done. Total blobs removed: {removed}  →  {args.out_dir}")
