"""
Convert DA3 depths NPZ to uint16 PNG depth maps (mm),
matching the VDA format used by the tracking pipeline.

Usage:
    python convert_da3_npz.py \
        --npz /work/courses/3dv/team22/foundationpose/data/depth_npz/20250804_104715_aria_rgb_cam_depths.npz \
        --out_dir /work/courses/3dv/team22/foundationpose/data/20250804_104715/depth_da3
"""
import argparse, os, glob
import numpy as np
import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str,
                        default="/work/courses/3dv/team22/foundationpose/data/depth_npz/20250804_104715_aria_rgb_cam_depths.npz")
    parser.add_argument("--out_dir", type=str,
                        default="/work/courses/3dv/team22/foundationpose/data/20250804_104715/depth_da3")
    parser.add_argument("--scene_dir", type=str,
                        default="/work/courses/3dv/team22/foundationpose/data/20250804_104715",
                        help="Scene dir to get frame id strings from rgb/")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    depths = np.load(args.npz)["depths"]  # (N, H, W) float32 metres
    N = depths.shape[0]
    print(f"Loaded {N} frames, shape {depths.shape}, range [{depths.min():.3f}, {depths.max():.3f}]m")

    color_files = sorted(glob.glob(os.path.join(args.scene_dir, "rgb", "*.png")))
    id_strs = [os.path.splitext(os.path.basename(f))[0] for f in color_files]

    if len(id_strs) != N:
        print(f"Warning: {len(id_strs)} rgb frames but {N} depth frames — using sequential ids")
        id_strs = [f"{i:06d}" for i in range(N)]

    for i in range(N):
        d_mm = (depths[i] * 1000).astype(np.uint16)
        cv2.imwrite(os.path.join(args.out_dir, f"{id_strs[i]}.png"), d_mm)
        if i % 100 == 0:
            print(f"  [{i}/{N}]")

    print(f"Done — {N} PNGs saved to {args.out_dir}")


if __name__ == "__main__":
    main()
