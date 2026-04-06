"""
Offline Depth Pro depth map generation.
Saves uint16 PNG depth maps (mm) to test_scene_dir/depth_pro/
matching the same format as VDA depth maps.

Run once, then use --depth_dir depth_pro in tracking jobs.
"""
import os, argparse
import numpy as np
import cv2
import torch
from datareader import YcbineoatReader

DEPTH_PRO_DIR = "/work/courses/3dv/team22/ml-depth-pro"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_scene_dir", type=str,
                        default="/work/courses/3dv/team22/foundationpose/data/20250804_104715")
    parser.add_argument("--depth_pro_ckpt", type=str,
                        default=f"{DEPTH_PRO_DIR}/checkpoints/depth_pro.pt")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output dir for depth PNGs. Defaults to test_scene_dir/depth_pro/")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.test_scene_dir, "depth_pro")
    os.makedirs(out_dir, exist_ok=True)

    from depth_pro_wrapper import DepthProWrapper
    model = DepthProWrapper(checkpoint_path=args.depth_pro_ckpt)
    print("Depth Pro loaded")

    reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=float("inf"))

    for i in range(len(reader.color_files)):
        color = reader.get_color(i)
        depth = model.estimate(color, reader.K)          # float32 metres

        depth_mm = (depth * 1000).astype(np.uint16)
        out_path = os.path.join(out_dir, f"{reader.id_strs[i]}.png")
        cv2.imwrite(out_path, depth_mm)

        if i % 50 == 0:
            print(f"  [{i}/{len(reader.color_files)}] saved {out_path}")

    print(f"Done — {len(reader.color_files)} depth maps saved to {out_dir}")


if __name__ == "__main__":
    main()
