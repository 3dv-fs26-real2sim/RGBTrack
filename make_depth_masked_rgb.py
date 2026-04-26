"""Create RGB video masked by depth validity.

For each frame: pixels where depth == 0 (background, far, cut off)
are set to black. Active depth pixels keep their original RGB color.
"""
import argparse
import os

import cv2
import numpy as np
from datareader import YcbineoatReader

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--depth_dir", default=None,
                    help="Override depth dir (default: scene_dir/depth)")
    ap.add_argument("--out_video", required=True)
    ap.add_argument("--fps",       type=int, default=50)
    ap.add_argument("--table_depth", type=float, default=None,
                    help="Cutoff in metres — pixels beyond this are masked out")
    ap.add_argument("--depth_scale", type=float, default=1.0)
    args = ap.parse_args()

    reader    = YcbineoatReader(video_dir=args.scene_dir, shorter_side=None,
                                zfar=float("inf"))
    depth_dir = args.depth_dir or os.path.join(args.scene_dir, "depth")
    N         = len(reader.color_files)

    color0 = reader.get_color(0)
    h, w   = color0.shape[:2]

    # Auto-detect table cutoff from first 50 frames if not given
    if args.table_depth is None:
        print("Sampling first 50 frames for table cutoff...")
        depths = []
        for i in range(min(50, N)):
            p = os.path.join(depth_dir, f"{reader.id_strs[i]}.png")
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

    out = cv2.VideoWriter(args.out_video, cv2.VideoWriter_fourcc(*"mp4v"),
                          args.fps, (w, h))

    for i in range(N):
        color = reader.get_color(i)
        p     = os.path.join(depth_dir, f"{reader.id_strs[i]}.png")
        d     = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if d is None:
            out.write(cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
            continue

        depth = d.astype(np.float32) / 1000.0 * args.depth_scale
        mask  = ((depth > 0.01) & (depth <= table_cutoff)).astype(np.uint8)

        frame = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        frame[mask == 0] = 0

        out.write(frame)
        if i % 100 == 0:
            print(f"  frame {i}/{N}")

    out.release()
    print(f"Done → {args.out_video}")
