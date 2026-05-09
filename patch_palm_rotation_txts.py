"""Patch FP ob_in_cam txts with palm-delta rotation inside a grasp window.
Same rotation-only swap as postprocess_palm_rotation.py but writes only txts
(no rendering, no GPU). Runs in seconds.

Reads:  --fp_dir/<frame>.txt           (FP 4×4 cam_from_model)
        --palm_poses_npz["poses"]      (N, 4, 4) T_cam_palm per frame
Writes: --out_dir/<frame>.txt          (rotation patched inside grasp window)

Anchor pose: --fp_dir/000000.txt rotation. Translation is always FP's.
"""

import argparse
import glob
import os
from pathlib import Path
import numpy as np


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp_dir",          required=True, help="dir with FP ob_in_cam/*.txt")
    ap.add_argument("--out_dir",         required=True, help="dir to write patched txts")
    ap.add_argument("--palm_poses_npz",  required=True)
    ap.add_argument("--grasp_start_frame", type=int, default=220)
    ap.add_argument("--grasp_end_frame",   type=int, default=450)
    args = ap.parse_args()

    fp_files = sorted(glob.glob(f"{args.fp_dir}/*.txt"))
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    palm = np.load(args.palm_poses_npz, allow_pickle=True)["poses"].astype(np.float64)
    n_palm = palm.shape[0]
    print(f"FP frames: {len(fp_files)}  palm frames: {n_palm}  "
          f"grasp [{args.grasp_start_frame},{args.grasp_end_frame}]")

    anchor_pose     = np.loadtxt(fp_files[0]).reshape(4, 4)
    anchor_palm_inv = np.linalg.inv(palm[0])
    anchor_R        = anchor_pose[:3, :3]

    n_patched = 0
    for i, p in enumerate(fp_files):
        T_fp = np.loadtxt(p).reshape(4, 4)
        if args.grasp_start_frame <= i <= args.grasp_end_frame:
            T_cam_palm     = palm[min(i, n_palm - 1)]
            palm_R_delta   = (T_cam_palm @ anchor_palm_inv)[:3, :3]
            pose = T_fp.copy()                              # keep FP translation
            pose[:3, :3] = palm_R_delta @ anchor_R          # only rotation overridden
            n_patched += 1
        else:
            pose = T_fp
        np.savetxt(out / Path(p).name, pose.reshape(4, 4))

    print(f"wrote {len(fp_files)} txts → {out}  (patched: {n_patched})")
