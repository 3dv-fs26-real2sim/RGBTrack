"""Scale a per-frame depth map sequence so the duck's metric Z at frame 0
matches what BSD (on frame-0 RGB + mask + CAD) predicts.

Reads:
    --depth_dir   PNGs (uint16 mm)
    --rgb         frame-0 RGB
    --mask        frame-0 binary mask of the object
    --cam_K       3x3 intrinsics
    --mesh        CAD .obj (mm by default; scaled to m inside)

Writes:
    --out_dir     PNGs scaled so mask_mean(depth_in_m) ≈ bsd_z at frame 0
"""

import argparse
import glob
import os
from pathlib import Path

import cv2
import numpy as np
import trimesh

from estimater import (
    FoundationPose,
    PoseRefinePredictor,
    ScorePredictor,
    set_logging_format,
    set_seed,
)
from datareader import YcbineoatReader   # noqa: F401  (only for compat with FP env)
from tools import binary_search_depth
import nvdiffrast.torch as dr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth_dir", required=True)
    ap.add_argument("--rgb",       required=True, help="frame-0 RGB png")
    ap.add_argument("--mask",      required=True, help="frame-0 mask png")
    ap.add_argument("--cam_K",     required=True, help="cam_K.txt (3x3)")
    ap.add_argument("--mesh",      required=True, help="CAD .obj")
    ap.add_argument("--out_dir",   required=True)
    args = ap.parse_args()

    set_logging_format(); set_seed(0)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    K = np.loadtxt(args.cam_K).reshape(3, 3)
    rgb = cv2.cvtColor(cv2.imread(args.rgb), cv2.COLOR_BGR2RGB)
    mask = (cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE) > 127)

    mesh = trimesh.load(args.mesh, force="mesh")
    mesh.apply_scale(0.001)
    scorer = ScorePredictor(); refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                         mesh=mesh, scorer=scorer, refiner=refiner,
                         debug=0, debug_dir="/tmp", glctx=glctx)

    pose0 = binary_search_depth(est, mesh, rgb, mask, K, debug=False)
    bsd_z = float(pose0[2, 3])
    d0 = cv2.imread(f"{args.depth_dir}/000000.png",
                    cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    pred_z = float(d0[mask].mean()) if mask.any() else 1.0
    scale = bsd_z / pred_z if pred_z > 0 else 1.0
    print(f"BSD z = {bsd_z:.3f}  pred mean z = {pred_z:.3f}  scale = {scale:.3f}")

    files = sorted(glob.glob(f"{args.depth_dir}/*.png"))
    for p in files:
        d = cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        d_scaled = (d * scale * 1000.0).clip(0, 65535).astype(np.uint16)
        cv2.imwrite(str(out_dir / os.path.basename(p)), d_scaled)
    print(f"wrote {len(files)} scaled depth PNGs → {out_dir}")


if __name__ == "__main__":
    main()
