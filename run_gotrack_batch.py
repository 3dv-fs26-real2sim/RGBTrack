"""
Batch-refine FP poses with GoTrack over a frame range.

Reads:
  <scene_dir>/rgb/<frame>.png
  <scene_dir>/fp/ob_in_cam/<frame>.txt
  <scene_dir>/cam_K.txt

Writes:
  <scene_dir>/fp_gotrack/ob_in_cam/<frame>.txt   refined 4×4 poses
  <scene_dir>/fp_gotrack/track_vis/<frame>.png   bbox+axis overlay

Prints per-frame Δt (cm) and Δθ (deg) between init and refined.
"""

import argparse
import glob
import os
from pathlib import Path
import time

import cv2
import numpy as np
import trimesh

from gotrack_refiner import GoTrackRefiner


def rotation_delta_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    R = R1.T @ R2
    cos = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def draw_overlay(rgb_bgr: np.ndarray, K: np.ndarray, pose_4x4: np.ndarray,
                 mesh_in_m: trimesh.Trimesh, color: tuple) -> np.ndarray:
    """Project mesh bbox edges + axes into the image."""
    extents = mesh_in_m.extents
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)
    corners = np.array([
        [bbox[0,0], bbox[0,1], bbox[0,2]], [bbox[1,0], bbox[0,1], bbox[0,2]],
        [bbox[1,0], bbox[1,1], bbox[0,2]], [bbox[0,0], bbox[1,1], bbox[0,2]],
        [bbox[0,0], bbox[0,1], bbox[1,2]], [bbox[1,0], bbox[0,1], bbox[1,2]],
        [bbox[1,0], bbox[1,1], bbox[1,2]], [bbox[0,0], bbox[1,1], bbox[1,2]],
    ])
    cam_pts = (pose_4x4 @ np.hstack([corners, np.ones((8,1))]).T).T[:, :3]
    proj = (K @ cam_pts.T).T
    proj = proj[:, :2] / np.clip(proj[:, 2:3], 1e-6, None)
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    out = rgb_bgr.copy()
    for a, b in edges:
        pa = tuple(int(x) for x in proj[a])
        pb = tuple(int(x) for x in proj[b])
        cv2.line(out, pa, pb, color, 2)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scene_dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--mesh", required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end",   type=int, default=-1, help="-1 = all frames")
    p.add_argument("--n_iter", type=int, default=3)
    p.add_argument("--rotation_only", action="store_true",
                   help="Keep init translation, only take refined rotation from GoTrack.")
    p.add_argument("--gotrack_root", default="/work/courses/3dv/team22/gotrack")
    p.add_argument("--mask_dir", default=None,
                   help="Optional. Per-frame SAM masks. When provided, GoTrack's "
                        "crop window is anchored on the mask 2D bbox instead of "
                        "the projected mesh — robust to small rotation errors in "
                        "the init pose.")
    args = p.parse_args()

    scene = Path(args.scene_dir)
    out   = scene / "fp_gotrack"
    (out / "ob_in_cam").mkdir(parents=True, exist_ok=True)
    (out / "track_vis").mkdir(parents=True, exist_ok=True)

    K = np.loadtxt(scene / "cam_K.txt").reshape(3, 3)
    rgb_files = sorted(glob.glob(str(scene / "rgb" / "*.png")))
    if args.end < 0: args.end = len(rgb_files)
    sel = rgb_files[args.start:args.end]
    print(f"Refining {len(sel)} frames ({args.start}..{args.end})")

    refiner = GoTrackRefiner(
        ckpt_path=args.ckpt, mesh_path=args.mesh,
        gotrack_root=args.gotrack_root, obj_id=1, device="cuda",
    )
    mesh_m = refiner._mesh   # already scaled to metres in __init__

    deltas_t = []
    deltas_r = []
    t0 = time.time()
    for p_rgb in sel:
        name = Path(p_rgb).stem
        init_path = scene / "fp" / "ob_in_cam" / f"{name}.txt"
        if not init_path.exists():
            print(f"  skip {name}: no init pose")
            continue
        rgb = cv2.cvtColor(cv2.imread(p_rgb), cv2.COLOR_BGR2RGB)
        init = np.loadtxt(init_path).reshape(4, 4)

        mask = None
        if args.mask_dir is not None:
            mp = Path(args.mask_dir) / f"{name}.png"
            if mp.exists():
                mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)

        refined = refiner.refine(rgb, K, init, n_iter=args.n_iter, mask=mask)
        if args.rotation_only:
            refined[:3, 3] = init[:3, 3]   # keep init translation

        dt = float(np.linalg.norm(refined[:3,3] - init[:3,3])) * 100.0
        dr = rotation_delta_deg(init[:3,:3], refined[:3,:3])
        deltas_t.append(dt); deltas_r.append(dr)
        print(f"  {name}  Δt={dt:5.2f}cm  Δθ={dr:5.2f}°")

        np.savetxt(out / "ob_in_cam" / f"{name}.txt", refined.reshape(4,4))

        # overlay: green = init, red = refined
        bgr = cv2.imread(p_rgb)
        vis = draw_overlay(bgr, K, init,    mesh_m, color=(0,200,0))   # green
        vis = draw_overlay(vis, K, refined, mesh_m, color=(0,0,255))   # red
        cv2.imwrite(str(out / "track_vis" / f"{name}.png"), vis)

    n = len(deltas_t)
    if n > 0:
        print(f"\nTotal: {n} frames in {time.time()-t0:.1f}s")
        print(f"  Δt (cm):  median {np.median(deltas_t):.2f}  max {max(deltas_t):.2f}")
        print(f"  Δθ (deg): median {np.median(deltas_r):.2f}  max {max(deltas_r):.2f}")
        print(f"\nOutputs: {out}")
