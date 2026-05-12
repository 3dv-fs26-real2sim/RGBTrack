"""Render a hybrid pose stream: base FP poses everywhere, GoTrack poses inside
a chosen range. Outputs per-frame bbox+axis overlays + a stitched mp4.

Usage:
    python render_hybrid.py \
        --scene_dir /work/scratch/hudela/20250804_104715_full \
        --base_dir  /work/scratch/hudela/20250804_104715_full/fp_no_freeze/ob_in_cam \
        --gotrack_dir /work/scratch/hudela/20250804_104715_full/fp_gotrack/ob_in_cam \
        --mesh /work/courses/3dv/team22/foundationpose/data/object/duck/duck.obj \
        --start 200 --end 400 \
        --out_dir /work/scratch/hudela/20250804_104715_full/fp_hybrid \
        --out_mp4 /work/scratch/hudela/20250804_104715_full_hybrid.mp4 \
        --fps 50
"""

import argparse
import glob
from pathlib import Path
import cv2
import numpy as np
import trimesh


def draw_bbox_axes(bgr: np.ndarray, K: np.ndarray, pose_4x4: np.ndarray,
                   extents_m: np.ndarray, axis_len_m: float = 0.05,
                   color_bbox=(255, 0, 0)) -> np.ndarray:
    out = bgr.copy()
    # bbox
    bbox = np.stack([-extents_m/2, extents_m/2], axis=0).reshape(2, 3)
    corners = np.array([
        [bbox[0,0], bbox[0,1], bbox[0,2]], [bbox[1,0], bbox[0,1], bbox[0,2]],
        [bbox[1,0], bbox[1,1], bbox[0,2]], [bbox[0,0], bbox[1,1], bbox[0,2]],
        [bbox[0,0], bbox[0,1], bbox[1,2]], [bbox[1,0], bbox[0,1], bbox[1,2]],
        [bbox[1,0], bbox[1,1], bbox[1,2]], [bbox[0,0], bbox[1,1], bbox[1,2]],
    ])
    cam = (pose_4x4 @ np.hstack([corners, np.ones((8,1))]).T).T[:, :3]
    proj = (K @ cam.T).T
    proj = proj[:, :2] / np.clip(proj[:, 2:3], 1e-6, None)
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    for a, b in edges:
        cv2.line(out, tuple(int(x) for x in proj[a]),
                 tuple(int(x) for x in proj[b]), color_bbox, 2)
    # axes (X red, Y green, Z blue, OpenCV BGR)
    origin = (pose_4x4 @ np.array([0,0,0,1]))[:3]
    o2 = (K @ origin); o2 = (o2[:2] / o2[2]).astype(int)
    for vec, col in zip(np.eye(3)*axis_len_m, [(0,0,255),(0,255,0),(255,0,0)]):
        p = (pose_4x4 @ np.array([*vec, 1]))[:3]
        p2 = (K @ p); p2 = (p2[:2] / p2[2]).astype(int)
        cv2.line(out, tuple(o2), tuple(p2), col, 2)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir",   required=True)
    ap.add_argument("--base_dir",    required=True, help="FP base ob_in_cam dir")
    ap.add_argument("--gotrack_dir", required=True, help="GoTrack ob_in_cam dir")
    ap.add_argument("--mesh",        required=True)
    ap.add_argument("--start",       type=int, default=200)
    ap.add_argument("--end",         type=int, default=400)
    ap.add_argument("--out_dir",     required=True)
    ap.add_argument("--out_mp4",     required=True)
    ap.add_argument("--fps",         type=int, default=50)
    args = ap.parse_args()

    scene = Path(args.scene_dir)
    K = np.loadtxt(scene / "cam_K.txt").reshape(3, 3)
    mesh = trimesh.load(args.mesh, force="mesh")
    if mesh.extents.max() > 10: mesh.apply_scale(0.001)
    extents = np.array(mesh.extents, dtype=np.float32)

    rgb_files = sorted(glob.glob(str(scene / "rgb" / "*.png")))
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    base = Path(args.base_dir); got = Path(args.gotrack_dir)

    n_got = 0
    for p in rgb_files:
        i = int(Path(p).stem)
        name = Path(p).stem
        use_gotrack = (args.start <= i <= args.end) and (got / f"{name}.txt").exists()
        pose_path = (got if use_gotrack else base) / f"{name}.txt"
        if not pose_path.exists():
            print(f"  skip {name}: no pose"); continue
        pose = np.loadtxt(pose_path).reshape(4, 4)
        bgr = cv2.imread(p)
        col = (0,0,255) if use_gotrack else (255,0,0)   # red for GoTrack frames
        vis = draw_bbox_axes(bgr, K, pose, extents, color_bbox=col)
        tag = "GOTRACK" if use_gotrack else "FP"
        cv2.putText(vis, f"{tag} {i:06d}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        cv2.imwrite(str(out_dir / f"{name}.png"), vis)
        if use_gotrack: n_got += 1
    print(f"rendered {len(rgb_files)} frames ({n_got} GoTrack)")

    # stitch
    src = sorted(glob.glob(str(out_dir / "*.png")))
    h, w = cv2.imread(src[0]).shape[:2]
    vw = cv2.VideoWriter(args.out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    for s in src: vw.write(cv2.imread(s))
    vw.release()
    print(f"wrote {args.out_mp4}")
