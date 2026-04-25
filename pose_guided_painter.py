"""Pose-guided duck painter.

Paints all frames yellow (duck) / blue (background) using:
- Frames outside grasp window: union(SAM2VP mask, CAD projection at FP pose)
- Frames inside grasp window:  CAD projection at palm-delta propagated pose

Output: rgb_painted/ and rgb_painted_jpg/ in the scene directory.

Usage:
    python pose_guided_painter.py \
        --scene_dir  <scene> \
        --ob_in_cam  <debug_dir>/ob_in_cam \
        --palm_poses_npz <path>/palm_poses_*.npz \
        --mesh_file  <duck.obj> \
        --grasp_start_frame 250 \
        --grasp_end_frame   450
"""
import argparse
import glob
import os

import cv2
import numpy as np
import trimesh

from datareader import YcbineoatReader
from tools import render_cad_mask


def paint_frame(img, mask):
    """Yellow duck, blue background, original V channel preserved."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    out = hsv.copy()
    out[ mask, 0] = 25;  out[ mask, 1] = 220
    out[~mask, 0] = 115; out[~mask, 1] = 180
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir",        required=True)
    ap.add_argument("--ob_in_cam",        required=True,
                    help="Directory with ob_in_cam/*.txt poses from a previous FP run")
    ap.add_argument("--palm_poses_npz",   required=True)
    ap.add_argument("--mesh_file",        required=True)
    ap.add_argument("--grasp_start_frame",type=int, default=250)
    ap.add_argument("--grasp_end_frame",  type=int, default=450)
    args = ap.parse_args()

    mesh = trimesh.load(args.mesh_file)
    mesh.apply_scale(0.001)

    reader      = YcbineoatReader(video_dir=args.scene_dir, shorter_side=None, zfar=np.inf)
    palm_poses  = np.load(args.palm_poses_npz, allow_pickle=True)["poses"].astype(np.float64)
    n_palm      = palm_poses.shape[0]
    mask_dir    = os.path.join(args.scene_dir, "masks")
    out_dir     = os.path.join(args.scene_dir, "rgb_painted")
    jpg_dir     = os.path.join(args.scene_dir, "rgb_painted_jpg")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(jpg_dir, exist_ok=True)

    N = len(reader.color_files)
    h, w = reader.get_color(0).shape[:2]

    # Find last good pose before grasp window for palm-delta anchor
    anchor_pose     = None
    anchor_palm_inv = None
    anchor_idx      = args.grasp_start_frame - 1

    pose_path = os.path.join(args.ob_in_cam, f"{reader.id_strs[anchor_idx]}.txt")
    if os.path.exists(pose_path):
        anchor_pose     = np.loadtxt(pose_path).reshape(4, 4)
        anchor_palm_inv = np.linalg.inv(palm_poses[min(anchor_idx, n_palm - 1)])
    else:
        print(f"Warning: anchor pose not found at {pose_path}, using frame 0")
        anchor_pose     = np.loadtxt(os.path.join(args.ob_in_cam,
                                     f"{reader.id_strs[0]}.txt")).reshape(4, 4)
        anchor_palm_inv = np.linalg.inv(palm_poses[0])

    for i in range(N):
        color    = reader.get_color(i)
        id_str   = reader.id_strs[i]
        palm_idx = min(i, n_palm - 1)

        in_grasp = args.grasp_start_frame <= i <= args.grasp_end_frame

        if in_grasp:
            # Palm-delta propagated pose
            pose = palm_poses[palm_idx] @ anchor_palm_inv @ anchor_pose
        else:
            # FP pose from previous run
            fp_path = os.path.join(args.ob_in_cam, f"{id_str}.txt")
            if os.path.exists(fp_path):
                pose = np.loadtxt(fp_path).reshape(4, 4)
            else:
                pose = anchor_pose

        # CAD projection at pose
        cad_mask = render_cad_mask(pose, mesh, reader.K, w=w, h=h)
        if cad_mask is None:
            cad_mask = np.zeros((h, w), np.uint8)

        # SAM2VP mask (use for non-grasp frames to catch body texture)
        mp = os.path.join(mask_dir, f"{id_str}.png")
        sam2_mask = (cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 127) if os.path.exists(mp) \
                    else np.zeros((h, w), bool)

        duck_mask = (cad_mask > 0) | sam2_mask

        result = paint_frame(cv2.cvtColor(color, cv2.COLOR_RGB2BGR), duck_mask)
        cv2.imwrite(os.path.join(out_dir, f"{id_str}.png"), result)
        cv2.imwrite(os.path.join(jpg_dir,  f"{id_str}.jpg"), result)

        if i % 100 == 0:
            print(f"frame {i}/{N}")

    print(f"Done → {out_dir}")
