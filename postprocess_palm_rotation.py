"""Patch rotation of FP-produced ob_in_cam txts with palm-delta rotation
inside a hardcoded grasp window, then re-render the track-vis frames.

Reads:  <fp_debug_dir>/ob_in_cam/*.txt   (FP poses)
        <palm_poses_npz>                 (T_cam_palm per frame)
Writes: <out_debug_dir>/ob_in_cam/*.txt  (patched poses)
        <out_debug_dir>/track_vis/*.png  (re-rendered visualization)
"""
import argparse
import os

import cv2
import imageio
import numpy as np
import trimesh

from datareader import YcbineoatReader
from Utils import draw_posed_3d_box, draw_xyz_axis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh_file", required=True)
    ap.add_argument("--test_scene_dir", required=True)
    ap.add_argument("--fp_debug_dir", required=True,
                    help="Existing FP run dir containing ob_in_cam/*.txt")
    ap.add_argument("--out_debug_dir", required=True)
    ap.add_argument("--palm_poses_npz", required=True)
    ap.add_argument("--grasp_start_frame", type=int, default=250)
    ap.add_argument("--grasp_end_frame",   type=int, default=450)
    args = ap.parse_args()

    os.makedirs(f"{args.out_debug_dir}/ob_in_cam", exist_ok=True)
    os.makedirs(f"{args.out_debug_dir}/track_vis", exist_ok=True)

    mesh = trimesh.load(args.mesh_file)
    mesh.apply_scale(0.001)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    reader     = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)
    palm_poses = np.load(args.palm_poses_npz, allow_pickle=True)["poses"].astype(np.float64)
    n_palm     = palm_poses.shape[0]

    # Anchor at frame just before grasp window starts.
    anchor_idx = max(0, args.grasp_start_frame - 1)
    anchor_pose = np.loadtxt(
        os.path.join(args.fp_debug_dir, "ob_in_cam", f"{reader.id_strs[anchor_idx]}.txt")
    ).reshape(4, 4)
    anchor_palm_inv = np.linalg.inv(palm_poses[min(anchor_idx, n_palm - 1)])

    for i in range(len(reader.color_files)):
        id_str  = reader.id_strs[i]
        fp_pose = np.loadtxt(
            os.path.join(args.fp_debug_dir, "ob_in_cam", f"{id_str}.txt")
        ).reshape(4, 4)

        if args.grasp_start_frame <= i <= args.grasp_end_frame:
            T_cam_palm     = palm_poses[min(i, n_palm - 1)]
            palm_rot_delta = (T_cam_palm @ anchor_palm_inv)[:3, :3]
            pose = fp_pose.copy()                                 # FP translation
            pose[:3, :3] = palm_rot_delta @ anchor_pose[:3, :3]   # palm-delta rotation
            tag = "GRASPED"
        else:
            pose = fp_pose
            tag  = "FREE"

        np.savetxt(
            os.path.join(args.out_debug_dir, "ob_in_cam", f"{id_str}.txt"),
            pose.reshape(4, 4),
        )

        color       = reader.get_color(i)
        center_pose = pose @ np.linalg.inv(to_origin)
        color = cv2.putText(color, f"{tag} {i}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
        vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K,
                            thickness=3, transparency=0, is_input_rgb=True)
        imageio.imwrite(
            os.path.join(args.out_debug_dir, "track_vis", f"{id_str}.png"),
            vis,
        )

        if i % 50 == 0:
            print(f"frame {i}/{len(reader.color_files)}  state={tag}")

    print("done")


if __name__ == "__main__":
    main()
