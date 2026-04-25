"""
FoundationPose duck tracker — palm-anchored with hardcoded grasp window.

Normal FP tracking outside the grasp window.
Inside [--grasp_start_frame, --grasp_end_frame]: full palm-delta propagation.
  T_cam_obj[t] = T_cam_palm[t] @ inv(T_cam_palm[t0]) @ T_cam_obj[t0]
where t0 is the frame just before the grasp window starts.

Inputs:
  --test_scene_dir/depth/       VDA depth PNGs (uint16 mm)
  --test_scene_dir/masks/       SAM2VP duck masks
  --palm_poses_npz              (N,4,4) T_cam_palm per RGB frame (Aria frame)
  --grasp_start_frame           first frame of grasp window (default 250 = ~5s @ 50fps)
  --grasp_end_frame             last frame of grasp window  (default 450 = ~9s @ 50fps)
"""
import time
import os
import argparse

import cv2
import numpy as np
import trimesh
import imageio

from estimater import *
from datareader import *
from tools import *


LOG_INTERVAL = 5


def resolve_palm_idx(i: int, offset: int, stride: int, n_palm: int) -> int:
    idx = offset + i * stride
    return max(0, min(n_palm - 1, idx))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument("--mesh_file", type=str,
                        default=f"{code_dir}/demo_data/mustard0/mesh/textured_simple.obj")
    parser.add_argument("--test_scene_dir", type=str,
                        default=f"{code_dir}/demo_data/mustard0")
    parser.add_argument("--palm_poses_npz", type=str, required=True,
                        help="NPZ with 'poses' (N,4,4) T_cam_palm per RGB frame.")
    parser.add_argument("--palm_frame_offset", type=int, default=0)
    parser.add_argument("--palm_frame_stride", type=int, default=1)
    parser.add_argument("--depth_dir", type=str, default=None,
                        help="Depth PNG dir (default: test_scene_dir/depth/)")
    parser.add_argument("--grasp_start_frame", type=int, default=250,
                        help="First frame of grasp window (~5s @ 50fps)")
    parser.add_argument("--grasp_end_frame", type=int, default=450,
                        help="Last frame of grasp window (~9s @ 50fps)")
    parser.add_argument("--reinit_interval", type=int, default=100,
                        help="Re-run est.register every N frames (0 = disabled)")
    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--track_refine_iter", type=int, default=1)
    parser.add_argument("--debug", type=int, default=1)
    parser.add_argument("--debug_dir", type=str, default=f"{code_dir}/debug")
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    mesh = trimesh.load(args.mesh_file)
    mesh.apply_scale(0.001)

    debug_dir = args.debug_dir
    os.system(f"rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam")

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    scorer  = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx   = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
        scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=args.debug, glctx=glctx,
    )
    logging.info("estimator initialized")

    reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)

    palm_data  = np.load(args.palm_poses_npz, allow_pickle=True)
    palm_poses = palm_data["poses"].astype(np.float64)
    n_palm     = palm_poses.shape[0]
    logging.info(f"Loaded {n_palm} palm poses from {args.palm_poses_npz} "
                 f"(RGB frames = {len(reader.color_files)}, "
                 f"offset={args.palm_frame_offset}, stride={args.palm_frame_stride})")

    depth_dir = args.depth_dir or os.path.join(args.test_scene_dir, "depth")

    def load_depth_png(id_str):
        return cv2.imread(os.path.join(depth_dir, f"{id_str}.png"),
                          cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

    anchor_pose     = None
    anchor_palm_inv = None
    depth_scale     = 1.0
    pose            = None

    for i in range(len(reader.color_files)):
        color = reader.get_color(i)
        t1    = time.time()

        depth = load_depth_png(reader.id_strs[i])

        mask_path = os.path.join(args.test_scene_dir, "masks", f"{reader.id_strs[i]}.png")
        mask = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        mask = cv2.morphologyEx(mask * 255, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (60, 60)),
                                ).astype(bool).astype(np.uint8)

        palm_idx   = resolve_palm_idx(i, args.palm_frame_offset, args.palm_frame_stride, n_palm)
        T_cam_palm = palm_poses[palm_idx]

        in_grasp = args.grasp_start_frame <= i <= args.grasp_end_frame

        if i == 0:
            pose = binary_search_depth(est, mesh, color, mask.astype(bool), reader.K, debug=True)
            logging.info(f"Initial duck pose:\n{pose}")
            obj_pixels  = mask > 0
            bsd_z       = float(pose[2, 3])
            vda_z       = depth[obj_pixels].mean() if obj_pixels.any() else 1.0
            depth_scale = bsd_z / vda_z if vda_z > 0 else 1.0
            logging.info(f"Depth scale: {depth_scale:.3f}")
            anchor_pose     = pose.copy()
            anchor_palm_inv = np.linalg.inv(T_cam_palm)
            state = "FREE"

        elif in_grasp:
            state    = "GRASPED"
            d_scaled = depth * depth_scale
            fp_pose  = est.track_one(
                rgb=color, depth=d_scaled, K=reader.K,
                iteration=args.track_refine_iter,
            )
            # Translation from FP (depth-backed, reliable).
            # Rotation from palm delta (FP rotation is noisy under occlusion).
            palm_rot_delta = (T_cam_palm @ anchor_palm_inv)[:3, :3]
            pose = fp_pose.copy()
            pose[:3, :3] = palm_rot_delta @ anchor_pose[:3, :3]
            est.pose_last = torch.from_numpy(pose).float().cuda()

        else:
            state    = "FREE"
            d_scaled = depth * depth_scale
            reinit = (args.reinit_interval > 0 and i > 0
                      and i % args.reinit_interval == 0
                      and mask.sum() > 0)
            if reinit:
                pose = est.register(
                    K=reader.K, rgb=color, depth=d_scaled,
                    ob_mask=mask.astype(bool), iteration=args.est_refine_iter,
                )
                logging.info(f"[frame {i}] periodic re-register")
            else:
                pose = est.track_one(
                    rgb=color, depth=d_scaled, K=reader.K,
                    iteration=args.track_refine_iter,
                )
            # Keep anchor fresh so grasp window starts from the most recent FP pose.
            anchor_pose     = pose.copy()
            anchor_palm_inv = np.linalg.inv(T_cam_palm)

        if i % LOG_INTERVAL == 0:
            logging.info(f"[frame {i}] state={state}")

        t2 = time.time()
        os.makedirs(f"{debug_dir}/ob_in_cam", exist_ok=True)
        np.savetxt(f"{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt", pose.reshape(4, 4))

        if args.debug >= 1:
            center_pose = pose @ np.linalg.inv(to_origin)
            color = cv2.putText(color, f"fps {int(1/(t2-t1))} {state if i>0 else 'INIT'}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K,
                                thickness=3, transparency=0, is_input_rgb=True)

        if args.debug >= 2:
            os.makedirs(f"{debug_dir}/track_vis", exist_ok=True)
            imageio.imwrite(f"{debug_dir}/track_vis/{reader.id_strs[i]}.png", vis)
