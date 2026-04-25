"""
FoundationPose duck tracker — Iterative Contour Strategy.

Per-frame:
  1. Run FP track_one on the SAM2VP mask → T_raw (translation locked to mask contour).
  2. Anomaly detection: mask area ratio outside [0.85, 1.15] of last_good_area.
  3. On anomaly (hand grabs / hat dropout):
       T_hybrid = (last_good_R, T_raw[:3,3])
       — translation follows the duck's contour, rotation stays frozen.
  4. Render expected silhouette at final pose → green overlay for mask_vis.
     Red = SAM2VP mask. Green = CAD projection. Overlap shows what Stage 2
     will use to prompt SAM2 into recovering the hat.

Velocity buffer removed — it suffers from poisoning when the mask bleeds
onto the hand before the threshold trips.
"""
import argparse
import os
import time

import cv2
import imageio
import numpy as np
import trimesh

from estimater import *
from datareader import *
from tools import *


# ── Settings ──────────────────────────────────────────────────────────────────
AREA_RATIO_LO = 0.85    # mask shrinks below this → anomaly (hat dropout / occlusion)
AREA_RATIO_HI = 1.15    # mask grows above this  → anomaly (hand bleed-in)
LOG_INTERVAL  = 5
# ──────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument("--mesh_file", type=str, required=True)
    parser.add_argument("--test_scene_dir", type=str, required=True)
    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--track_refine_iter", type=int, default=2)
    parser.add_argument("--debug", type=int, default=2)
    parser.add_argument("--debug_dir", type=str, default=f"{code_dir}/debug")
    parser.add_argument("--depth_dir", type=str, default=None)
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    mesh = trimesh.load(args.mesh_file)
    mesh.apply_scale(0.001)

    debug_dir = args.debug_dir
    os.system(f"rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam {debug_dir}/mask_vis")

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

    reader    = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)
    depth_dir = args.depth_dir or os.path.join(args.test_scene_dir, "depth")

    def load_depth_png(id_str):
        return cv2.imread(os.path.join(depth_dir, f"{id_str}.png"),
                          cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

    depth_scale    = 1.0
    pose           = None
    last_good_area = None
    last_good_R    = None
    n_anom         = 0

    for i in range(len(reader.color_files)):
        color = reader.get_color(i)
        t1    = time.time()

        depth = load_depth_png(reader.id_strs[i])
        mask_path = os.path.join(args.test_scene_dir, "masks", f"{reader.id_strs[i]}.png")
        mask = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        mask = cv2.morphologyEx(mask * 255, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (60, 60)),
                                ).astype(bool).astype(np.uint8)

        mask_area = float(mask.sum())

        if i == 0:
            pose = binary_search_depth(est, mesh, color, mask.astype(bool), reader.K, debug=True)
            obj_pixels  = mask > 0
            bsd_z       = float(pose[2, 3])
            vda_z       = depth[obj_pixels].mean() if obj_pixels.any() else 1.0
            depth_scale = bsd_z / vda_z if vda_z > 0 else 1.0
            logging.info(f"Initial pose:\n{pose}\nDepth scale: {depth_scale:.3f}")
            last_good_area = mask_area
            last_good_R    = pose[:3, :3].copy()
            tag = "INIT"

        else:
            d_scaled = depth * depth_scale
            T_raw = est.track_one(
                rgb=color, depth=d_scaled, K=reader.K,
                iteration=args.track_refine_iter,
            )

            area_ratio  = mask_area / max(last_good_area, 1)
            mask_broken = area_ratio < AREA_RATIO_LO or area_ratio > AREA_RATIO_HI

            if mask_broken:
                n_anom += 1
                tag = f"ANOM({area_ratio:.2f}x)"
                # Hybrid pose: contour-locked translation + frozen rotation
                pose = np.eye(4)
                pose[:3, :3] = last_good_R
                pose[:3, 3]  = T_raw[:3, 3]
                est.pose_last = torch.from_numpy(pose).float().cuda()
            else:
                tag = "OK"
                pose           = T_raw.copy()
                last_good_area = mask_area
                last_good_R    = pose[:3, :3].copy()

            if mask_broken or i % LOG_INTERVAL == 0:
                logging.info(f"[frame {i}] {tag}  area_ratio={area_ratio:.2f}  n_anom={n_anom}")

        t2 = time.time()
        os.makedirs(f"{debug_dir}/ob_in_cam", exist_ok=True)
        np.savetxt(f"{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt", pose.reshape(4, 4))

        if args.debug >= 1:
            center_pose = pose @ np.linalg.inv(to_origin)
            label_col   = (0, 0, 255) if "ANOM" in tag else (255, 0, 0)
            color = cv2.putText(color, f"fps {int(1/(t2-t1))} {tag}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_col, 2)
            vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K,
                                thickness=3, transparency=0, is_input_rgb=True)

        if args.debug >= 2:
            imageio.imwrite(f"{debug_dir}/track_vis/{reader.id_strs[i]}.png", vis)

        # Mask vis: red = SAM2VP mask, green = CAD projection at final pose.
        h_img, w_img = color.shape[:2]
        mv = (color * 0.6).astype(np.uint8)
        mv = np.where(mask[..., None] > 0,
                      (color * 0.5 + np.array([0, 0, 255], np.uint8) * 0.5
                       ).astype(np.uint8), mv)
        exp_mask = render_cad_mask(pose, mesh, reader.K, w=w_img, h=h_img)
        if exp_mask is not None and exp_mask.any():
            mv = np.where(exp_mask[..., None] > 0,
                          (mv * 0.5 + np.array([0, 255, 0], np.uint8) * 0.5
                           ).astype(np.uint8), mv)
        label_col_mv = (0, 0, 255) if "ANOM" in tag else (200, 200, 200)
        mv = cv2.putText(mv, f"{i} {tag}", (10, 30),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_col_mv, 2)
        imageio.imwrite(f"{debug_dir}/mask_vis/{reader.id_strs[i]}.png", mv[..., ::-1])

    logging.info(f"Total anomalies: {n_anom}/{len(reader.color_files)-1}")
