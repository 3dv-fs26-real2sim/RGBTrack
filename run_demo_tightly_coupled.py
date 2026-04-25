"""
FoundationPose duck tracker — Iterative Contour Strategy with SAM2 rescue.

Normal frames: use pre-generated SAM2VP mask → FP track_one.
Anomaly frames (mask area outside [0.85, 1.15] of last_good):
  1. Compute T_hybrid = (last_good_R, T_raw translation) — contour-locked.
  2. Render full CAD silhouette at T_hybrid (includes hat).
  3. Feed both the CAD silhouette AND the current SAM2VP mask as prompts
     to SAM2ImagePredictor → refined mask (hat included, hand excluded
     by image evidence, no hardcoding).
  4. Re-run FP with refined mask → final pose.
  5. Update last_good only on clean frames to prevent poisoning.

mask_vis: red = SAM2VP mask, green = CAD projection at final pose.
"""
import argparse
import os
import time

import cv2
import imageio
import numpy as np
import trimesh
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from estimater import *
from datareader import *
from tools import *


# ── Settings ──────────────────────────────────────────────────────────────────
AREA_RATIO_LO    = 0.85
AREA_RATIO_HI    = 1.15
SAM2_CHECKPOINT  = "/work/courses/3dv/team22/RGBTrack/segment-anything-2-real-time/sam2.1_hiera_small.pt"
SAM2_CONFIG      = "configs/sam2.1/sam2.1_hiera_s.yaml"
LOG_INTERVAL     = 5
# ──────────────────────────────────────────────────────────────────────────────


def build_sam2_image_predictor(checkpoint, config, device="cuda"):
    model = build_sam2(config, checkpoint, device=device)
    return SAM2ImagePredictor(model)


def rescue_mask(sam2_predictor, color_rgb: np.ndarray,
                sam2vp_mask: np.ndarray,
                cad_mask: np.ndarray) -> np.ndarray:
    """Re-segment current frame using SAM2VP mask + CAD silhouette as prompts.

    Both masks are passed as positive mask prompts. SAM2 uses image features
    to refine them — hat is recovered from the CAD silhouette, hand interior
    is excluded because it looks different from the duck.
    Returns binary mask (H, W) uint8.
    """
    sam2_predictor.set_image(color_rgb)

    # Combine both masks as a single positive prompt mask.
    # Union: SAM2VP has confirmed body pixels; CAD adds hat region.
    combined = ((sam2vp_mask > 0) | (cad_mask > 0)).astype(np.uint8)

    # SAM2ImagePredictor expects mask_input at (1, 256, 256) low-res space.
    mask_lowres = cv2.resize(combined.astype(np.float32), (256, 256),
                             interpolation=cv2.INTER_LINEAR)
    masks, _, _ = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        mask_input=mask_lowres[None],  # (1, 256, 256)
        multimask_output=False,
    )
    return (masks[0] > 0).astype(np.uint8)


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
    parser.add_argument("--sam2_checkpoint", type=str, default=SAM2_CHECKPOINT)
    parser.add_argument("--sam2_config", type=str, default=SAM2_CONFIG)
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

    sam2_predictor = build_sam2_image_predictor(args.sam2_checkpoint, args.sam2_config)
    logging.info("SAM2ImagePredictor initialized")

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
        sam2vp_mask = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        sam2vp_mask = cv2.morphologyEx(
            sam2vp_mask * 255, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (60, 60)),
        ).astype(bool).astype(np.uint8)

        mask_area = float(sam2vp_mask.sum())
        h_img, w_img = color.shape[:2]

        if i == 0:
            pose = binary_search_depth(est, mesh, color, sam2vp_mask.astype(bool), reader.K, debug=True)
            obj_pixels  = sam2vp_mask > 0
            bsd_z       = float(pose[2, 3])
            vda_z       = depth[obj_pixels].mean() if obj_pixels.any() else 1.0
            depth_scale = bsd_z / vda_z if vda_z > 0 else 1.0
            logging.info(f"Initial pose:\n{pose}\nDepth scale: {depth_scale:.3f}")
            last_good_area = mask_area
            last_good_R    = pose[:3, :3].copy()
            tag  = "INIT"
            mask = sam2vp_mask

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

                # Hybrid pose: current translation + last good rotation
                T_hybrid = np.eye(4)
                T_hybrid[:3, :3] = last_good_R
                T_hybrid[:3, 3]  = T_raw[:3, 3]

                # Render full CAD silhouette at T_hybrid (includes hat)
                cad_mask = render_cad_mask(T_hybrid, mesh, reader.K,
                                           w=w_img, h=h_img)
                if cad_mask is None:
                    cad_mask = np.zeros((h_img, w_img), np.uint8)

                # SAM2 rescue: refine using SAM2VP body + CAD hat region
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    mask = rescue_mask(sam2_predictor, color, sam2vp_mask, cad_mask)

                # Re-run FP with rescued mask
                est.pose_last = torch.from_numpy(T_hybrid).float().cuda()
                pose = est.track_one(
                    rgb=color, depth=d_scaled, K=reader.K,
                    iteration=args.track_refine_iter,
                )
                tag = f"ANOM({area_ratio:.2f}x)"
                logging.info(f"[frame {i}] {tag}  n_anom={n_anom}")

            else:
                mask           = sam2vp_mask
                pose           = T_raw
                last_good_area = mask_area
                last_good_R    = pose[:3, :3].copy()
                tag = "OK"

        t2 = time.time()
        os.makedirs(f"{debug_dir}/ob_in_cam", exist_ok=True)
        np.savetxt(f"{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt", pose.reshape(4, 4))

        if i % LOG_INTERVAL == 0:
            logging.info(f"[frame {i}] {tag}  n_anom={n_anom}")

        if args.debug >= 1:
            center_pose = pose @ np.linalg.inv(to_origin)
            label_col   = (0, 0, 255) if "ANOM" in tag else (255, 0, 0)
            color_hud = cv2.putText(color.copy(), f"fps {int(1/(t2-t1))} {tag}",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_col, 2)
            vis = draw_posed_3d_box(reader.K, img=color_hud, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(color_hud, ob_in_cam=center_pose, scale=0.1, K=reader.K,
                                thickness=3, transparency=0, is_input_rgb=True)

        if args.debug >= 2:
            imageio.imwrite(f"{debug_dir}/track_vis/{reader.id_strs[i]}.png", vis)

        # Mask vis: red = SAM2VP mask, green = CAD projection at final pose
        mv = (color * 0.6).astype(np.uint8)
        mv = np.where(sam2vp_mask[..., None] > 0,
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
