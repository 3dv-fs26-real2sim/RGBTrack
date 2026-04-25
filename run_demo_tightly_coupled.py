"""
FoundationPose duck tracker — Iterative Contour Strategy with SAM2 rescue.

Normal frames: pre-generated SAM2VP masks → FP track_one.

On anomaly (mask area outside [0.85, 1.15] of last_good_area, after warmup):
  1. SAM2ImagePredictor rescues using union(current_mask, CAD_at_last_good_pose)
     as logit prompt → perfect mask (duck + hat, hand excluded by image features)
  2. Reinitialize live SAM2 camera predictor with perfect mask at frame N,
     then replay last SAM2_CONTEXT_FRAMES frames to warm up memory bank
  3. FP re-runs with perfect mask, seeded from last_good_pose
  4. Subsequent frames use live SAM2 (with full memory context)

Next anomaly → same rescue cycle.
"""
import argparse
import os
import time

import cv2
import imageio
import numpy as np
import trimesh
import torch

from sam2.build_sam import build_sam2, build_sam2_camera_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

from estimater import *
from datareader import *
from tools import *


# ── Settings ──────────────────────────────────────────────────────────────────
AREA_RATIO_LO       = 0.85
AREA_RATIO_HI       = 1.15
WARMUP_FRAMES       = 10    # skip anomaly check while last_good_area stabilises
SAM2_CONTEXT_FRAMES = 15    # frames replayed after reinit to warm up memory bank
SAM2_CHECKPOINT     = "/work/courses/3dv/team22/RGBTrack/segment-anything-2-real-time/sam2.1_hiera_small.pt"
SAM2_CONFIG         = "configs/sam2.1/sam2.1_hiera_s.yaml"
LOG_INTERVAL        = 5
# ──────────────────────────────────────────────────────────────────────────────


def sam2_ip_rescue(ip_predictor, color_rgb, cad_mask):
    """SAM2ImagePredictor: CAD silhouette ONLY as logit prompt → refined mask.

    The SAM2VP mask is poisoned by the hand — discard it entirely.
    CAD projection defines where the duck is; SAM2 refines to image edges.
    """
    ip_predictor.set_image(color_rgb)
    cad_256 = cv2.resize((cad_mask > 0).astype(np.float32), (256, 256),
                         interpolation=cv2.INTER_NEAREST)
    logits = (cad_256 - 0.5) * 20.0
    masks, _, _ = ip_predictor.predict(
        point_coords=None, point_labels=None,
        mask_input=logits[None], multimask_output=False,
    )
    return (masks[0] > 0).astype(np.uint8)


class LiveSAM2:
    """SAM2 camera predictor with support for mid-run reinit + memory warmup."""

    def __init__(self, checkpoint, config, device="cuda"):
        self.checkpoint = checkpoint
        self.config     = config
        self.device     = device
        self.predictor  = None
        self.active     = False

    def _new_predictor(self):
        return build_sam2_camera_predictor(
            self.config, self.checkpoint, device=self.device)

    def initialize(self, frame_rgb, mask, context_frames=None):
        """Initialize (or reinitialize) from frame_rgb + mask.

        If context_frames is a list of RGB frames, replay them after init
        to warm up SAM2's memory bank before continuing forward.
        """
        self.predictor = self._new_predictor()
        self.predictor.load_first_frame(frame_rgb)
        self.predictor.add_new_mask(frame_idx=0, obj_id=1,
                                    mask=mask.astype(bool))
        # Warm up memory with recent frames (no mask needed — SAM2 tracks visually)
        if context_frames:
            for f in context_frames:
                self.predictor.track(f)
        self.active = True

    def track(self, frame_rgb):
        _, logits = self.predictor.track(frame_rgb)
        mask = (logits[0] > 0.0).cpu().numpy().astype(np.uint8)
        if mask.ndim == 3:
            mask = mask[0]
        return mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument("--mesh_file",         type=str, required=True)
    parser.add_argument("--test_scene_dir",    type=str, required=True)
    parser.add_argument("--est_refine_iter",   type=int, default=5)
    parser.add_argument("--track_refine_iter", type=int, default=2)
    parser.add_argument("--debug",             type=int, default=2)
    parser.add_argument("--debug_dir",         type=str, default=f"{code_dir}/debug")
    parser.add_argument("--depth_dir",         type=str, default=None)
    parser.add_argument("--sam2_checkpoint",   type=str, default=SAM2_CHECKPOINT)
    parser.add_argument("--sam2_config",       type=str, default=SAM2_CONFIG)
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    mesh = trimesh.load(args.mesh_file)
    mesh.apply_scale(0.001)

    debug_dir = args.debug_dir
    os.system(f"rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis "
              f"{debug_dir}/ob_in_cam {debug_dir}/mask_vis")

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    scorer  = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx   = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
        scorer=scorer, refiner=refiner, debug_dir=debug_dir,
        debug=args.debug, glctx=glctx,
    )
    logging.info("estimator initialized")

    ip_model     = build_sam2(args.sam2_config, args.sam2_checkpoint, device="cuda")
    ip_predictor = SAM2ImagePredictor(ip_model)
    live_sam2    = LiveSAM2(args.sam2_checkpoint, args.sam2_config)

    reader    = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)
    depth_dir = args.depth_dir or os.path.join(args.test_scene_dir, "depth")

    def load_depth_png(id_str):
        return cv2.imread(os.path.join(depth_dir, f"{id_str}.png"),
                          cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

    def load_pregenerated_mask(id_str):
        path = os.path.join(args.test_scene_dir, "masks", f"{id_str}.png")
        m = (cv2.imread(path, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        return cv2.morphologyEx(
            m * 255, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
        ).astype(bool).astype(np.uint8)

    depth_scale    = 1.0
    pose           = None
    last_good_pose = None
    last_good_area = None
    n_anom         = 0
    # Rolling frame buffer for SAM2 memory warmup
    frame_buffer   = []

    for i in range(len(reader.color_files)):
        color = reader.get_color(i)
        t1    = time.time()
        depth = load_depth_png(reader.id_strs[i])
        h_img, w_img = color.shape[:2]

        # Keep a rolling buffer of recent frames for memory warmup
        frame_buffer.append(color)
        if len(frame_buffer) > SAM2_CONTEXT_FRAMES:
            frame_buffer.pop(0)

        # Get mask: pre-generated or live SAM2
        if live_sam2.active:
            mask = live_sam2.track(color)
        else:
            mask = load_pregenerated_mask(reader.id_strs[i])

        mask_area = float(mask.sum())

        if i == 0:
            pose = binary_search_depth(est, mesh, color, mask.astype(bool),
                                       reader.K, debug=True)
            obj_pixels  = mask > 0
            bsd_z       = float(pose[2, 3])
            vda_z       = depth[obj_pixels].mean() if obj_pixels.any() else 1.0
            depth_scale = bsd_z / vda_z if vda_z > 0 else 1.0
            logging.info(f"Initial pose:\n{pose}\nDepth scale: {depth_scale:.3f}")
            last_good_area = mask_area
            last_good_pose = pose.copy()
            tag = "INIT"

        else:
            d_scaled    = depth * depth_scale
            clean_depth = d_scaled * (mask > 0)  # blind FP to hand/background
            T_raw       = est.track_one(rgb=color, depth=clean_depth, K=reader.K,
                                        iteration=args.track_refine_iter)
            area_ratio  = mask_area / max(last_good_area, 1)
            mask_broken = (i > WARMUP_FRAMES and
                           (area_ratio < AREA_RATIO_LO or area_ratio > AREA_RATIO_HI))

            if mask_broken:
                n_anom += 1

                # CAD silhouette at last good pose (T_raw is poisoned by hand)
                cad_mask = render_cad_mask(last_good_pose, mesh, reader.K,
                                           w=w_img, h=h_img)
                if cad_mask is None:
                    cad_mask = np.zeros((h_img, w_img), np.uint8)

                # SAM2IP: CAD only — discard poisoned hand mask entirely
                with torch.inference_mode(), \
                     torch.autocast("cuda", dtype=torch.bfloat16):
                    perfect_mask = sam2_ip_rescue(ip_predictor, color, cad_mask)

                # Reinitialize live SAM2 with perfect mask + memory warmup
                context = frame_buffer[:-1]
                live_sam2.initialize(color, perfect_mask,
                                     context_frames=context)

                # FP re-run: seed from last good pose, depth masked to duck only
                est.pose_last = torch.from_numpy(last_good_pose).float().cuda()
                clean_depth_rescue = d_scaled * (perfect_mask > 0)
                pose = est.track_one(rgb=color, depth=clean_depth_rescue,
                                     K=reader.K,
                                     iteration=args.track_refine_iter)
                last_good_pose = pose.copy()
                last_good_area = float(perfect_mask.sum())
                mask = perfect_mask
                tag  = f"RESCUE({area_ratio:.2f}x)"
                logging.info(f"[frame {i}] {tag}  n_anom={n_anom}")

            else:
                pose           = T_raw
                last_good_area = mask_area
                last_good_pose = pose.copy()
                tag = "OK" if i > WARMUP_FRAMES else "WARM"

        t2 = time.time()
        os.makedirs(f"{debug_dir}/ob_in_cam", exist_ok=True)
        np.savetxt(f"{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt",
                   pose.reshape(4, 4))

        if i % LOG_INTERVAL == 0:
            logging.info(f"[frame {i}] {tag}  n_anom={n_anom}")

        if args.debug >= 1:
            center_pose = pose @ np.linalg.inv(to_origin)
            label_col   = (0, 0, 255) if "RESCUE" in tag else (255, 0, 0)
            color_hud   = cv2.putText(color.copy(),
                                      f"fps {int(1/(t2-t1))} {tag}",
                                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                      0.7, label_col, 2)
            vis = draw_posed_3d_box(reader.K, img=color_hud,
                                    ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(color_hud, ob_in_cam=center_pose, scale=0.1,
                                K=reader.K, thickness=3, transparency=0,
                                is_input_rgb=True)

        if args.debug >= 2:
            imageio.imwrite(f"{debug_dir}/track_vis/{reader.id_strs[i]}.png", vis)

        # mask_vis: red = mask used, green = CAD projection at final pose
        mv = (color * 0.6).astype(np.uint8)
        mv = np.where(mask[..., None] > 0,
                      (color * 0.5 + np.array([0, 0, 255], np.uint8) * 0.5
                       ).astype(np.uint8), mv)
        exp_mask = render_cad_mask(pose, mesh, reader.K, w=w_img, h=h_img)
        if exp_mask is not None and exp_mask.any():
            mv = np.where(exp_mask[..., None] > 0,
                          (mv * 0.5 + np.array([0, 255, 0], np.uint8) * 0.5
                           ).astype(np.uint8), mv)
        label_col_mv = (0, 0, 255) if "RESCUE" in tag else (200, 200, 200)
        mv = cv2.putText(mv, f"{i} {tag}", (10, 30),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_col_mv, 2)
        imageio.imwrite(f"{debug_dir}/mask_vis/{reader.id_strs[i]}.png",
                        mv[..., ::-1])

    logging.info(f"Total rescues: {n_anom}/{len(reader.color_files)-1}")
