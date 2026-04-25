"""
FoundationPose duck tracker — Stage 1: anomaly detection (logging only).

Per-frame:
  1. Load pre-generated SAM2VP mask (live SAM2 swap is Stage 2).
  2. Run FP track_one to get raw pose T_raw.
  3. Predict T_pred from a 5-frame velocity buffer:
        t_pred = t_{-1} + mean_translation_velocity      (or frozen v)
        R_pred = R_{-1}
  4. Kinematic anomaly check:
        ‖t_raw - t_pred‖ > TR_THRESH_M
        angle(R_raw, R_pred) > ROT_THRESH_DEG
  5. Visual gate (Gemini fix): true anomaly only if kinematic jump AND
     mask area dropped below AREA_DROP_RATIO × last_good. This prevents
     misfiring on genuine rapid motion where SAM2 still sees the duck.
  6. During anomaly:
       - Freeze the velocity vector so the linear prediction does not drift.
       - Skip pushing the rescued/anomalous pose to the velocity buffer.
  7. Log anomalies but DO NOT rescue yet — accept T_raw regardless.

Stage 2 will replace step 7 with: render expected silhouette at T_pred
via tools.render_cad_mask, erode/dilate to get safe prompt regions, sample
+/- point prompts, reprompt live SAM2, re-run FP refiner.

Anomaly thresholds set from palm rotation stats on frames 200-400:
per-frame max ≈ 2.85°, p99 ≈ 1.51° → 5° is safely above genuine motion.
"""
import argparse
import os
import time
from collections import deque

import cv2
import imageio
import numpy as np
import trimesh

from estimater import *
from datareader import *
from tools import *


# ── Anomaly thresholds ────────────────────────────────────────────────────────
TR_THRESH_M       = 0.05    # 5 cm/frame translation jump
ROT_THRESH_DEG    = 5.0     # 5°/frame rotation jump
AREA_DROP_RATIO   = 0.80    # mask area below 80% of last-good → visual degradation
HIST_LEN          = 5       # frames in velocity buffer
LOG_INTERVAL      = 5
# ──────────────────────────────────────────────────────────────────────────────


def rotation_angle_deg(R: np.ndarray) -> float:
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def predict_pose(history: deque,
                 frozen_velocity: np.ndarray | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    """T_pred = (t_last + v, R_last). v is frozen velocity if provided,
    otherwise mean translation delta of the buffer.
    Returns (T_pred, v_used). None if history too short.
    """
    if len(history) < 2:
        return None, None
    poses = list(history)
    if frozen_velocity is not None:
        v = frozen_velocity
    else:
        deltas = [poses[i + 1][:3, 3] - poses[i][:3, 3] for i in range(len(poses) - 1)]
        v = np.mean(np.stack(deltas, axis=0), axis=0)
    T_pred = poses[-1].copy()
    T_pred[:3, 3] = poses[-1][:3, 3] + v
    return T_pred, v


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

    reader    = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)
    depth_dir = args.depth_dir or os.path.join(args.test_scene_dir, "depth")

    def load_depth_png(id_str):
        return cv2.imread(os.path.join(depth_dir, f"{id_str}.png"),
                          cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

    history          = deque(maxlen=HIST_LEN)
    depth_scale      = 1.0
    pose             = None
    last_good_R      = None
    last_good_area   = None
    frozen_velocity  = None
    n_anom           = 0

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
            tag = "INIT"
        else:
            d_scaled        = depth * depth_scale
            T_pred, v_used  = predict_pose(history, frozen_velocity)
            T_raw = est.track_one(
                rgb=color, depth=d_scaled, K=reader.K,
                iteration=args.track_refine_iter,
            )

            tr_jump = 0.0
            rt_jump = 0.0
            kine_anom = False
            if T_pred is not None:
                tr_jump = float(np.linalg.norm(T_raw[:3, 3] - T_pred[:3, 3]))
                R_delta = T_raw[:3, :3] @ T_pred[:3, :3].T
                rt_jump = rotation_angle_deg(R_delta)
                kine_anom = (tr_jump > TR_THRESH_M) or (rt_jump > ROT_THRESH_DEG)

            # Gate kinematic anomaly with visual degradation (Gemini fix #3):
            # only treat it as occlusion if mask area also dropped — otherwise
            # it's likely genuine fast motion which we should accept.
            area_drop = (last_good_area is not None
                         and mask_area < AREA_DROP_RATIO * last_good_area)
            anomaly = kine_anom and area_drop

            if anomaly:
                n_anom += 1
                logging.info(f"[frame {i}] ANOMALY  Δt={tr_jump*100:.1f}cm  "
                             f"Δθ={rt_jump:.1f}°  area={mask_area/max(last_good_area,1):.2f}")
                tag = "ANOM"
                # Translation and mask are fine — only rotation is bad.
                # Use T_raw translation + last accepted rotation.
                pose = T_raw.copy()
                if last_good_R is not None:
                    pose[:3, :3] = last_good_R
            else:
                frozen_velocity = None
                last_good_area  = mask_area
                last_good_R     = T_raw[:3, :3].copy()
                tag = "OK" if not kine_anom else "FAST"
                pose = T_raw

        # Don't poison the velocity buffer with rescued/anomalous poses
        # (Gemini fix #2). Push to buffer only when state is healthy.
        if i == 0 or tag in ("OK", "FAST"):
            history.append(pose.copy())

        if i % LOG_INTERVAL == 0:
            logging.info(f"[frame {i}] {tag}  anomalies_so_far={n_anom}")

        t2 = time.time()
        os.makedirs(f"{debug_dir}/ob_in_cam", exist_ok=True)
        np.savetxt(f"{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt", pose.reshape(4, 4))

        if args.debug >= 1:
            center_pose = pose @ np.linalg.inv(to_origin)
            color = cv2.putText(color, f"fps {int(1/(t2-t1))} {tag} a{n_anom}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 0, 255) if tag == "ANOM" else (255, 0, 0), 2)
            vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K,
                                thickness=3, transparency=0, is_input_rgb=True)

        if args.debug >= 2:
            imageio.imwrite(f"{debug_dir}/track_vis/{reader.id_strs[i]}.png", vis)

    logging.info(f"Total anomalies: {n_anom}/{len(reader.color_files)-1}")
