"""
FoundationPose tracker — VDA depth pipeline + anomaly-freeze rotation.

Combines:
  - run_demo_vda_hand.py   : VDA depth, depth scaling, occlusion recovery,
                             translation correction, depth_pro/metric3d support
  - run_demo_tightly_coupled.py @0247609 : velocity-based anomaly detection,
                             freeze rotation on anomaly (keep translation from FP)

On every frame:
  1. FP track_one → T_raw
  2. Predict T_pred from 5-frame velocity buffer
  3. Kinematic anomaly = large Δt or Δθ AND mask area dropped
  4. Anomaly → pose = T_raw but R = last_good_R  (translation trusted, rotation frozen)
  5. Healthy  → pose = T_raw, update last_good_R and velocity buffer
  6. Occlusion recovery re-init via binary_search_depth when mask recovers
"""
import argparse, os, time
from collections import deque

import cv2
import imageio
import numpy as np
import trimesh

from estimater import *
from datareader import *
from tools import *

try:
    from metric3d_wrapper import Metric3DWrapper
except ImportError:
    Metric3DWrapper = None
try:
    from depth_pro_wrapper import DepthProWrapper
except ImportError:
    DepthProWrapper = None

# ── Anomaly thresholds ────────────────────────────────────────────────────────
TR_THRESH_M      = 0.05   # 5 cm/frame translation jump
ROT_THRESH_DEG   = 5.0    # 5°/frame rotation jump
AREA_DROP_RATIO  = 0.80   # mask area < 80% of last-good → visual degradation
HIST_LEN         = 5      # velocity buffer length
OCCLUSION_THR    = 0.90   # duck mask below this → occluded
RECOVERY_THR     = 0.95   # duck mask above this → re-init after occlusion
LOG_INTERVAL     = 5
# ─────────────────────────────────────────────────────────────────────────────


def rotation_angle_deg(R: np.ndarray) -> float:
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def predict_pose(history: deque, frozen_velocity):
    if len(history) < 2:
        return None, None
    poses = list(history)
    if frozen_velocity is not None:
        v = frozen_velocity
    else:
        deltas = [poses[k+1][:3,3] - poses[k][:3,3] for k in range(len(poses)-1)]
        v = np.mean(np.stack(deltas), axis=0)
    T_pred = poses[-1].copy()
    T_pred[:3, 3] = poses[-1][:3, 3] + v
    return T_pred, v


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument("--mesh_file",        type=str, required=True)
    parser.add_argument("--test_scene_dir",   type=str, required=True)
    parser.add_argument("--est_refine_iter",  type=int, default=2)
    parser.add_argument("--track_refine_iter",type=int, default=2)
    parser.add_argument("--debug",            type=int, default=2)
    parser.add_argument("--debug_dir",        type=str, default=f"{code_dir}/debug")
    parser.add_argument("--masks_dir",        type=str, default=None)
    parser.add_argument("--depth_dir",        type=str, default=None)
    parser.add_argument("--depth_dir_occ",    type=str, default=None)
    parser.add_argument("--metric3d_ckpt",    type=str, default=None)
    parser.add_argument("--depth_pro_ckpt",   type=str, default=None)
    args = parser.parse_args()

    set_logging_format(); set_seed(0)

    mesh = trimesh.load(args.mesh_file)
    mesh.apply_scale(0.001)

    debug_dir = args.debug_dir
    os.system(f"rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam")
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

    scorer  = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx   = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                         mesh=mesh, scorer=scorer, refiner=refiner,
                         debug_dir=debug_dir, debug=args.debug, glctx=glctx)
    logging.info("estimator initialized")

    metric3d = None
    if args.metric3d_ckpt:
        metric3d = Metric3DWrapper(checkpoint_path=args.metric3d_ckpt)
    elif args.depth_pro_ckpt:
        metric3d = DepthProWrapper(checkpoint_path=args.depth_pro_ckpt)

    reader       = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)
    masks_dir    = args.masks_dir    or os.path.join(args.test_scene_dir, "masks")
    depth_dir    = args.depth_dir    or os.path.join(args.test_scene_dir, "depth")
    depth_dir_occ= args.depth_dir_occ or depth_dir

    def load_depth(d_dir, id_str):
        return cv2.imread(os.path.join(d_dir, f"{id_str}.png"),
                          cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

    # State
    history         = deque(maxlen=HIST_LEN)
    depth_scale     = 1.0
    depth_scale_occ = 1.0
    pose            = None
    last_good_R     = None
    last_good_area  = None
    frozen_velocity = None
    frame0_area     = None
    was_occluded    = False
    n_anom          = 0

    for i in range(len(reader.color_files)):
        color = reader.get_color(i)
        t1    = time.time()
        id_str = reader.id_strs[i]

        if metric3d is not None:
            depth = depth_occ = metric3d.estimate(color, reader.K)
        else:
            depth     = load_depth(depth_dir,     id_str)
            depth_occ = load_depth(depth_dir_occ, id_str)

        mask = cv2.imread(os.path.join(masks_dir, f"{id_str}.png"), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)
        mask_area = float(mask.sum())

        if i == 0:
            pose = binary_search_depth(est, mesh, color, mask.astype(bool), reader.K, debug=True)
            logging.info(f"Initial pose:\n{pose}")
            obj_px      = mask > 0
            bsd_z       = float(pose[2, 3])
            vda_z       = depth[obj_px].mean() if obj_px.any() else 1.0
            occ_z       = depth_occ[obj_px].mean() if obj_px.any() else 1.0
            depth_scale     = bsd_z / vda_z if vda_z > 0 else 1.0
            depth_scale_occ = bsd_z / occ_z if occ_z > 0 else 1.0
            logging.info(f"Depth scale: {depth_scale:.3f}  (occ): {depth_scale_occ:.3f}")
            frame0_area    = mask_area
            last_good_area = mask_area
            last_good_R    = pose[:3, :3].copy()
            tag = "INIT"
        else:
            occluded = (frame0_area is not None) and \
                       (mask_area < OCCLUSION_THR * frame0_area)
            d_scaled = (depth_occ * depth_scale_occ) if occluded else (depth * depth_scale)

            # Translation correction — snap XY to mask centroid before FP
            if not occluded and mask.any():
                vs, us = np.where(mask > 0)
                uc = float((us.min() + us.max()) / 2.0)
                vc = float((vs.min() + vs.max()) / 2.0)
                K  = reader.K
                pl = est.pose_last if est.pose_last.dim() == 2 else est.pose_last[0]
                tz = float(pl[2, 3])
                if tz > 0.01:
                    tx = (uc - K[0,2]) * tz / K[0,0]
                    ty = (vc - K[1,2]) * tz / K[1,1]
                    est.pose_last = est.pose_last.clone()
                    if est.pose_last.dim() == 3:
                        est.pose_last[0,0,3] = tx; est.pose_last[0,1,3] = ty
                    else:
                        est.pose_last[0,3] = tx; est.pose_last[1,3] = ty

            T_raw = est.track_one(rgb=color, depth=d_scaled, K=reader.K,
                                   iteration=args.track_refine_iter)

            # Anomaly detection: kinematic jump + area drop
            T_pred, v_used = predict_pose(history, frozen_velocity)
            tr_jump = rt_jump = 0.0
            kine_anom = False
            if T_pred is not None:
                tr_jump   = float(np.linalg.norm(T_raw[:3,3] - T_pred[:3,3]))
                R_delta   = T_raw[:3,:3] @ T_pred[:3,:3].T
                rt_jump   = rotation_angle_deg(R_delta)
                kine_anom = (tr_jump > TR_THRESH_M) or (rt_jump > ROT_THRESH_DEG)

            area_drop = (last_good_area is not None) and \
                        (mask_area < AREA_DROP_RATIO * last_good_area)
            anomaly   = kine_anom and area_drop

            if anomaly:
                n_anom += 1
                logging.info(f"[frame {i}] ANOMALY  Δt={tr_jump*100:.1f}cm  "
                             f"Δθ={rt_jump:.1f}°  area={mask_area/max(last_good_area,1):.2f}")
                tag  = "ANOM"
                pose = T_raw.copy()
                if last_good_R is not None:
                    pose[:3, :3] = last_good_R   # freeze rotation
            else:
                frozen_velocity = None
                last_good_area  = mask_area
                last_good_R     = T_raw[:3, :3].copy()
                tag  = "OK" if not kine_anom else "FAST"
                pose = T_raw

            # Occlusion recovery re-init
            if occluded:
                was_occluded = True
            elif was_occluded and mask_area >= RECOVERY_THR * frame0_area:
                logging.info(f"[frame {i}] Recovery re-init")
                pose = binary_search_depth(est, mesh, color, mask.astype(bool),
                                           reader.K, debug=False)
                last_good_R = pose[:3, :3].copy()
                was_occluded = False

        # Only push healthy frames to velocity buffer
        if i == 0 or tag in ("OK", "FAST"):
            history.append(pose.copy())

        if i % LOG_INTERVAL == 0:
            logging.info(f"[frame {i}] {tag}  anomalies_so_far={n_anom}")

        t2 = time.time()
        np.savetxt(f"{debug_dir}/ob_in_cam/{id_str}.txt", pose.reshape(4,4))

        if args.debug >= 1:
            center_pose = pose @ np.linalg.inv(to_origin)
            color = cv2.putText(color, f"fps {int(1/(t2-t1))} {tag} a{n_anom}",
                                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0,0,255) if tag=="ANOM" else (255,0,0), 2)
            vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K,
                                thickness=3, transparency=0, is_input_rgb=True)
        if args.debug >= 2:
            imageio.imwrite(f"{debug_dir}/track_vis/{id_str}.png", vis)

    logging.info(f"Total anomalies: {n_anom}/{len(reader.color_files)-1}")
