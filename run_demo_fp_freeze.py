"""
FoundationPose tracker — VDA depth pipeline + ScoreNet rotation gate.

On every frame:
  1. Translation correction → snap pose_last.xy to mask centroid
  2. FP track_one → T_raw (translation trusted)
  3. ScoreNet on T_raw. If score < baseline*(1-SCORE_DROP_MARGIN),
     reject the rotation update and keep last_good_R; translation stays.
  4. Occlusion recovery re-init via binary_search_depth when mask recovers
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

# ── Settings ──────────────────────────────────────────────────────────────────
OCCLUSION_THR      = 0.90   # mask below this fraction of frame0 → occluded
RECOVERY_THR       = 0.95   # mask above this fraction of frame0 → re-init
SCORE_DROP_MARGIN  = 0.35   # reject rotation if score < baseline * (1 - margin)
STUCK_OCC_FRAMES   = 30     # occluded streak length to consider re-anchoring
STUCK_TR_RANGE_M   = 0.005  # if translation range over last STUCK_OCC_FRAMES < 5mm → stuck
LOG_INTERVAL       = 5
# ─────────────────────────────────────────────────────────────────────────────


def score_pose(est, rgb, depth, K):
    pose_np = est.pose_last.cpu().numpy().reshape(1, 4, 4)
    scores, _ = est.scorer.predict(
        mesh=est.mesh, rgb=rgb, depth=depth, K=K,
        ob_in_cams=pose_np, normal_map=None,
        mesh_tensors=est.mesh_tensors, glctx=est.glctx,
        mesh_diameter=est.diameter, get_vis=False,
    )
    return float(scores[0])


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
    depth_scale     = 1.0
    depth_scale_occ = 1.0
    pose            = None
    last_good_R     = None
    baseline_score  = None
    frame0_area     = None
    was_occluded    = False
    occ_streak       = 0
    tr_history       = deque(maxlen=STUCK_OCC_FRAMES)  # recent translations
    stuck_reinit_done = False  # already re-anchored this occlusion period
    n_frozen         = 0
    n_stuck_reinit   = 0

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
            last_good_R    = pose[:3, :3].copy()
            baseline_score = score_pose(est, color, depth * depth_scale, reader.K)
            logging.info(f"Baseline ScoreNet: {baseline_score:.4f}")
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

            pose = est.track_one(rgb=color, depth=d_scaled, K=reader.K,
                                 iteration=args.track_refine_iter)

            # ScoreNet rotation gate: if FP's new pose scores too low, the
            # rotation is suspect — keep last_good_R, accept translation.
            score = score_pose(est, color, d_scaled, reader.K)
            if score < baseline_score * (1.0 - SCORE_DROP_MARGIN) and last_good_R is not None:
                pose[:3, :3] = last_good_R
                n_frozen += 1
                tag = "FROZEN"
            else:
                last_good_R = pose[:3, :3].copy()
                tag = "OK"

            # Occlusion bookkeeping + stuck-while-occluded re-init
            if occluded:
                was_occluded = True
                occ_streak += 1
                # If we've been occluded for a while AND translation has barely
                # moved, the tracker is stuck — force BSD on whatever mask is
                # there. Only fires once per occlusion period: if duck is still
                # not moving and still occluded after, don't BSD again.
                if (not stuck_reinit_done
                        and occ_streak >= STUCK_OCC_FRAMES
                        and mask.any()):
                    tr_arr = np.stack(list(tr_history))
                    tr_range = float(np.linalg.norm(tr_arr.max(0) - tr_arr.min(0)))
                    if tr_range < STUCK_TR_RANGE_M:
                        logging.info(f"[frame {i}] Stuck-occluded re-init "
                                     f"(streak={occ_streak}, tr_range={tr_range*1000:.1f}mm)")
                        pose = binary_search_depth(est, mesh, color,
                                                   mask.astype(bool), reader.K, debug=False)
                        last_good_R = pose[:3, :3].copy()
                        n_stuck_reinit += 1
                        stuck_reinit_done = True
            else:
                occ_streak = 0
                stuck_reinit_done = False
                if was_occluded and mask_area >= RECOVERY_THR * frame0_area:
                    logging.info(f"[frame {i}] Recovery re-init")
                    pose = binary_search_depth(est, mesh, color, mask.astype(bool),
                                               reader.K, debug=False)
                    last_good_R = pose[:3, :3].copy()
                    baseline_score = max(baseline_score,
                                         score_pose(est, color, d_scaled, reader.K))
                    was_occluded = False

        tr_history.append(pose[:3, 3].copy())

        if i % LOG_INTERVAL == 0:
            score_str = f"{score:.3f}" if i > 0 else f"{baseline_score:.3f}"
            logging.info(f"[frame {i}] {tag}  score={score_str}  "
                         f"frozen={n_frozen}  stuck_reinit={n_stuck_reinit}")

        t2 = time.time()
        np.savetxt(f"{debug_dir}/ob_in_cam/{id_str}.txt", pose.reshape(4,4))

        if args.debug >= 1:
            center_pose = pose @ np.linalg.inv(to_origin)
            color = cv2.putText(color, f"fps {int(1/(t2-t1))} {tag} f{n_frozen}",
                                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0,0,255) if tag=="FROZEN" else (255,0,0), 2)
            vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K,
                                thickness=3, transparency=0, is_input_rgb=True)
        if args.debug >= 2:
            imageio.imwrite(f"{debug_dir}/track_vis/{id_str}.png", vis)

