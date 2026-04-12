"""
FoundationPose duck tracker — ScoreNet rotation quality gate.

ScoreNet (already loaded in est.scorer) is called every frame after
track_one_new. The score at frame 0 is recorded as the baseline.
When the score drops below baseline * (1 - SCORE_DROP_MARGIN), the
rotation update is rejected and the last accepted rotation is held.
Translation is always accepted from FoundationPose.

Recovery re-init (binary_search_depth) fires when the duck mask
recovers to 0.95 of its frame-0 area after an occlusion period.

Requires:
- test_scene_dir/depth/  — VDA depth PNGs (uint16 mm)
- test_scene_dir/masks/  — SAM2VP duck masks
"""
import time
import torch
from estimater import *
from datareader import *
import argparse
from tools import *
import numpy as np
try:
    from metric3d_wrapper import Metric3DWrapper
except ImportError:
    Metric3DWrapper = None

try:
    from depth_pro_wrapper import DepthProWrapper
except ImportError:
    DepthProWrapper = None

try:
    import h5py
    from hand_mask_renderer import HandMaskRenderer
    _HAND_CALIB_AVAILABLE = True
except ImportError:
    _HAND_CALIB_AVAILABLE = False

# ── Settings ───────────────────────────────────────────────────────────────────
OCCLUSION_THRESHOLD = 0.90   # duck mask below this → occluded
RECOVERY_THRESHOLD  = 0.95   # duck mask above this → recovered (re-init)

SCORE_DROP_MARGIN   = 0.35   # reject rotation if score < baseline * (1 - margin)

LOG_INTERVAL        = 5
# ──────────────────────────────────────────────────────────────────────────────


def score_current_pose(est, rgb, depth, K):
    """Run ScoreNet on est.pose_last. Returns scalar float score."""
    pose_np = est.pose_last.cpu().numpy().reshape(1, 4, 4)
    scores, _ = est.scorer.predict(
        mesh=est.mesh,
        rgb=rgb,
        depth=depth,
        K=K,
        ob_in_cams=pose_np,
        normal_map=None,
        mesh_tensors=est.mesh_tensors,
        glctx=est.glctx,
        mesh_diameter=est.diameter,
        get_vis=False,
    )
    return float(scores[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument("--mesh_file", type=str,
                        default=f"{code_dir}/demo_data/mustard0/mesh/textured_simple.obj")
    parser.add_argument("--test_scene_dir", type=str,
                        default=f"{code_dir}/demo_data/mustard0")
    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--track_refine_iter", type=int, default=1)
    parser.add_argument("--debug", type=int, default=1)
    parser.add_argument("--debug_dir", type=str, default=f"{code_dir}/debug")
    parser.add_argument("--metric3d_ckpt", type=str, default=None,
                        help="Path to Metric3D checkpoint. If set, uses Metric3D instead of VDA depth PNGs.")
    parser.add_argument("--depth_pro_ckpt", type=str, default=None,
                        help="Path to Depth Pro checkpoint. If set, uses Depth Pro instead of VDA depth PNGs.")
    parser.add_argument("--depth_dir", type=str, default=None,
                        help="Override depth PNG directory (e.g. depth_pro/ for pre-generated maps). Defaults to test_scene_dir/depth/")
    parser.add_argument("--depth_dir_occ", type=str, default=None,
                        help="Depth PNG directory to use when occluded. Falls back to --depth_dir if not set.")
    parser.add_argument("--h5_file", type=str, default=None,
                        help="H5 file with observations/qpos_arm and qpos_hand for affine depth calibration.")
    parser.add_argument("--urdf_file", type=str, default=None,
                        help="URDF path for HandMaskRenderer (affine depth calibration).")
    parser.add_argument("--hand_calib_frame", type=int, default=200,
                        help="Frame index where hand is fully grasping duck — used as second affine anchor.")
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    mesh = trimesh.load(args.mesh_file)
    mesh.apply_scale(0.001)

    debug     = args.debug
    debug_dir = args.debug_dir
    os.system(f"rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam")

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    scorer  = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx   = dr.RasterizeCudaContext()

    est = FoundationPose(
        model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
        scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx,
    )
    logging.info("estimator initialized")

    metric3d = None
    if args.metric3d_ckpt:
        assert Metric3DWrapper is not None, "metric3d_wrapper not found"
        metric3d = Metric3DWrapper(checkpoint_path=args.metric3d_ckpt)
        logging.info("Metric3D loaded")
    elif args.depth_pro_ckpt:
        assert DepthProWrapper is not None, "depth_pro_wrapper not found"
        metric3d = DepthProWrapper(checkpoint_path=args.depth_pro_ckpt)
        logging.info("Depth Pro loaded")

    reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)

    # ── Affine depth calibration (optional, needs --h5_file + --urdf_file) ──────
    hand_renderer  = None
    h5_qpos_arm    = None
    h5_qpos_hand   = None
    if args.h5_file and args.urdf_file and _HAND_CALIB_AVAILABLE:
        import h5py as _h5py
        with _h5py.File(args.h5_file, "r") as f:
            h5_qpos_arm  = f["observations/qpos_arm"][()]
            h5_qpos_hand = f["observations/qpos_hand"][()]
        hand_renderer = HandMaskRenderer(urdf_path=args.urdf_file, cam_K=reader.K)
        logging.info(f"HandMaskRenderer ready — affine calib at frame {args.hand_calib_frame}")

    frame0_mask_area   = None
    depth_scale        = 1.0
    depth_scale_occ    = 1.0
    depth_offset       = 0.0          # affine offset: depth_corrected = scale*raw + offset
    raw_z_duck_frame0  = None         # stored for affine fit at hand_calib_frame
    true_z_duck_frame0 = None
    baseline_score     = None
    last_good_duck_rot = None
    was_occluded       = False

    depth_dir_vis = args.depth_dir or os.path.join(args.test_scene_dir, "depth")
    depth_dir_occ = args.depth_dir_occ or depth_dir_vis

    def load_depth_png(d_dir, id_str):
        path = os.path.join(d_dir, f"{id_str}.png")
        return cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

    for i in range(len(reader.color_files)):
        color = reader.get_color(i)
        t1    = time.time()

        if metric3d is not None:
            depth     = metric3d.estimate(color, reader.K)
            depth_occ = depth
        else:
            depth     = load_depth_png(depth_dir_vis, reader.id_strs[i])
            depth_occ = load_depth_png(depth_dir_occ, reader.id_strs[i])

        mask_path = os.path.join(args.test_scene_dir, "masks", f"{reader.id_strs[i]}.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)

        if i == 0:
            pose = binary_search_depth(est, mesh, color, mask.astype(bool), reader.K, debug=True)
            logging.info(f"Initial duck pose:\n{pose}")
            frame0_mask_area   = float(mask.sum())
            last_good_duck_rot = pose[:3, :3].copy()
            obj_pixels  = mask > 0
            bsd_z       = float(pose[2, 3])
            vda_z       = depth[obj_pixels].mean() if obj_pixels.any() else 1.0
            depth_scale = bsd_z / vda_z if vda_z > 0 else 1.0
            occ_z       = depth_occ[obj_pixels].mean() if obj_pixels.any() else 1.0
            depth_scale_occ = bsd_z / occ_z if occ_z > 0 else 1.0
            logging.info(f"Depth scale (vis): {depth_scale:.3f}  (occ): {depth_scale_occ:.3f}")

            # Store duck anchor for later affine calibration
            raw_z_duck_frame0  = float(vda_z)
            true_z_duck_frame0 = float(bsd_z)

            # Record ScoreNet baseline at init
            baseline_score = score_current_pose(est, color, depth * depth_scale, reader.K)
            logging.info(f"Baseline ScoreNet score: {baseline_score:.4f}")

        else:
            mask_area = float(mask.sum())
            occluded  = (frame0_mask_area > 0) and (mask_area < OCCLUSION_THRESHOLD * frame0_mask_area)

            # ── Affine depth calibration at hand_calib_frame ──────────────────
            # Once the hand is fully grasping the duck, fit a two-point affine
            # (scale + offset) using duck frame-0 BSD and hand FK as anchors.
            if (hand_renderer is not None
                    and i == args.hand_calib_frame
                    and raw_z_duck_frame0 is not None
                    and h5_qpos_arm is not None):
                h_mask = hand_renderer.render(h5_qpos_arm[i], h5_qpos_hand[i])
                h_pixels = h_mask > 127
                true_z_hand = hand_renderer.get_wrist_z_in_cam(h5_qpos_arm[i], h5_qpos_hand[i])
                raw_z_hand  = float(depth[h_pixels].mean()) if h_pixels.any() else None
                if raw_z_hand and true_z_hand and abs(raw_z_hand - raw_z_duck_frame0) > 0.02:
                    A = np.array([[raw_z_duck_frame0, 1.0],
                                  [raw_z_hand,        1.0]])
                    b_vec = np.array([true_z_duck_frame0, true_z_hand])
                    new_scale, new_offset = np.linalg.solve(A, b_vec)
                    if 0.5 < new_scale < 3.0:   # sanity check
                        depth_scale  = new_scale
                        depth_offset = new_offset
                        depth_scale_occ = new_scale
                        logging.info(
                            f"[frame {i}] Affine calib: scale={depth_scale:.4f}  "
                            f"offset={depth_offset:.4f}m  "
                            f"(raw_duck={raw_z_duck_frame0:.3f} true_duck={true_z_duck_frame0:.3f}  "
                            f"raw_hand={raw_z_hand:.3f} true_hand={true_z_hand:.3f})")
                    else:
                        logging.warning(f"[frame {i}] Affine calib rejected (scale={new_scale:.3f}), keeping pure scale")
                else:
                    logging.warning(
                        f"[frame {i}] Affine calib skipped: "
                        f"raw_z_hand={raw_z_hand}  true_z_hand={true_z_hand}  "
                        f"depth_diff={abs((raw_z_hand or 0) - raw_z_duck_frame0):.3f}")

            d_scaled  = (depth_occ * depth_scale_occ + depth_offset) if occluded else (depth * depth_scale + depth_offset)

            # ── FP++ translation correction (from FoundationPose-plus-plus) ───
            # Back-project mask centroid to correct pose_last xy before FP
            # refinement, keeping z from the current pose (prevents crop drift).
            if not occluded and mask.any():
                vs, us = np.where(mask > 0)
                uc = float((us.min() + us.max()) / 2.0)
                vc = float((vs.min() + vs.max()) / 2.0)
                K = reader.K
                pl = est.pose_last if est.pose_last.dim() == 2 else est.pose_last[0]
                tz = float(pl[2, 3])
                if tz > 0.01:
                    tx = (uc - K[0, 2]) * tz / K[0, 0]
                    ty = (vc - K[1, 2]) * tz / K[1, 1]
                    est.pose_last = est.pose_last.clone()
                    if est.pose_last.dim() == 3:
                        est.pose_last[0, 0, 3] = tx
                        est.pose_last[0, 1, 3] = ty
                    else:
                        est.pose_last[0, 3] = tx
                        est.pose_last[1, 3] = ty

            pose = est.track_one(
                rgb=color, depth=d_scaled, K=reader.K,
                iteration=args.track_refine_iter,
            )

            # ── ScoreNet quality gate (only when occluded — saves ~half the fps) ─
            if occluded:
                current_score = score_current_pose(est, color, d_scaled, reader.K)
                score_thresh  = baseline_score * (1.0 - SCORE_DROP_MARGIN)
                rot_accepted  = current_score >= score_thresh
                if not rot_accepted:
                    pose[:3, :3] = last_good_duck_rot
                else:
                    last_good_duck_rot = pose[:3, :3].copy()
            else:
                current_score = None
                rot_accepted  = True
                last_good_duck_rot = pose[:3, :3].copy()

            # ── Recovery re-init after occlusion ──────────────────────────────
            if occluded:
                was_occluded = True
            elif was_occluded:
                if mask_area >= RECOVERY_THRESHOLD * frame0_mask_area:
                    logging.info(f"[frame {i}] Recovery re-init")
                    pose = binary_search_depth(
                        est, mesh, color, mask.astype(bool), reader.K, debug=False)
                    last_good_duck_rot = pose[:3, :3].copy()
                    # rescore after re-init to keep baseline meaningful
                    baseline_score = max(
                        baseline_score,
                        score_current_pose(est, color, d_scaled, reader.K))
                    was_occluded = False

            if i % LOG_INTERVAL == 0:
                score_str = f"{current_score:.4f}" if current_score is not None else "n/a"
                logging.info(f"[frame {i}] score={score_str}  "
                             f"accepted={rot_accepted}  occluded={occluded}")

        t2 = time.time()
        os.makedirs(f"{debug_dir}/ob_in_cam", exist_ok=True)
        np.savetxt(f"{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt", pose.reshape(4, 4))

        if debug >= 1:
            center_pose = pose @ np.linalg.inv(to_origin)
            color = cv2.putText(color, f"fps {int(1/(t2-t1))}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K,
                                thickness=3, transparency=0, is_input_rgb=True)

        if debug >= 2:
            os.makedirs(f"{debug_dir}/track_vis", exist_ok=True)
            imageio.imwrite(f"{debug_dir}/track_vis/{reader.id_strs[i]}.png", vis)
