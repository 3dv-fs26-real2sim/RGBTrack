"""
FoundationPose duck tracker — VGGT depth, track_one_new (with duck mask).

Same as run_demo_vda_hand.py but uses track_one_new(mask=mask) every frame
instead of track_one (no mask), for comparison on fast-motion tracking.
"""
import time
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
    parser.add_argument("--depth_dir", type=str, default=None,
                        help="Override depth PNG directory. Defaults to test_scene_dir/depth/")
    parser.add_argument("--depth_dir_occ", type=str, default=None,
                        help="Depth PNG directory to use when occluded. Falls back to --depth_dir if not set.")
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

    reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)

    frame0_mask_area   = None
    depth_scale        = 1.0
    depth_scale_occ    = 1.0
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

            baseline_score = score_current_pose(est, color, depth * depth_scale, reader.K)
            logging.info(f"Baseline ScoreNet score: {baseline_score:.4f}")

        else:
            mask_area = float(mask.sum())
            occluded  = (frame0_mask_area > 0) and (mask_area < OCCLUSION_THRESHOLD * frame0_mask_area)
            d_scaled  = (depth_occ * depth_scale_occ) if occluded else (depth * depth_scale)

            # track_one_new (with duck mask) when visible; track_one when occluded
            if occluded:
                pose = est.track_one(
                    rgb=color, depth=d_scaled, K=reader.K,
                    iteration=args.track_refine_iter,
                )
            else:
                pose = est.track_one_new(
                    rgb=color, depth=d_scaled, K=reader.K,
                    iteration=args.track_refine_iter, mask=mask,
                )

            # ── ScoreNet quality gate (only when occluded) ─────────────────────
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
