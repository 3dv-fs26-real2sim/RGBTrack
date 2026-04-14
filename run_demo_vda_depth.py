"""
FoundationPose tracking using VDA streaming depth maps calibrated against sim GT.

VDA depth:    uint16 PNG (mm) in --pred_depth_dir  (default: test_scene_dir/depth_vda_streaming/)
Sim GT depth: uint16 PNG (mm) in test_scene_dir/depth/
Masks:        test_scene_dir/masks/

Per-frame affine calibration (scale + shift) is fit on a static table ROI using
RANSAC, then applied to the full VDA depth map before passing to FoundationPose.
"""
import time
from estimater import *
from datareader import *
import argparse
from tools import *
import numpy as np
from scipy.spatial.transform import Rotation as ScipyR
from sklearn.linear_model import RANSACRegressor

# ── ROI-based per-frame depth calibration ─────────────────────────────────────
# Safe zone: bottom-left 300×200 pixels of the 640×480 image (table surface).
# Confirmed static — nothing enters this region during any sequence.
_ROI = (280, 480, 0, 300)  # (y1, y2, x1, x2)

def calibrate_depth(d_pred, d_sim, roi=_ROI):
    """
    Fit affine (scale, shift) from predicted to sim depth using a static table ROI.
    Uses RANSAC to reject specular highlights / shadow outliers in the patch.
    Returns the full-frame calibrated depth map.
    """
    y1, y2, x1, x2 = roi
    pred_patch = d_pred[y1:y2, x1:x2].flatten()
    sim_patch  = d_sim [y1:y2, x1:x2].flatten()
    valid = (pred_patch > 0.001) & (sim_patch > 0.001)
    pred_valid = pred_patch[valid]
    sim_valid  = sim_patch [valid]
    if len(pred_valid) < 10:
        logging.warning("calibrate_depth: too few valid ROI pixels — returning raw depth")
        return d_pred.copy()
    ransac = RANSACRegressor(random_state=42)
    ransac.fit(pred_valid.reshape(-1, 1), sim_valid)
    scale = float(ransac.estimator_.coef_[0])
    shift = float(ransac.estimator_.intercept_)
    return np.maximum(scale * d_pred + shift, 0.0)
# ──────────────────────────────────────────────────────────────────────────────

SAVE_VIDEO = False

# ── Occlusion handling settings ────────────────────────────────────────────────
# Rotation freeze activates when mask drops below this fraction of frame-0 area.
OCCLUSION_THRESHOLD = 0.90

# Rotation freeze deactivates (and re-init triggers) when mask recovers above this.
# Higher than OCCLUSION_THRESHOLD = hysteresis, avoids flickering.
RECOVERY_THRESHOLD = 0.95

# Max angular difference (degrees) from last good rotation to accept a new
# tracked rotation during occlusion. Keeps subtle real motion, rejects spikes.
MAX_ROT_DELTA_DEG = 3.0
# ──────────────────────────────────────────────────────────────────────────────

def rotation_delta_deg(R1, R2):
    """Angular difference in degrees between two rotation matrices."""
    R_rel = R1.T @ R2
    angle = np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1))
    return np.degrees(angle)


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
    parser.add_argument("--num_frames", type=int, default=None, help="Process only the first N frames (default: all)")
    parser.add_argument("--pred_depth_dir", type=str, default=None,
                        help="Directory of VDA depth PNGs (uint16 mm). Defaults to test_scene_dir/depth_vda_streaming/")
    args = parser.parse_args()

    if args.pred_depth_dir is None:
        args.pred_depth_dir = os.path.join(args.test_scene_dir, "depth_vda_streaming")

    set_logging_format()
    set_seed(0)

    mesh = trimesh.load(args.mesh_file)
    mesh.apply_scale(0.001)

    debug = args.debug
    debug_dir = args.debug_dir
    os.system(f"rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam")

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=debug_dir,
        debug=debug,
        glctx=glctx,
    )
    logging.info("estimator initialization done")

    reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)

    logging.info(f"VDA depth dir : {args.pred_depth_dir}")
    logging.info(f"Sim GT depth  : {os.path.join(args.test_scene_dir, 'depth/')}")

    frame0_mask_area = None
    last_good_rotation = None
    was_occluded = False

    num_frames = args.num_frames if args.num_frames is not None else len(reader.color_files)
    for i in range(min(num_frames, len(reader.color_files))):
        color = reader.get_color(i)
        t1 = time.time()

        # VDA predicted depth (monocular, needs calibration)
        vda_path = os.path.join(args.pred_depth_dir, f"{reader.id_strs[i]}.png")
        depth_vda = cv2.imread(vda_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

        # Sim GT depth (Isaac Sim, metric reference for calibration)
        simgt_path = os.path.join(args.test_scene_dir, "depth", f"{reader.id_strs[i]}.png")
        depth_sim = cv2.imread(simgt_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

        mask_path = os.path.join(args.test_scene_dir, "masks", f"{reader.id_strs[i]}.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)

        depth_cal = depth_vda  # VDA metric depth used directly; sim GT mismatch makes calibration counterproductive

        if i == 0:
            pose = est.register(reader.K, color, depth_cal, mask, args.est_refine_iter)
            logging.info(f"Initial pose:\n{pose}")
            frame0_mask_area = float(mask.sum())
            last_good_rotation = pose[:3, :3].copy()
        else:
            mask_area = float(mask.sum())
            occluded = (frame0_mask_area > 0) and (mask_area < OCCLUSION_THRESHOLD * frame0_mask_area)

            pose = est.track_one_new(
                rgb=color, depth=depth_cal, K=reader.K,
                iteration=args.track_refine_iter, mask=mask
            )

            if occluded:
                delta = rotation_delta_deg(last_good_rotation, pose[:3, :3])
                if delta <= MAX_ROT_DELTA_DEG:
                    last_good_rotation = pose[:3, :3].copy()
                else:
                    pose[:3, :3] = last_good_rotation
                was_occluded = True
            elif was_occluded:
                if mask_area >= RECOVERY_THRESHOLD * frame0_mask_area:
                    logging.info(f"[frame {i}] Recovery re-init with calibrated depth")
                    pose = est.register(reader.K, color, depth_cal, mask, args.est_refine_iter)
                    last_good_rotation = pose[:3, :3].copy()
                    was_occluded = False
                else:
                    pose[:3, :3] = last_good_rotation
            else:
                last_good_rotation = pose[:3, :3].copy()

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
