"""
FoundationPose + MediaPipe Hands duck tracker.

During occlusion, if MediaPipe detects hand landmarks on the duck mask
(confirmed grasp), duck rotation is updated using the palm rotation delta.

Grasp logic fires only once per video (one-shot).

Release: hand landmarks no longer overlap duck mask for RELEASE_CONSEC frames.
After release: mask-based recovery re-init, then hand logic disabled forever.

Requires:
- test_scene_dir/depth/  — VDA depth PNGs (uint16 mm)
- test_scene_dir/masks/  — SAM2VP duck masks
"""
import time
from estimater import *
from datareader import *
import argparse
from tools import *
import numpy as np
from scipy.spatial.transform import Rotation as ScipyR
from mediapipe_hand_tracker import MediaPipeHandTracker

SAVE_VIDEO = False

# ── Settings ───────────────────────────────────────────────────────────────────
OCCLUSION_THRESHOLD  = 0.90   # duck mask below this → occluded
RECOVERY_THRESHOLD   = 0.95   # duck mask above this → recovered (re-init)

RELEASE_CONSEC       = 5      # frames hand must be off mask to fire release
ROT_AVG_WINDOW       = 3      # frames to average before applying rotation
CONSISTENCY_WINDOW   = 3      # frames of history to establish trend
CONSISTENCY_MIN_DOT  = 0.7    # min axis alignment to be considered consistent
CLIP_INCONSISTENT    = 1.0    # max deg/frame when motion is inconsistent with trend
LOG_INTERVAL         = 5      # log grasp state every N frames
# ──────────────────────────────────────────────────────────────────────────────

def average_rotations(R_ref, R_list):
    vecs = [ScipyR.from_matrix(R_ref.T @ R).as_rotvec() for R in R_list]
    return R_ref @ ScipyR.from_rotvec(np.mean(vecs, axis=0)).as_matrix()

def clip_rotation_consistent(R_prev, R_new, max_deg, vel_history):
    R_rel  = R_prev.T @ R_new
    angle  = np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1))
    rotvec = ScipyR.from_matrix(R_rel).as_rotvec()

    allowed = max_deg
    if len(vel_history) >= 2:
        trend      = np.mean(vel_history[-CONSISTENCY_WINDOW:], axis=0)
        trend_norm = np.linalg.norm(trend)
        rv_norm    = np.linalg.norm(rotvec)
        if trend_norm > 1e-6 and rv_norm > 1e-6:
            dot = np.dot(rotvec, trend) / (rv_norm * trend_norm)
            if dot < CONSISTENCY_MIN_DOT:
                allowed = CLIP_INCONSISTENT

    vel_history.append(rotvec.copy())
    if len(vel_history) > CONSISTENCY_WINDOW:
        vel_history.pop(0)

    if np.degrees(angle) <= allowed:
        return R_prev @ ScipyR.from_matrix(R_rel).as_matrix()
    t = np.radians(allowed) / angle if angle > 1e-9 else 0.0
    return R_prev @ ScipyR.from_rotvec(rotvec * t).as_matrix()


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
    mp_hand = MediaPipeHandTracker()
    logging.info("estimator + MediaPipe initialized")

    reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)

    frame0_mask_area   = None
    depth_scale        = 1.0
    last_good_duck_rot = None
    was_occluded       = False

    # Grasp state (one-shot)
    grasp_done         = False  # hand logic permanently disabled after first cycle
    hand_released      = False  # release detected, waiting for mask recovery
    grasp_entered      = False  # True once hand landmarks detected on duck mask
    off_mask_count     = 0      # consecutive frames hand has been off the mask
    vel_history        = []
    raw_rot_buffer     = []

    for i in range(len(reader.color_files)):
        color = reader.get_color(i)
        t1    = time.time()

        depth_path = os.path.join(args.test_scene_dir, "depth", f"{reader.id_strs[i]}.png")
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

        mask_path = os.path.join(args.test_scene_dir, "masks", f"{reader.id_strs[i]}.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)

        # ── Hand mask (for subtraction during grasp) ──────────────────────────
        hand_mask_path = os.path.join(args.test_scene_dir, "hand_masks", f"{reader.id_strs[i]}.png")
        hand_mask = None
        if os.path.exists(hand_mask_path):
            hm = cv2.imread(hand_mask_path, cv2.IMREAD_GRAYSCALE)
            if hm is not None:
                hand_mask = (hm > 127).astype(np.uint8)

        # ── MediaPipe (every frame, CPU) ──────────────────────────────────────
        hand_rot_delta = mp_hand.update(color)
        hand_on_mask   = mp_hand.on_mask(mask)

        # ── Duck tracker ──────────────────────────────────────────────────────
        if i == 0:
            pose = binary_search_depth(est, mesh, color, mask.astype(bool), reader.K, debug=True)
            logging.info(f"Initial duck pose:\n{pose}")
            frame0_mask_area = float(mask.sum())
            last_good_duck_rot = pose[:3, :3].copy()
            obj_pixels = mask > 0
            vda_z      = depth[obj_pixels].mean() if obj_pixels.any() else 1.0
            bsd_z      = float(pose[2, 3])
            depth_scale = bsd_z / vda_z if vda_z > 0 else 1.0
            logging.info(f"Depth scale: {depth_scale:.3f}")
        else:
            mask_area = float(mask.sum())
            occluded  = (frame0_mask_area > 0) and (mask_area < OCCLUSION_THRESHOLD * frame0_mask_area)

            # Subtract hand from duck mask during active grasp
            if grasp_entered and not grasp_done and hand_mask is not None:
                track_mask = np.logical_and(mask, np.logical_not(hand_mask)).astype(np.uint8)
            else:
                track_mask = mask

            pose = est.track_one_new(
                rgb=color, depth=depth * depth_scale, K=reader.K,
                iteration=args.track_refine_iter, mask=track_mask
            )

            # ── Grasp / release detection ──────────────────────────────────────
            if not grasp_done and not hand_released:
                if hand_on_mask:
                    grasp_entered  = True
                    off_mask_count = 0
                elif grasp_entered:
                    off_mask_count += 1

                if grasp_entered and off_mask_count >= RELEASE_CONSEC:
                    hand_released = True
                    logging.info(f"[frame {i}] Release detected (hand off mask {off_mask_count} frames)")

            if i % LOG_INTERVAL == 0:
                logging.info(f"[frame {i}] occluded={occluded}  on_mask={hand_on_mask}  "
                             f"grasp_entered={grasp_entered}  hand_released={hand_released}  grasp_done={grasp_done}")

            # ── State machine ──────────────────────────────────────────────────
            if occluded:
                if hand_released or grasp_done:
                    # Post-release: average + clip
                    raw_rot_buffer.append(pose[:3, :3].copy())
                    if len(raw_rot_buffer) > ROT_AVG_WINDOW:
                        raw_rot_buffer.pop(0)
                    avg_rot = average_rotations(last_good_duck_rot, raw_rot_buffer)
                    clipped = clip_rotation_consistent(last_good_duck_rot, avg_rot, 5.0, vel_history)
                    pose[:3, :3] = clipped
                    last_good_duck_rot = clipped
                elif grasp_entered and hand_rot_delta is not None:
                    # Active grasp — apply MediaPipe palm delta, clipped to 2 deg/frame
                    new_rot = hand_rot_delta @ last_good_duck_rot
                    clipped = clip_rotation_consistent(last_good_duck_rot, new_rot, 2.0, vel_history)
                    pose[:3, :3] = clipped
                    last_good_duck_rot = clipped
                else:
                    # Occluded, no grasp data — average + clip fallback
                    raw_rot_buffer.append(pose[:3, :3].copy())
                    if len(raw_rot_buffer) > ROT_AVG_WINDOW:
                        raw_rot_buffer.pop(0)
                    avg_rot = average_rotations(last_good_duck_rot, raw_rot_buffer)
                    clipped = clip_rotation_consistent(last_good_duck_rot, avg_rot, 5.0, vel_history)
                    pose[:3, :3] = clipped
                    last_good_duck_rot = clipped
                was_occluded = True

            elif was_occluded and mask_area >= RECOVERY_THRESHOLD * frame0_mask_area:
                logging.info(f"[frame {i}] Recovery re-init")
                pose = binary_search_depth(est, mesh, color, mask.astype(bool), reader.K, debug=False)
                last_good_duck_rot = pose[:3, :3].copy()
                was_occluded = False
                raw_rot_buffer.clear()
                vel_history.clear()
                if hand_released:
                    grasp_done    = True
                    hand_released = False
                    logging.info(f"[frame {i}] Grasp cycle complete — hand logic disabled")

            elif was_occluded:
                raw_rot_buffer.append(pose[:3, :3].copy())
                if len(raw_rot_buffer) > ROT_AVG_WINDOW:
                    raw_rot_buffer.pop(0)
                avg_rot = average_rotations(last_good_duck_rot, raw_rot_buffer)
                clipped = clip_rotation_consistent(last_good_duck_rot, avg_rot, 5.0, vel_history)
                pose[:3, :3] = clipped
                last_good_duck_rot = clipped

            else:
                last_good_duck_rot = pose[:3, :3].copy()

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

    mp_hand.close()
