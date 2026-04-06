"""
FoundationPose + MediaPipe Hands duck tracker.

Rotation state machine during occlusion:
  - STATIC:  hand not moving → lock rotation (last_good_duck_rot)
  - MOVING:  hand moving (MediaPipe delta > threshold) → accumulate deltas
             from rot_at_move_start. FoundationPose only used for translation.
  - UNOCCLUDED: re-init with binary_search_depth, give full control to FP.

Motion detection: MediaPipe delta magnitude > MOTION_THRESHOLD_DEG for
MOTION_CONFIRM frames → MOVING. Below MOTION_STOP_DEG for STOP_CONFIRM
frames → STATIC again.

One-shot: after first unocclude following a grasp → grasp_done, hand logic off.

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
from mediapipe_hand_tracker import MediaPipeHandTracker

SAVE_VIDEO = False

# ── Settings ───────────────────────────────────────────────────────────────────
OCCLUSION_THRESHOLD  = 0.90   # duck mask below this → occluded
RECOVERY_THRESHOLD   = 0.95   # duck mask above this → recovered (re-init)

RELEASE_CONSEC       = 5      # frames hand must be off mask to fire release

# Motion detection
MOTION_THRESHOLD_DEG = 1.0    # deg/frame — MediaPipe delta above this = moving
MOTION_CONFIRM       = 2      # consecutive frames above threshold to enter MOVING
MOTION_STOP_DEG      = 0.3    # deg/frame — delta below this = stopped
STOP_CONFIRM         = 4      # consecutive frames below stop threshold to enter STATIC

LOG_INTERVAL         = 5
# ──────────────────────────────────────────────────────────────────────────────

def rot_delta_deg(R_delta):
    """Rotation magnitude in degrees of a delta rotation matrix."""
    angle = np.arccos(np.clip((np.trace(R_delta) - 1) / 2, -1, 1))
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
    grasp_done         = False
    hand_released      = False
    grasp_entered      = False
    off_mask_count     = 0

    # Motion state
    is_moving          = False   # currently in MOVING state
    rot_at_move_start  = None    # duck rotation when motion started
    accumulated_rot    = None    # product of all MediaPipe deltas since move start
    moving_count       = 0       # consecutive frames above motion threshold
    stopped_count      = 0       # consecutive frames below stop threshold

    for i in range(len(reader.color_files)):
        color = reader.get_color(i)
        t1    = time.time()

        depth_path = os.path.join(args.test_scene_dir, "depth", f"{reader.id_strs[i]}.png")
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

        mask_path = os.path.join(args.test_scene_dir, "masks", f"{reader.id_strs[i]}.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)

        # ── MediaPipe (every frame, CPU) ──────────────────────────────────────
        hand_rot_delta = mp_hand.update(color)
        hand_on_mask   = mp_hand.on_mask(mask)

        # ── Duck tracker ──────────────────────────────────────────────────────
        if i == 0:
            pose = binary_search_depth(est, mesh, color, mask.astype(bool), reader.K, debug=True)
            logging.info(f"Initial duck pose:\n{pose}")
            frame0_mask_area   = float(mask.sum())
            last_good_duck_rot = pose[:3, :3].copy()
            obj_pixels  = mask > 0
            vda_z       = depth[obj_pixels].mean() if obj_pixels.any() else 1.0
            bsd_z       = float(pose[2, 3])
            depth_scale = bsd_z / vda_z if vda_z > 0 else 1.0
            logging.info(f"Depth scale: {depth_scale:.3f}")
        else:
            mask_area = float(mask.sum())
            occluded  = (frame0_mask_area > 0) and (mask_area < OCCLUSION_THRESHOLD * frame0_mask_area)

            # Always run FP for translation; rotation overridden below when needed
            pose = est.track_one_new(
                rgb=color, depth=depth * depth_scale, K=reader.K,
                iteration=args.track_refine_iter, mask=mask
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
                    logging.info(f"[frame {i}] Release detected")

            # ── Motion detection (only when occluded + grasp active) ───────────
            if occluded and grasp_entered and not hand_released and not grasp_done:
                if hand_rot_delta is not None:
                    deg = rot_delta_deg(hand_rot_delta)
                else:
                    deg = 0.0

                if not is_moving:
                    if deg > MOTION_THRESHOLD_DEG:
                        moving_count += 1
                        stopped_count = 0
                        if moving_count >= MOTION_CONFIRM:
                            is_moving         = True
                            rot_at_move_start = last_good_duck_rot.copy()
                            accumulated_rot   = np.eye(3)
                            moving_count      = 0
                            logging.info(f"[frame {i}] Motion started (delta {deg:.2f} deg)")
                    else:
                        moving_count = 0
                else:  # is_moving
                    if deg < MOTION_STOP_DEG:
                        stopped_count += 1
                        moving_count   = 0
                        if stopped_count >= STOP_CONFIRM:
                            is_moving     = False
                            stopped_count = 0
                            logging.info(f"[frame {i}] Motion stopped")
                    else:
                        stopped_count = 0
            else:
                # Reset motion state when not in active occluded grasp
                if not occluded or grasp_done:
                    is_moving     = False
                    moving_count  = 0
                    stopped_count = 0

            # ── State machine ──────────────────────────────────────────────────
            if occluded:
                if hand_released or grasp_done:
                    # Post-release: freeze rotation
                    pose[:3, :3] = last_good_duck_rot

                elif grasp_entered and is_moving and hand_rot_delta is not None:
                    # MOVING: accumulate MediaPipe deltas from move-start rotation
                    accumulated_rot = hand_rot_delta @ accumulated_rot
                    new_rot = accumulated_rot @ rot_at_move_start
                    pose[:3, :3] = new_rot
                    last_good_duck_rot = new_rot

                elif grasp_entered:
                    # STATIC during grasp: lock rotation
                    pose[:3, :3] = last_good_duck_rot

                else:
                    # No grasp data: lock rotation
                    pose[:3, :3] = last_good_duck_rot

                was_occluded = True

            elif was_occluded and mask_area >= RECOVERY_THRESHOLD * frame0_mask_area:
                logging.info(f"[frame {i}] Recovery re-init")
                pose = binary_search_depth(est, mesh, color, mask.astype(bool), reader.K, debug=False)
                last_good_duck_rot = pose[:3, :3].copy()
                was_occluded  = False
                is_moving     = False
                moving_count  = 0
                stopped_count = 0
                accumulated_rot   = None
                rot_at_move_start = None
                if hand_released:
                    grasp_done    = True
                    hand_released = False
                    logging.info(f"[frame {i}] Grasp cycle complete — hand logic disabled")

            elif was_occluded:
                # Partially recovered — keep locked rotation
                pose[:3, :3] = last_good_duck_rot

            else:
                # Fully unoccluded — FP has full control
                last_good_duck_rot = pose[:3, :3].copy()

            if i % LOG_INTERVAL == 0:
                logging.info(f"[frame {i}] occluded={occluded}  on_mask={hand_on_mask}  "
                             f"is_moving={is_moving}  grasp_entered={grasp_entered}  "
                             f"hand_released={hand_released}  grasp_done={grasp_done}")

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
