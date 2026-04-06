"""
FoundationPose + MediaPipe Hands duck tracker.

Rotation state machine during occlusion:
  - STATIC:  hand not moving → lock rotation (last_good_duck_rot)
  - MOVING:  hand moving (MediaPipe delta > threshold) → accumulate deltas
             from rot_at_move_start. Optical flow used when MP not available.
  - UNOCCLUDED: re-init with binary_search_depth, give full control to FP.

Motion detection: MediaPipe delta magnitude > MOTION_THRESHOLD_DEG for
MOTION_CONFIRM frames → MOVING. Below MOTION_STOP_DEG for STOP_CONFIRM
frames → STATIC again.

One-shot: after first unocclude following a grasp → grasp_done, hand logic off.

Optical flow (option 4): goodFeaturesToTrack + LK on duck pixels while
unoccluded; solvePnPRansac during occlusion to recover per-frame rotation
delta. Used as fill-in on non-MediaPipe frames in MOVING state.

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
MAX_MID_AIR_REINITS  = 1      # max re-inits allowed while still occluded

# Optical flow settings
OF_MAX_PTS           = 100    # max keypoints to track
OF_QUALITY           = 0.01   # goodFeaturesToTrack quality level
OF_MIN_DIST          = 5      # min distance between keypoints (px)

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
    accumulated_rot    = None    # product of all deltas since move start
    moving_count       = 0       # consecutive frames above motion threshold
    stopped_count      = 0       # consecutive frames below stop threshold
    mid_air_reinit_count = 0     # number of mid-air re-inits used so far

    # Optical flow state
    of_prev_gray  = None         # grayscale of previous frame
    of_pts        = None         # current tracked 2D keypoints (N, 2) float32
    of_3d_pts     = None         # 3D camera-space positions at detection time (N, 3) float32
    of_R_prev     = None         # previous solvePnP result (for computing delta)

    for i in range(len(reader.color_files)):
        color = reader.get_color(i)
        t1    = time.time()

        depth_path = os.path.join(args.test_scene_dir, "depth", f"{reader.id_strs[i]}.png")
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

        mask_path = os.path.join(args.test_scene_dir, "masks", f"{reader.id_strs[i]}.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)

        # ── MediaPipe (every 3 frames, CPU) ───────────────────────────────────
        if i % 3 == 0:
            hand_rot_delta = mp_hand.update(color)
            hand_on_mask   = mp_hand.on_mask(mask)
        else:
            hand_rot_delta = None
            hand_on_mask   = mp_hand.on_mask(mask)  # cheap, keep every frame

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
            gray      = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)

            # Always run FP for translation; rotation overridden below when needed
            pose = est.track_one_new(
                rgb=color, depth=depth * depth_scale, K=reader.K,
                iteration=args.track_refine_iter, mask=mask
            )

            # ── Optical flow: refresh keypoints on clean duck view ─────────────
            if not occluded and not was_occluded:
                roi      = (mask > 0).astype(np.uint8) * 255
                new_pts  = cv2.goodFeaturesToTrack(
                    gray, maxCorners=OF_MAX_PTS, qualityLevel=OF_QUALITY,
                    minDistance=OF_MIN_DIST, mask=roi)
                if new_pts is not None:
                    pts   = new_pts.reshape(-1, 2)
                    d_sc  = depth * depth_scale
                    v_pts, v_3d = [], []
                    for pt in pts:
                        u = int(np.clip(round(pt[0]), 0, d_sc.shape[1] - 1))
                        v = int(np.clip(round(pt[1]), 0, d_sc.shape[0] - 1))
                        d = float(d_sc[v, u])
                        if d > 0.1:
                            X = (u - reader.K[0, 2]) * d / reader.K[0, 0]
                            Y = (v - reader.K[1, 2]) * d / reader.K[1, 1]
                            v_pts.append(pt)
                            v_3d.append([X, Y, d])
                    if v_pts:
                        of_pts    = np.array(v_pts, dtype=np.float32)
                        of_3d_pts = np.array(v_3d,  dtype=np.float32)
                        of_R_prev = None  # reset pose reference for delta computation
                of_prev_gray = gray.copy()

            # ── Optical flow: track + estimate rotation delta ──────────────────
            of_rot_delta = None
            if occluded and of_pts is not None and of_prev_gray is not None and len(of_pts) >= 4:
                new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    of_prev_gray, gray,
                    of_pts.reshape(-1, 1, 2).astype(np.float32), None)
                if new_pts is not None and status is not None:
                    ok       = status.ravel().astype(bool)
                    good_2d  = new_pts.reshape(-1, 2)[ok]
                    good_3d  = of_3d_pts[ok]
                    if len(good_3d) >= 4:
                        try:
                            ret, rvec, tvec, inliers = cv2.solvePnPRansac(
                                good_3d, good_2d,
                                reader.K.astype(np.float64), None)
                            if ret and inliers is not None and len(inliers) >= 4:
                                R_curr, _ = cv2.Rodrigues(rvec)
                                if of_R_prev is not None:
                                    of_rot_delta = R_curr @ of_R_prev.T
                                of_R_prev = R_curr
                        except cv2.error:
                            pass
                    of_pts    = good_2d
                    of_3d_pts = good_3d
                of_prev_gray = gray.copy()
            elif not occluded:
                of_prev_gray = gray.copy()

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
                            if mid_air_reinit_count < MAX_MID_AIR_REINITS:
                                logging.info(f"[frame {i}] Motion stopped — mid-air re-init "
                                             f"({mid_air_reinit_count + 1}/{MAX_MID_AIR_REINITS})")
                                pose = binary_search_depth(
                                    est, mesh, color, mask.astype(bool), reader.K, debug=2)
                                last_good_duck_rot   = pose[:3, :3].copy()
                                mid_air_reinit_count += 1
                            else:
                                logging.info(f"[frame {i}] Motion stopped — max mid-air re-inits "
                                             f"reached, locking rotation")
                            accumulated_rot   = None
                            rot_at_move_start = None
                            of_R_prev         = None  # reset OF delta reference
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

                elif grasp_entered and is_moving:
                    # MOVING: prefer OF delta (direct duck measurement),
                    #         fall back to MediaPipe palm delta
                    rot_delta = of_rot_delta if of_rot_delta is not None else hand_rot_delta
                    if rot_delta is not None:
                        accumulated_rot    = rot_delta @ accumulated_rot
                        new_rot            = accumulated_rot @ rot_at_move_start
                        pose[:3, :3]       = new_rot
                        last_good_duck_rot = new_rot
                    else:
                        pose[:3, :3] = last_good_duck_rot

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
                last_good_duck_rot   = pose[:3, :3].copy()
                was_occluded         = False
                is_moving            = False
                moving_count         = 0
                stopped_count        = 0
                mid_air_reinit_count = 0
                accumulated_rot      = None
                rot_at_move_start    = None
                of_R_prev            = None
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
                of_pts_n = len(of_pts) if of_pts is not None else 0
                logging.info(f"[frame {i}] occluded={occluded}  on_mask={hand_on_mask}  "
                             f"is_moving={is_moving}  grasp_entered={grasp_entered}  "
                             f"hand_released={hand_released}  grasp_done={grasp_done}  "
                             f"of_pts={of_pts_n}  of_delta={'yes' if of_rot_delta is not None else 'no'}")

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
