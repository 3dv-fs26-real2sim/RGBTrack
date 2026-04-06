"""
FoundationPose dual-tracker: duck (main object) + hand (for occlusion rotation).

During occlusion, duck rotation is updated using the hand's rotation delta
instead of being frozen. This gives physically plausible rotation estimates
while the hand holds the duck.

Hand-guided rotation stops (and re-init fires) when the hand-duck distance
increases by RELEASE_DIST_DELTA from its minimum seen during occlusion — i.e.,
when the hand lets go and moves away.

Requires:
- test_scene_dir/depth/      — VDA depth PNGs (uint16 mm)
- test_scene_dir/masks/      — SAM2VP duck masks
- test_scene_dir/hand_masks/ — SAM2VP hand masks (seeded from HAND_SEED_FRAME)
"""
import time
from estimater import *
from datareader import *
import argparse
from tools import *
import numpy as np
from scipy.spatial.transform import Rotation as ScipyR

SAVE_VIDEO = False

# ── Settings ───────────────────────────────────────────────────────────────────
OCCLUSION_THRESHOLD = 0.90   # duck mask below this → occluded
RECOVERY_THRESHOLD  = 0.95   # duck mask above this → recovered (re-init)
MAX_ROT_DELTA_DEG   = 3.0    # accept small duck rotation during occlusion (no-hand fallback)

HAND_SEED_FRAME     = 90     # frame where hand mask was initialized
RELEASE_DIST_DELTA  = 0.04   # metres — re-init when hand moves this far from its closest point
DIST_LOG_INTERVAL   = 5      # log hand-duck distance every N frames
# ──────────────────────────────────────────────────────────────────────────────

def rotation_delta_deg(R1, R2):
    R_rel = R1.T @ R2
    angle = np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1))
    return np.degrees(angle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument("--mesh_file", type=str,
                        default=f"{code_dir}/demo_data/mustard0/mesh/textured_simple.obj")
    parser.add_argument("--hand_mesh_file", type=str,
                        default="/work/courses/3dv/team22/foundationpose/data/object/hand/ORCA_v1_simplified.obj")
    parser.add_argument("--test_scene_dir", type=str,
                        default=f"{code_dir}/demo_data/mustard0")
    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--track_refine_iter", type=int, default=1)
    parser.add_argument("--debug", type=int, default=1)
    parser.add_argument("--debug_dir", type=str, default=f"{code_dir}/debug")
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    # ── Duck mesh ──────────────────────────────────────────────────────────────
    mesh = trimesh.load(args.mesh_file)
    mesh.apply_scale(0.001)

    # ── Hand mesh ──────────────────────────────────────────────────────────────
    hand_mesh_raw = trimesh.load(args.hand_mesh_file)
    if hasattr(hand_mesh_raw, 'geometry'):
        hand_mesh = trimesh.util.concatenate(list(hand_mesh_raw.geometry.values()))
    else:
        hand_mesh = hand_mesh_raw
    hand_mesh.apply_scale(0.001)  # mm -> metres

    debug = args.debug
    debug_dir = args.debug_dir
    os.system(f"rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam")

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()

    # Duck estimator
    est = FoundationPose(
        model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
        scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx,
    )
    # Hand estimator (separate instance)
    est_hand = FoundationPose(
        model_pts=hand_mesh.vertices, model_normals=hand_mesh.vertex_normals, mesh=hand_mesh,
        scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=0, glctx=glctx,
    )
    logging.info("estimators initialized")

    reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)

    frame0_mask_area   = None
    depth_scale        = 1.0
    last_good_duck_rot = None
    was_occluded       = False
    hand_pose_last     = None
    hand_rot_last      = None
    hand_rot_delta     = None
    min_occlusion_dist = None   # lowest hand-duck dist seen during current occlusion

    for i in range(len(reader.color_files)):
        color = reader.get_color(i)
        t1 = time.time()

        depth_path = os.path.join(args.test_scene_dir, "depth", f"{reader.id_strs[i]}.png")
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

        mask_path = os.path.join(args.test_scene_dir, "masks", f"{reader.id_strs[i]}.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)

        hand_mask_path = os.path.join(args.test_scene_dir, "hand_masks", f"{reader.id_strs[i]}.png")
        hand_mask_exists = os.path.exists(hand_mask_path)
        if hand_mask_exists:
            hand_mask = cv2.imread(hand_mask_path, cv2.IMREAD_GRAYSCALE)
            hand_mask = (hand_mask > 127).astype(np.uint8)

        # ── Hand tracker ──────────────────────────────────────────────────────
        if i == HAND_SEED_FRAME and hand_mask_exists and hand_mask.sum() > 100:
            logging.info(f"[frame {i}] Initializing hand tracker")
            hand_pose_last = binary_search_depth(est_hand, hand_mesh, color, hand_mask.astype(bool), reader.K, debug=False)
            hand_rot_last = hand_pose_last[:3, :3].copy()
            hand_rot_delta = None
        elif i > HAND_SEED_FRAME and hand_pose_last is not None and hand_mask_exists:
            hand_pose_last = est_hand.track_one_new(
                rgb=color, depth=depth * depth_scale, K=reader.K,
                iteration=args.track_refine_iter, mask=hand_mask
            )
            hand_rot_new = hand_pose_last[:3, :3].copy()
            hand_rot_delta = hand_rot_new @ hand_rot_last.T  # rotation change this frame
            hand_rot_last = hand_rot_new
        else:
            hand_rot_delta = None

        # ── Duck tracker ──────────────────────────────────────────────────────
        if i == 0:
            pose = binary_search_depth(est, mesh, color, mask.astype(bool), reader.K, debug=True)
            logging.info(f"Initial duck pose:\n{pose}")
            frame0_mask_area = float(mask.sum())
            last_good_duck_rot = pose[:3, :3].copy()
            obj_pixels = mask > 0
            vda_z = depth[obj_pixels].mean() if obj_pixels.any() else 1.0
            bsd_z = float(pose[2, 3])
            depth_scale = bsd_z / vda_z if vda_z > 0 else 1.0
            logging.info(f"Depth scale: {depth_scale:.3f}")
        else:
            mask_area = float(mask.sum())
            occluded = (frame0_mask_area > 0) and (mask_area < OCCLUSION_THRESHOLD * frame0_mask_area)

            pose = est.track_one_new(
                rgb=color, depth=depth * depth_scale, K=reader.K,
                iteration=args.track_refine_iter, mask=mask
            )

            # ── Hand-duck distance ─────────────────────────────────────────────
            hand_duck_dist = None
            if hand_pose_last is not None:
                hand_duck_dist = np.linalg.norm(pose[:3, 3] - hand_pose_last[:3, 3])
                if i % DIST_LOG_INTERVAL == 0:
                    logging.info(f"[frame {i}] hand-duck dist: {hand_duck_dist:.4f} m  occluded={occluded}")

            # Detect hand release: dist has grown >RELEASE_DIST_DELTA from occlusion minimum
            hand_released = (
                was_occluded
                and hand_duck_dist is not None
                and min_occlusion_dist is not None
                and (hand_duck_dist - min_occlusion_dist) > RELEASE_DIST_DELTA
            )
            if hand_released:
                logging.info(f"[frame {i}] Hand release detected — dist {hand_duck_dist:.4f} m "
                             f"(min was {min_occlusion_dist:.4f} m)")

            # ── State machine ──────────────────────────────────────────────────
            if occluded and not hand_released:
                # Update rolling minimum distance while hand holds duck
                if hand_duck_dist is not None:
                    if min_occlusion_dist is None or hand_duck_dist < min_occlusion_dist:
                        min_occlusion_dist = hand_duck_dist

                if hand_rot_delta is not None:
                    # Apply hand rotation delta to duck
                    new_rot = hand_rot_delta @ last_good_duck_rot
                    pose[:3, :3] = new_rot
                    last_good_duck_rot = new_rot
                else:
                    # No hand data — small-delta freeze fallback
                    delta = rotation_delta_deg(last_good_duck_rot, pose[:3, :3])
                    if delta <= MAX_ROT_DELTA_DEG:
                        last_good_duck_rot = pose[:3, :3].copy()
                    else:
                        pose[:3, :3] = last_good_duck_rot
                was_occluded = True

            elif hand_released or (was_occluded and mask_area >= RECOVERY_THRESHOLD * frame0_mask_area):
                reason = "hand release" if hand_released else "mask recovered"
                logging.info(f"[frame {i}] Recovery re-init ({reason})")
                pose = binary_search_depth(est, mesh, color, mask.astype(bool), reader.K, debug=False)
                last_good_duck_rot = pose[:3, :3].copy()
                was_occluded = False
                min_occlusion_dist = None

            elif was_occluded:
                pose[:3, :3] = last_good_duck_rot  # frozen until recovered

            else:
                last_good_duck_rot = pose[:3, :3].copy()
                min_occlusion_dist = None

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
