"""
FoundationPose duck tracker — palm-anchored, flow-driven state machine.

Occlusion is NOT checked. State is driven entirely by optical flow on the
duck mask, with a debounce buffer to avoid flapping.

States (STATIC ↔ MOVING):
  * STATIC: pose frozen. FP is NOT run per-frame.
  * MOVING: pose = T_cam_palm[t] @ inv(T_cam_palm[t0]) @ T_cam_obj[t0]
    where t0 = last STATIC→MOVING transition frame.

Transitions:
  * STATIC → MOVING: anchor palm_inv at this frame (anchor_pose = last frozen).
  * MOVING → STATIC: run FP once to re-acquire the pose (it may have drifted
    during palm-delta propagation), then freeze.

Hysteresis:
  * STATIC → MOVING requires BUFFER_START_FRAMES consecutive frames with
    flow_med > MOVE_FLOW_START_PX.
  * MOVING → STATIC requires BUFFER_STOP_FRAMES consecutive frames with
    flow_med < MOVE_FLOW_STOP_PX.
Unreliable flow signal (mask too small, too few trackable points) is
sticky — the buffer is neither advanced nor reset, so a brief full-
occlusion blackout does not derail the state.

Init (frame 0):
  * binary_search_depth → first pose. State = STATIC. Anchor set.

Inputs:
  --test_scene_dir/depth/       VDA depth PNGs (uint16 mm) — used only for
                                the one-shot FP re-acquire at transitions.
  --test_scene_dir/masks/       SAM2VP duck masks.
  --palm_poses_npz              (N,4,4) T_cam_palm per RGB frame (Aria frame).

Assumes the palm NPZ is 1:1 aligned with the RGB frame index. If not, pass
a remapping via --palm_frame_offset / --palm_frame_stride.
"""
import time
import os
import argparse

import cv2
import numpy as np
import trimesh
import imageio

from estimater import *
from datareader import *
from tools import *


# ── Settings ───────────────────────────────────────────────────────────────────
# Occlusion is NOT used. Optical-flow drives STATIC/MOVING transitions.
# Entry is sensitive (original behavior). Exit is strict: must have been MOVING
# a while AND flow must drop to near-zero for a long confirmation window.
MOVE_FLOW_START_PX    = 1.5    # flow > this (sustained) → enter MOVING
MOVE_FLOW_STOP_PX     = 0.3    # flow < this (sustained) → exit MOVING
BUFFER_START_FRAMES   = 5      # consecutive frames above START → flip to MOVING
BUFFER_STOP_FRAMES    = 15     # consecutive frames below STOP → flip to STATIC
MIN_MOVING_FRAMES     = 15     # require MOVING to last at least this many frames
                               # before STOP is allowed (ignores brief dips at entry)

MOVE_MIN_TRACK_PTS    = 8      # below this, flow signal considered unreliable
MOVE_MIN_MASK_AREA    = 400    # below this, mask too small to extract features

LOG_INTERVAL          = 5
# ──────────────────────────────────────────────────────────────────────────────


def extract_edge_features(gray: np.ndarray, mask: np.ndarray,
                          max_corners: int = 200) -> np.ndarray | None:
    """goodFeaturesToTrack restricted to duck mask edge region."""
    if mask.sum() < MOVE_MIN_MASK_AREA:
        return None
    # Focus on the mask boundary (and its neighborhood) where textured edges live.
    edge = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_GRADIENT,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    pts = cv2.goodFeaturesToTrack(
        gray, maxCorners=max_corners, qualityLevel=0.01,
        minDistance=4, mask=edge,
    )
    return pts


def compute_mask_flow(prev_gray: np.ndarray | None, gray: np.ndarray,
                      prev_mask: np.ndarray | None,
                      mask: np.ndarray) -> float | None:
    """LK optical flow on previous-frame edge points. Returns median flow (px).

    Returns None if the signal is unreliable (mask too small, too few features,
    or LK failed). Caller should treat None as "no evidence either way".
    """
    if prev_gray is None or prev_mask is None:
        return None
    p0 = extract_edge_features(prev_gray, prev_mask)
    if p0 is None or len(p0) < MOVE_MIN_TRACK_PTS:
        return None
    p1, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, p0, None,
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if p1 is None:
        return None
    good = status.ravel() == 1
    if good.sum() < MOVE_MIN_TRACK_PTS:
        return None
    disp = np.linalg.norm(p1[good] - p0[good], axis=2).ravel()
    return float(np.median(disp))


def resolve_palm_idx(i: int, offset: int, stride: int, n_palm: int) -> int:
    idx = offset + i * stride
    return max(0, min(n_palm - 1, idx))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument("--mesh_file", type=str,
                        default=f"{code_dir}/demo_data/mustard0/mesh/textured_simple.obj")
    parser.add_argument("--test_scene_dir", type=str,
                        default=f"{code_dir}/demo_data/mustard0")
    parser.add_argument("--palm_poses_npz", type=str, required=True,
                        help="NPZ with 'poses' (N,4,4) T_cam_palm per RGB frame.")
    parser.add_argument("--palm_frame_offset", type=int, default=0)
    parser.add_argument("--palm_frame_stride", type=int, default=1)
    parser.add_argument("--depth_dir", type=str, default=None,
                        help="Depth PNG dir (default: test_scene_dir/depth/)")
    parser.add_argument("--depth_dir_occ", type=str, default=None,
                        help="Depth PNG dir used when occluded (default: --depth_dir)")
    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--track_refine_iter", type=int, default=1)
    parser.add_argument("--debug", type=int, default=1)
    parser.add_argument("--debug_dir", type=str, default=f"{code_dir}/debug")
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

    reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)

    palm_data  = np.load(args.palm_poses_npz, allow_pickle=True)
    palm_poses = palm_data["poses"].astype(np.float64)
    n_palm     = palm_poses.shape[0]
    logging.info(f"Loaded {n_palm} palm poses from {args.palm_poses_npz} "
                 f"(RGB frames = {len(reader.color_files)}, "
                 f"offset={args.palm_frame_offset}, stride={args.palm_frame_stride})")

    depth_dir_vis = args.depth_dir or os.path.join(args.test_scene_dir, "depth")
    depth_dir_occ = args.depth_dir_occ or depth_dir_vis

    def load_depth_png(d_dir, id_str):
        return cv2.imread(os.path.join(d_dir, f"{id_str}.png"),
                          cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

    # Anchor state for palm-delta propagation
    anchor_pose     = None    # T_cam_obj at anchor frame (numpy 4x4)
    anchor_palm_inv = None    # inv(T_cam_palm[anchor_frame])

    # State machine: STATIC ↔ MOVING, driven by optical flow with debounce buffer
    state              = "STATIC"
    buffer_count       = 0     # consecutive flow frames agreeing with transition
    frames_in_state    = 0     # how long we've been in the current state
    prev_gray          = None
    prev_mask          = None

    depth_scale = 1.0
    pose        = None

    for i in range(len(reader.color_files)):
        color = reader.get_color(i)
        gray  = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
        t1    = time.time()

        depth = load_depth_png(depth_dir_vis, reader.id_strs[i])

        mask_path = os.path.join(args.test_scene_dir, "masks", f"{reader.id_strs[i]}.png")
        mask = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)

        palm_idx = resolve_palm_idx(i, args.palm_frame_offset, args.palm_frame_stride, n_palm)
        T_cam_palm = palm_poses[palm_idx]

        if i == 0:
            pose = binary_search_depth(est, mesh, color, mask.astype(bool), reader.K, debug=True)
            logging.info(f"Initial duck pose:\n{pose}")

            obj_pixels = mask > 0
            bsd_z      = float(pose[2, 3])
            vda_z      = depth[obj_pixels].mean() if obj_pixels.any() else 1.0
            depth_scale = bsd_z / vda_z if vda_z > 0 else 1.0
            logging.info(f"Depth scale: {depth_scale:.3f}")

            anchor_pose     = pose.copy()
            anchor_palm_inv = np.linalg.inv(T_cam_palm)

        else:
            flow_med = compute_mask_flow(prev_gray, gray, prev_mask, mask)
            frames_in_state += 1

            # Hysteresis: sensitive entry, strict exit. STOP is also gated by
            # MIN_MOVING_FRAMES so a brief flow dip right after entering MOVING
            # doesn't bounce us back to STATIC.
            # Unreliable signal (None) is sticky — buffer neither advanced nor
            # reset, so brief full-occlusion blackouts don't derail state.
            transitioned = False
            if flow_med is not None:
                if state == "STATIC":
                    threshold_met = flow_med > MOVE_FLOW_START_PX
                    buffer_target = BUFFER_START_FRAMES
                    next_state    = "MOVING"
                    stop_gate_ok  = True
                else:  # MOVING
                    threshold_met = flow_med < MOVE_FLOW_STOP_PX
                    buffer_target = BUFFER_STOP_FRAMES
                    next_state    = "STATIC"
                    stop_gate_ok  = frames_in_state >= MIN_MOVING_FRAMES

                if threshold_met and stop_gate_ok:
                    buffer_count += 1
                    if buffer_count >= buffer_target:
                        if next_state == "MOVING":
                            # STATIC → MOVING: anchor palm at this frame; the
                            # last frozen pose is already in anchor_pose.
                            anchor_pose     = pose.copy()
                            anchor_palm_inv = np.linalg.inv(T_cam_palm)
                        else:
                            # MOVING → STATIC: re-acquire via FP once, freeze.
                            d_scaled = depth * depth_scale
                            pose = est.track_one(
                                rgb=color, depth=d_scaled, K=reader.K,
                                iteration=args.track_refine_iter,
                            )
                            anchor_pose     = pose.copy()
                            anchor_palm_inv = np.linalg.inv(T_cam_palm)
                        state = next_state
                        buffer_count = 0
                        frames_in_state = 0
                        transitioned = True
                else:
                    # evidence against transition — reset
                    buffer_count = 0
            # else: signal unreliable → keep buffer_count as-is (sticky)

            # Per-frame pose update based on (possibly just-flipped) state.
            # If we just transitioned MOVING→STATIC we already ran FP above.
            if state == "MOVING" and not transitioned:
                pose = T_cam_palm @ anchor_palm_inv @ anchor_pose
                est.pose_last = torch.from_numpy(pose).float().cuda()
            # STATIC (no transition) or STATIC after transition: keep pose.

            if i % LOG_INTERVAL == 0:
                flow_str = f"{flow_med:.2f}px" if flow_med is not None else "N/A"
                logging.info(f"[frame {i}] state={state}  buf={buffer_count}  "
                             f"in_state={frames_in_state}  flow_med={flow_str}")

        prev_gray = gray
        prev_mask = mask

        t2 = time.time()
        os.makedirs(f"{debug_dir}/ob_in_cam", exist_ok=True)
        np.savetxt(f"{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt", pose.reshape(4, 4))

        if args.debug >= 1:
            center_pose = pose @ np.linalg.inv(to_origin)
            color = cv2.putText(color, f"fps {int(1/(t2-t1))} {state}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K,
                                thickness=3, transparency=0, is_input_rgb=True)

        if args.debug >= 2:
            os.makedirs(f"{debug_dir}/track_vis", exist_ok=True)
            imageio.imwrite(f"{debug_dir}/track_vis/{reader.id_strs[i]}.png", vis)
