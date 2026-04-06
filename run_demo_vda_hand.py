"""
FoundationPose duck tracker — mesh-vertex optical flow (option 5).

During occlusion: duck mesh vertices projected with last known pose,
tracked frame-to-frame with LK optical flow, solvePnPRansac recovers
new rotation from (fixed 3D object verts → tracked 2D positions).

No rotation blocking — OF result used directly when available,
raw FoundationPose otherwise.

Recovery re-init (binary_search_depth) when mask recovers to 0.95.

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

# ── Settings ───────────────────────────────────────────────────────────────────
OCCLUSION_THRESHOLD = 0.90   # duck mask below this → occluded
RECOVERY_THRESHOLD  = 0.95   # duck mask above this → recovered (re-init)

OF_MAX_PTS          = 200    # max projected mesh vertices to track
OF_MIN_DIST         = 5      # min pixel distance between selected points

LOG_INTERVAL        = 5
# ──────────────────────────────────────────────────────────────────────────────


def project_mesh_verts(verts_obj, pose, K, h, w, mask, max_pts, min_dist):
    """
    Project object-space vertices with pose, return visible (N,2) and (N,3).
    Filters: positive depth, in-image bounds, on duck mask.
    Subsamples to max_pts with approximate min_dist spacing.
    """
    R, t = pose[:3, :3], pose[:3, 3]
    pts_cam = (R @ verts_obj.T).T + t          # (V,3) camera-space
    depth_v  = pts_cam[:, 2]

    # homogeneous image coords
    u = pts_cam[:, 0] / depth_v * K[0, 0] + K[0, 2]
    v = pts_cam[:, 1] / depth_v * K[1, 1] + K[1, 2]

    ui = np.round(u).astype(int)
    vi = np.round(v).astype(int)

    in_bounds = (depth_v > 0.05) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    idx = np.where(in_bounds)[0]
    if len(idx) == 0:
        return None, None

    # filter to duck mask
    on_mask = mask[vi[idx], ui[idx]] > 0
    idx = idx[on_mask]
    if len(idx) == 0:
        return None, None

    pts2d = np.stack([u[idx], v[idx]], axis=1).astype(np.float32)
    pts3d = verts_obj[idx].astype(np.float32)

    # greedy spatial subsample
    if len(pts2d) > max_pts:
        chosen = []
        grid = {}
        cell = int(min_dist)
        for j in np.random.permutation(len(pts2d)):
            cx, cy = int(pts2d[j, 0] // cell), int(pts2d[j, 1] // cell)
            if (cx, cy) not in grid:
                grid[(cx, cy)] = True
                chosen.append(j)
                if len(chosen) >= max_pts:
                    break
        pts2d = pts2d[chosen]
        pts3d = pts3d[chosen]

    return pts2d, pts3d


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
    logging.info("estimator initialized")

    reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)

    frame0_mask_area = None
    depth_scale      = 1.0
    was_occluded     = False

    # Optical flow state
    of_prev_gray = None
    of_pts       = None   # (N,2) float32 — current tracked 2D positions
    of_3d_pts    = None   # (N,3) float32 — fixed object-space 3D positions

    for i in range(len(reader.color_files)):
        color = reader.get_color(i)
        t1    = time.time()

        depth_path = os.path.join(args.test_scene_dir, "depth", f"{reader.id_strs[i]}.png")
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

        mask_path = os.path.join(args.test_scene_dir, "masks", f"{reader.id_strs[i]}.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)

        gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)

        if i == 0:
            pose = binary_search_depth(est, mesh, color, mask.astype(bool), reader.K, debug=True)
            logging.info(f"Initial duck pose:\n{pose}")
            frame0_mask_area = float(mask.sum())
            obj_pixels  = mask > 0
            vda_z       = depth[obj_pixels].mean() if obj_pixels.any() else 1.0
            bsd_z       = float(pose[2, 3])
            depth_scale = bsd_z / vda_z if vda_z > 0 else 1.0
            logging.info(f"Depth scale: {depth_scale:.3f}")
        else:
            mask_area = float(mask.sum())
            occluded  = (frame0_mask_area > 0) and (mask_area < OCCLUSION_THRESHOLD * frame0_mask_area)
            h_img, w_img = mask.shape[:2]

            # Always run FP (translation always used; rotation below)
            pose = est.track_one_new(
                rgb=color, depth=depth * depth_scale, K=reader.K,
                iteration=args.track_refine_iter, mask=mask
            )

            # ── Optical flow ───────────────────────────────────────────────────
            of_rot = None

            if not occluded:
                # Refresh projected mesh keypoints on clean duck view
                pts2d, pts3d = project_mesh_verts(
                    mesh.vertices, pose, reader.K, h_img, w_img, mask,
                    OF_MAX_PTS, OF_MIN_DIST)
                if pts2d is not None:
                    of_pts    = pts2d
                    of_3d_pts = pts3d

            elif occluded and of_pts is not None and of_prev_gray is not None and len(of_pts) >= 6:
                # Track projected mesh verts with LK
                new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    of_prev_gray, gray,
                    of_pts.reshape(-1, 1, 2), None,
                    winSize=(21, 21), maxLevel=3)

                if new_pts is not None and status is not None:
                    ok       = status.ravel().astype(bool)
                    good_2d  = new_pts.reshape(-1, 2)[ok]
                    good_3d  = of_3d_pts[ok]

                    if len(good_3d) >= 6:
                        try:
                            ret, rvec, tvec, inliers = cv2.solvePnPRansac(
                                good_3d, good_2d,
                                reader.K.astype(np.float64), None,
                                iterationsCount=100, reprojectionError=4.0)
                            if ret and inliers is not None and len(inliers) >= 4:
                                R_of, _ = cv2.Rodrigues(rvec)
                                of_rot = R_of
                                logging.debug(f"[frame {i}] OF solvePnP ok — "
                                              f"{len(inliers)}/{len(good_3d)} inliers")
                        except cv2.error:
                            pass

                    # Keep tracking the subset that survived LK
                    of_pts    = good_2d
                    of_3d_pts = good_3d

            # Update previous frame
            of_prev_gray = gray.copy()

            # ── Pose assembly ──────────────────────────────────────────────────
            if occluded:
                if of_rot is not None:
                    pose[:3, :3] = of_rot   # OF solvePnP result
                # else: raw FP rotation — no blocking, let it run

                was_occluded = True

            elif was_occluded and mask_area >= RECOVERY_THRESHOLD * frame0_mask_area:
                logging.info(f"[frame {i}] Recovery re-init")
                pose = binary_search_depth(
                    est, mesh, color, mask.astype(bool), reader.K, debug=False)
                was_occluded = False
                of_pts       = None   # force fresh keypoint detection next frame

            elif was_occluded:
                # partially recovered — FP runs free (no blocking)
                pass

            # else: fully unoccluded, FP has full control

            if i % LOG_INTERVAL == 0:
                of_n = len(of_pts) if of_pts is not None else 0
                logging.info(f"[frame {i}] occluded={occluded}  "
                             f"of_pts={of_n}  of_rot={'yes' if of_rot is not None else 'no'}")

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
