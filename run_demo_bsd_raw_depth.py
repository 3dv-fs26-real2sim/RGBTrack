"""
FoundationPose tracking: binary_search_depth init + raw pre-scaled depth for tracking.

Frame 0: binary_search_depth (RGB + mask only, no real depth needed) for clean init.
Frames 1+: raw depth PNGs loaded as-is (uint16 PNG mm → float32 m), no calibration.

Usage:
    python run_demo_bsd_raw_depth.py \
        --mesh_file  /path/to/mesh.obj \
        --test_scene_dir /path/to/scene \
        --pred_depth_dir /path/to/depth_pngs \
        --debug_dir  /path/to/debug \
        --debug 2
"""
import time
from estimater import *
from datareader import *
import argparse
from tools import *
import numpy as np
from scipy.spatial.transform import Rotation as ScipyR

OCCLUSION_THRESHOLD = 0.90
RECOVERY_THRESHOLD  = 0.95
MAX_ROT_DELTA_DEG   = 3.0


def rotation_delta_deg(R1, R2):
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
    parser.add_argument("--num_frames", type=int, default=None,
                        help="Process only the first N frames (default: all)")
    parser.add_argument("--pred_depth_dir", type=str, default=None,
                        help="Directory of depth PNGs (uint16 mm). Defaults to test_scene_dir/depth/")
    parser.add_argument("--bsd_depth_min", type=float, default=0.1,
                        help="binary_search_depth lower bound (m)")
    parser.add_argument("--bsd_depth_max", type=float, default=2.0,
                        help="binary_search_depth upper bound (m)")
    args = parser.parse_args()

    if args.pred_depth_dir is None:
        args.pred_depth_dir = os.path.join(args.test_scene_dir, "depth")

    set_logging_format()
    set_seed(0)

    mesh = trimesh.load(args.mesh_file)
    mesh.apply_scale(0.001)

    debug = args.debug
    debug_dir = args.debug_dir
    os.system(f"rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam")

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    scorer  = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx   = dr.RasterizeCudaContext()
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
    logging.info(f"Depth dir: {args.pred_depth_dir}")

    frame0_mask_area  = None
    last_good_rotation = None
    was_occluded      = False

    num_frames = args.num_frames if args.num_frames is not None else len(reader.color_files)
    for i in range(min(num_frames, len(reader.color_files))):
        color = reader.get_color(i)
        t1 = time.time()

        depth_path = os.path.join(args.pred_depth_dir, f"{reader.id_strs[i]}.png")
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

        mask_path = os.path.join(args.test_scene_dir, "masks", f"{reader.id_strs[i]}.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)

        if i == 0:
            logging.info("Frame 0: running binary_search_depth for clean initialization...")
            pose = binary_search_depth(
                est, mesh, color, mask.astype(bool), reader.K,
                depth_min=args.bsd_depth_min,
                depth_max=args.bsd_depth_max,
                w=color.shape[1], h=color.shape[0],
                debug=False,
                iteration=args.est_refine_iter,
            )
            logging.info(f"Initial pose (BSD):\n{pose}")
            frame0_mask_area   = float(mask.sum())
            last_good_rotation = pose[:3, :3].copy()
        else:
            mask_area = float(mask.sum())
            occluded  = (frame0_mask_area > 0) and (mask_area < OCCLUSION_THRESHOLD * frame0_mask_area)

            pose = est.track_one_new(
                rgb=color, depth=depth, K=reader.K,
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
                    logging.info(f"[frame {i}] Recovery re-init via binary_search_depth")
                    pose = binary_search_depth(
                        est, mesh, color, mask.astype(bool), reader.K,
                        depth_min=args.bsd_depth_min,
                        depth_max=args.bsd_depth_max,
                        w=color.shape[1], h=color.shape[0],
                        debug=False,
                        iteration=args.est_refine_iter,
                    )
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
