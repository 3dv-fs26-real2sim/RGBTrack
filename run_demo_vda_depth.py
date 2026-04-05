"""
FoundationPose tracking using pre-generated Video Depth Anything depth maps.

Depth maps must be uint16 PNG files (mm) in test_scene_dir/depth/.
SAM2 video masks must be in test_scene_dir/masks/.
"""
import time
from estimater import *
from datareader import *
import argparse
from tools import *
import numpy as np
from scipy.spatial.transform import Rotation as ScipyR

SAVE_VIDEO = False

# ── Occlusion handling settings ────────────────────────────────────────────────
# Single threshold: below this fraction of frame-0 mask area → occluded.
OCCLUSION_THRESHOLD = 0.93

# Weight given to frozen rotation during occlusion (SLERP).
# 1.0 = full freeze, lower = allow small tracked motion.
OCCLUSION_ROT_WEIGHT = 0.92

# Frames to wait after object becomes visible again before re-init.
RECOVERY_DELAY_FRAMES = 5
# ──────────────────────────────────────────────────────────────────────────────

def slerp_rotation(R_frozen, R_tracked, weight_frozen):
    r_frozen = ScipyR.from_matrix(R_frozen)
    r_tracked = ScipyR.from_matrix(R_tracked)
    r_out = ScipyR.slerp([0, 1], [r_frozen, r_tracked])(1.0 - weight_frozen)
    return r_out.as_matrix()


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

    frame0_mask_area = None
    depth_scale = 1.0
    last_good_rotation = None
    was_occluded = False
    frames_since_visible = 0

    for i in range(len(reader.color_files)):
        color = reader.get_color(i)
        t1 = time.time()

        depth_path = os.path.join(args.test_scene_dir, "depth", f"{reader.id_strs[i]}.png")
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

        mask_path = os.path.join(args.test_scene_dir, "masks", f"{reader.id_strs[i]}.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)

        if i == 0:
            pose = binary_search_depth(est, mesh, color, mask.astype(bool), reader.K, debug=True)
            logging.info(f"Initial pose:\n{pose}")
            frame0_mask_area = float(mask.sum())
            last_good_rotation = pose[:3, :3].copy()
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

            if occluded:
                # weighted SLERP: mostly frozen, allow tiny tracked motion
                pose[:3, :3] = slerp_rotation(last_good_rotation, pose[:3, :3], OCCLUSION_ROT_WEIGHT)
                was_occluded = True
                frames_since_visible = 0
            else:
                if was_occluded:
                    # object just became visible again — start counting
                    frames_since_visible += 1
                    if frames_since_visible >= RECOVERY_DELAY_FRAMES:
                        logging.info(f"[frame {i}] Recovery re-init with binary_search_depth")
                        pose = binary_search_depth(est, mesh, color, mask.astype(bool), reader.K, debug=False)
                        last_good_rotation = pose[:3, :3].copy()
                        was_occluded = False
                        frames_since_visible = 0
                    else:
                        pose[:3, :3] = last_good_rotation  # keep frozen while waiting
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
