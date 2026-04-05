import time
from estimater import *
from datareader import *
import argparse
from tools import *
import numpy as np
from metric3d_wrapper import Metric3DWrapper

SAVE_VIDEO = False

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
    parser.add_argument("--metric3d_ckpt", type=str,
                        default="/work/courses/3dv/team22/metric3d_vit_large.pth")
    parser.add_argument("--use_video_masks", action="store_true",
                        help="Use pre-generated SAM2 video masks from test_scene_dir/masks/")
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

    logging.info("loading Metric3D...")
    metric3d = Metric3DWrapper(checkpoint_path=args.metric3d_ckpt)
    logging.info("Metric3D ready")

    reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)

    for i in range(len(reader.color_files)):
        color = reader.get_color(i)
        t1 = time.time()

        # get depth from Metric3D every frame
        depth = metric3d.estimate(color, reader.K)

        # get mask
        if i == 0:
            mask = reader.get_mask(0).astype(bool)
            pose = est.register(
                K=reader.K, rgb=color, depth=depth, ob_mask=mask,
                iteration=args.est_refine_iter
            )
            logging.info(f"Initial pose:\n{pose}")
        else:
            if args.use_video_masks:
                mask_path = os.path.join(args.test_scene_dir, "masks", f"{reader.id_strs[i]}.png")
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = (mask > 127).astype(np.uint8)
            pose = est.track_one_new(
                rgb=color, depth=depth, K=reader.K,
                iteration=args.track_refine_iter, mask=mask
            )

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
