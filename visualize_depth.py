"""
Depth map visualization with colorbar scale.
Supports VDA (PNG dir), Metric3D (inline), and Depth Pro (inline or PNG dir).

Output: side-by-side RGB + colorized depth video with depth scale bar.

Usage examples:
  # VDA (pre-generated PNGs)
  python visualize_depth.py --source vda

  # Metric3D (inline inference)
  python visualize_depth.py --source metric3d \
      --metric3d_ckpt /work/courses/3dv/team22/metric3d_vit_large.pth

  # Depth Pro (inline inference)
  python visualize_depth.py --source depth_pro \
      --depth_pro_ckpt /work/courses/3dv/team22/ml-depth-pro/checkpoints/depth_pro.pt

  # Depth Pro (pre-generated PNGs)
  python visualize_depth.py --source depth_pro_maps
"""
import os, argparse
import numpy as np
import cv2
from datareader import YcbineoatReader
from estimater import *
from tools import *

CMAP = cv2.COLORMAP_PLASMA   # colormap for depth


def detect_table_cutoff(depths_sample):
    """
    Find the table/background boundary using Otsu thresholding on the
    depth histogram. The valley between the table peak and background
    peak is the cutoff — everything beyond is background.
    """
    valid = np.concatenate([d[d > 0.1].ravel() for d in depths_sample])

    # Normalise to uint8 for Otsu (map depth range → 0-255)
    d_min, d_max = float(valid.min()), float(valid.max())
    norm = ((valid - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    thresh_norm, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Convert back to metres
    cutoff = d_min + (thresh_norm / 255.0) * (d_max - d_min)
    print(f"Otsu table cutoff: {cutoff:.3f}m")
    return float(cutoff)


def mask_background(depth, cutoff):
    """Zero out pixels beyond cutoff and remove small isolated far patches."""
    out = depth.copy()
    out[out > cutoff] = 0.0

    # Remove isolated non-zero blobs (e.g. far reflections that slipped through)
    valid = (out > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    valid_clean = cv2.morphologyEx(valid, cv2.MORPH_OPEN, kernel)
    out[valid_clean == 0] = 0.0
    return out


def depth_to_rgb(depth, vmin, vmax):
    """Float32 depth (m) → uint8 RGB colourised frame (log scale, zeros = black)."""
    result = np.zeros((*depth.shape, 3), dtype=np.uint8)
    valid  = depth > 0
    if valid.any():
        d = np.clip(depth[valid], max(vmin, 1e-3), vmax)
        log_d   = np.log(d)
        log_min = np.log(max(vmin, 1e-3))
        log_max = np.log(vmax)
        norm = ((log_d - log_min) / (log_max - log_min) * 255).astype(np.uint8)
        colored = cv2.applyColorMap(norm.reshape(-1, 1), CMAP).reshape(-1, 3)
        result[valid] = colored[:, ::-1]  # BGR→RGB
    return result


def make_colorbar(vmin, vmax, h, bar_w=60, font_scale=0.45):
    """Vertical colorbar with min/max labels, height h."""
    bar = np.linspace(255, 0, h).astype(np.uint8).reshape(h, 1)
    bar = np.repeat(bar, bar_w - 20, axis=1)
    bar_colored = cv2.applyColorMap(bar, CMAP)
    bar_colored = cv2.cvtColor(bar_colored, cv2.COLOR_BGR2RGB)

    # pad right side for labels
    label_w = 55
    canvas = np.ones((h, bar_w - 20 + label_w, 3), dtype=np.uint8) * 30
    canvas[:, :bar_w - 20] = bar_colored

    def put(text, y):
        cv2.putText(canvas, text, (bar_w - 18, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (220, 220, 220), 1, cv2.LINE_AA)

    mid = np.exp((np.log(max(vmin, 1e-3)) + np.log(vmax)) / 2)
    put(f"{vmax:.2f}m", 12)
    put(f"{mid:.2f}m", h // 2 + 5)
    put(f"{vmin:.2f}m", h - 5)
    return canvas


def make_frame(rgb, depth, vmin, vmax, source_label):
    h, w = rgb.shape[:2]
    depth_vis  = depth_to_rgb(depth, vmin, vmax)
    colorbar   = make_colorbar(vmin, vmax, h)

    # label on depth panel
    depth_labeled = depth_vis.copy()
    cv2.putText(depth_labeled, source_label, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return np.concatenate([rgb, depth_labeled, colorbar], axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_scene_dir", type=str,
                        default="/work/courses/3dv/team22/foundationpose/data/20250804_104715")
    parser.add_argument("--source", type=str, required=True,
                        choices=["vda", "metric3d", "depth_pro", "depth_pro_maps", "custom"],
                        help="Depth source. Use 'custom' with --depth_dir for any pre-generated PNGs.")
    parser.add_argument("--metric3d_ckpt", type=str,
                        default="/work/courses/3dv/team22/metric3d_vit_large.pth")
    parser.add_argument("--depth_pro_ckpt", type=str,
                        default="/work/courses/3dv/team22/ml-depth-pro/checkpoints/depth_pro.pt")
    parser.add_argument("--out_video", type=str, default=None,
                        help="Output AVI path. Defaults to test_scene_dir/depth_<source>.avi")
    parser.add_argument("--depth_pro_maps_dir", type=str, default=None,
                        help="Directory of pre-generated depth PNGs when using --source depth_pro_maps")
    parser.add_argument("--depth_dir", type=str, default=None,
                        help="Directory of pre-generated depth PNGs when using --source custom")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--depth_scale", type=float, default=None,
                        help="Scale factor for depth maps (overrides auto-calibration)")
    parser.add_argument("--table_depth", type=float, default=None,
                        help="Hard cutoff in metres (after scaling): pixels at or beyond this are zeroed")
    parser.add_argument("--label", type=str, default=None,
                        help="Label shown in video (defaults to --source value)")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run binary_search_depth on frame 0 to compute depth scale (same as tracking pipeline)")
    parser.add_argument("--mesh_file", type=str,
                        default="/work/courses/3dv/team22/foundationpose/data/object/duck/duck.obj",
                        help="Mesh file for calibration (only used with --calibrate)")
    args = parser.parse_args()

    out_video = args.out_video or os.path.join(
        args.test_scene_dir, f"depth_{args.source}.mp4")

    reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=float("inf"))

    # ── Depth scale calibration via binary_search_depth (same as tracking) ────
    depth_scale = args.depth_scale or 1.0
    if args.calibrate and args.depth_scale is None:
        print("Calibrating depth scale via binary_search_depth...")
        import trimesh
        mesh = trimesh.load(args.mesh_file)
        mesh.apply_scale(0.001)
        scorer  = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx   = dr.RasterizeCudaContext()
        est_cal = FoundationPose(
            model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
            scorer=scorer, refiner=refiner, debug_dir="/tmp/vis_cal", debug=0, glctx=glctx)
        color0 = reader.get_color(0)
        mask_path = os.path.join(args.test_scene_dir, "masks", f"{reader.id_strs[0]}.png")
        mask0 = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask0 = (mask0 > 127).astype(np.uint8)
        depth0_raw = _get_depth(0, color0, reader, args, None)  # depth_model not loaded yet
        pose0 = binary_search_depth(est_cal, mesh, color0, mask0.astype(bool), reader.K, debug=False)
        obj_px = mask0 > 0
        vda_z  = float(depth0_raw[obj_px].mean()) if obj_px.any() else 1.0
        bsd_z  = float(pose0[2, 3])
        depth_scale = bsd_z / vda_z if vda_z > 0 else 1.0
        print(f"Calibrated depth scale: {depth_scale:.4f}  (raw={vda_z:.3f}m → metric={bsd_z:.3f}m)")
        del est_cal, scorer, refiner

    # Load depth model if needed
    depth_model = None
    if args.source == "metric3d":
        from metric3d_wrapper import Metric3DWrapper
        depth_model = Metric3DWrapper(checkpoint_path=args.metric3d_ckpt)
        print("Metric3D loaded")
    elif args.source in ("depth_pro", "depth_pro_maps"):
        if args.source == "depth_pro":
            from depth_pro_wrapper import DepthProWrapper
            depth_model = DepthProWrapper(checkpoint_path=args.depth_pro_ckpt)
            print("Depth Pro loaded")

    # First pass: sample frames → detect table cutoff + depth range
    print("Sampling frames for table detection and depth range...")
    sample_depths = []
    for i in range(min(50, len(reader.color_files))):
        color = reader.get_color(i)
        depth = _get_depth(i, color, reader, args, depth_model)
        if depth is not None:
            sample_depths.append(depth * depth_scale)

    # Table cutoff: use provided value or auto-detect
    if args.table_depth is not None:
        table_cutoff = args.table_depth
        print(f"Using fixed table cutoff: {table_cutoff:.3f}m")
    else:
        table_cutoff = detect_table_cutoff(sample_depths)
    masked = [mask_background(d, table_cutoff) for d in sample_depths]
    valid_vals = np.concatenate([d[d > 0.1].ravel() for d in masked])
    vmin = float(np.percentile(valid_vals, 2))
    vmax = table_cutoff
    print(f"Depth range (table only): {vmin:.3f}m – {vmax:.3f}m  (scale={depth_scale:.4f})")

    # Second pass: write video
    first_color = reader.get_color(0)
    h, w = first_color.shape[:2]
    bar_w = 95
    frame_w = w * 2 + bar_w
    out = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (frame_w, h))

    for i in range(len(reader.color_files)):
        color  = reader.get_color(i)
        depth  = _get_depth(i, color, reader, args, depth_model)
        if depth is None:
            continue
        depth  = mask_background(depth * depth_scale, table_cutoff)
        label  = args.label.upper() if args.label else args.source.upper()
        frame  = make_frame(color, depth, vmin, vmax, label)
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if i % 50 == 0:
            print(f"  [{i}/{len(reader.color_files)}]")

    out.release()
    print(f"Saved to {out_video}")


def _get_depth(i, color, reader, args, depth_model):
    if args.source == "vda":
        path = os.path.join(args.test_scene_dir, "depth", f"{reader.id_strs[i]}.png")
        raw  = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            return None
        return raw.astype(np.float32) / 1000.0

    elif args.source == "depth_pro_maps":
        maps_dir = args.depth_pro_maps_dir or os.path.join(args.test_scene_dir, "depth_pro")
        path = os.path.join(maps_dir, f"{reader.id_strs[i]}.png")
        raw  = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            return None
        return raw.astype(np.float32) / 1000.0

    elif args.source == "custom":
        assert args.depth_dir, "--depth_dir required with --source custom"
        path = os.path.join(args.depth_dir, f"{reader.id_strs[i]}.png")
        raw  = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            return None
        return raw.astype(np.float32) / 1000.0

    elif depth_model is not None:
        return depth_model.estimate(color, reader.K)

    return None


if __name__ == "__main__":
    main()
