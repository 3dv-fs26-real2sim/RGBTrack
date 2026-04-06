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

CMAP = cv2.COLORMAP_PLASMA   # colormap for depth


def depth_to_rgb(depth, vmin, vmax):
    """Float32 depth (m) → uint8 RGB colourised frame."""
    depth_clipped = np.clip(depth, vmin, vmax)
    norm = ((depth_clipped - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm, CMAP)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


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

    put(f"{vmax:.2f}m", 12)
    put(f"{(vmin+vmax)/2:.2f}m", h // 2 + 5)
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
                        choices=["vda", "metric3d", "depth_pro", "depth_pro_maps"],
                        help="Depth source")
    parser.add_argument("--metric3d_ckpt", type=str,
                        default="/work/courses/3dv/team22/metric3d_vit_large.pth")
    parser.add_argument("--depth_pro_ckpt", type=str,
                        default="/work/courses/3dv/team22/ml-depth-pro/checkpoints/depth_pro.pt")
    parser.add_argument("--out_video", type=str, default=None,
                        help="Output AVI path. Defaults to test_scene_dir/depth_<source>.avi")
    parser.add_argument("--depth_pro_maps_dir", type=str, default=None,
                        help="Directory of pre-generated depth PNGs when using --source depth_pro_maps")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--depth_scale", type=float, default=None,
                        help="Scale factor for VDA/PNG depths (auto-computed if not set)")
    args = parser.parse_args()

    out_video = args.out_video or os.path.join(
        args.test_scene_dir, f"depth_{args.source}.avi")

    reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=float("inf"))

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

    # First pass: compute global depth range for consistent colormap
    print("Computing depth range...")
    depths = []
    for i in range(min(50, len(reader.color_files))):   # sample 50 frames
        color = reader.get_color(i)
        depth = _get_depth(i, color, reader, args, depth_model)
        if depth is not None:
            depths.append(depth[depth > 0.1])
    all_vals = np.concatenate(depths)
    vmin = float(np.percentile(all_vals, 2))
    vmax = float(np.percentile(all_vals, 98))
    print(f"Depth range: {vmin:.3f}m – {vmax:.3f}m")

    # Second pass: write video
    first_color = reader.get_color(0)
    h, w = first_color.shape[:2]
    bar_w = 95
    frame_w = w * 2 + bar_w
    out = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"XVID"), args.fps, (frame_w, h))

    for i in range(len(reader.color_files)):
        color  = reader.get_color(i)
        depth  = _get_depth(i, color, reader, args, depth_model)
        if depth is None:
            continue
        frame  = make_frame(color, depth, vmin, vmax, args.source.upper())
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

    elif depth_model is not None:
        return depth_model.estimate(color, reader.K)

    return None


if __name__ == "__main__":
    main()
