"""
Per-frame depth histogram video.

For each frame renders the pixel-count distribution over depth (scaled),
and writes a video. Start with --source vda to test, then add more.

Usage:
    python plot_depth_histogram_video.py --source vda
    python plot_depth_histogram_video.py --source da3
"""
import os, argparse, glob
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

SCENE_DIR = "/work/courses/3dv/team22/foundationpose/data/20250804_104715"
DEBUG_DIR  = "/work/courses/3dv/team22/foundationpose/debug"

DEPTH_DIRS = {
    "vda":           "depth",
    "vda_streaming": "depth_vda_streaming",
    "da3":           "depth_da3",
    "metric3d":      "depth_metric3d",
    "depth_pro":     "depth_pro",
    "vggt":          "depth_vggt",
}

COLORS = {
    "vda":           "#4C72B0",
    "vda_streaming": "#2196F3",
    "da3":           "#DD8452",
    "metric3d":      "#55A868",
    "depth_pro":     "#C44E52",
    "vggt":          "#8172B3",
}

DEPTH_MIN = 0.05   # m
DEPTH_MAX = 2.00   # m
N_BINS    = 120
FPS       = 50


def get_id_strs(scene_dir):
    files = sorted(glob.glob(os.path.join(scene_dir, "rgb", "*.png")))
    return [os.path.splitext(os.path.basename(f))[0] for f in files]


def load_depth(depth_dir, id_str):
    raw = cv2.imread(os.path.join(depth_dir, f"{id_str}.png"), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    return raw.astype(np.float32) / 1000.0


def load_mask(masks_dir, id_str):
    m = cv2.imread(os.path.join(masks_dir, f"{id_str}.png"), cv2.IMREAD_GRAYSCALE)
    return (m > 127) if m is not None else None


def compute_scale(depth_dir, masks_dir, id_strs):
    """Compute depth_scale = duck-mask mean at frame 0 → 0.45m reference."""
    for id_str in id_strs[:5]:
        d = load_depth(depth_dir, id_str)
        m = load_mask(masks_dir, id_str)
        if d is None or m is None or not m.any():
            continue
        vals = d[m]
        vals = vals[vals > 0.01]
        if len(vals) == 0:
            continue
        raw_mean = float(np.mean(vals))
        # scale so duck sits at ~0.45m (typical scene distance)
        return 0.45 / raw_mean if raw_mean > 0 else 1.0
    return 1.0


def render_histogram(depths_scaled, labels, colors, frame_idx, n_frames, bins, layout="overlay", ymax=None):
    n = len(labels)
    if layout == "grid":
        cols = min(n, 4)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
        axes_flat = [axes[r][c] for r in range(rows) for c in range(cols)]
        fig.suptitle(f"Depth spectrum (scaled) — frame {frame_idx:04d} / {n_frames}",
                     fontsize=13, fontweight="bold")
        for ax, d, label, color in zip(axes_flat, depths_scaled, labels, colors):
            if d is not None:
                vals = d[(d > DEPTH_MIN) & (d < DEPTH_MAX)]
                if len(vals):
                    ax.hist(vals, bins=bins, color=color, alpha=0.6,
                            density=True, histtype="stepfilled")
                    ax.hist(vals, bins=bins, color=color, alpha=0.9,
                            density=True, histtype="step", linewidth=1.2)
            ax.set_xlim(DEPTH_MIN, DEPTH_MAX)
            if ymax:
                ax.set_ylim(0, ymax)
            ax.set_title(label.upper(), fontsize=11, color=color)
            ax.set_xlabel("Depth (m)", fontsize=9)
            ax.set_ylabel("Density", fontsize=9)
            ax.grid(True, alpha=0.3)
        # hide unused subplots
        for ax in axes_flat[n:]:
            ax.set_visible(False)
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        for d, label, color in zip(depths_scaled, labels, colors):
            if d is None:
                continue
            vals = d[(d > DEPTH_MIN) & (d < DEPTH_MAX)]
            if len(vals) == 0:
                continue
            ax.hist(vals, bins=bins, color=color, alpha=0.55, label=label.upper(),
                    density=True, histtype="stepfilled")
            ax.hist(vals, bins=bins, color=color, alpha=0.9,
                    density=True, histtype="step", linewidth=1.2)
        ax.set_xlim(DEPTH_MIN, DEPTH_MAX)
        if ymax:
            ax.set_ylim(0, ymax)
        ax.set_xlabel("Depth (m, scaled)", fontsize=11)
        ax.set_ylabel("Pixel density", fontsize=11)
        ax.set_title(f"Depth spectrum — frame {frame_idx:04d} / {n_frames}", fontsize=12)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    canvas = FigureCanvas(fig)
    canvas.draw()
    w, h = canvas.get_width_height()
    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
    plt.close(fig)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="vda",
                        help="Comma-separated sources, e.g. vda,da3,metric3d,depth_pro")
    parser.add_argument("--scene_dir", default=SCENE_DIR)
    parser.add_argument("--out", default=None,
                        help="Output mp4 path. Defaults to debug/depth_hist_<source>.mp4")
    parser.add_argument("--layout", type=str, default="overlay", choices=["overlay", "grid"],
                        help="overlay: all on one plot, grid: one subplot per source")
    parser.add_argument("--no_scale", action="store_true",
                        help="Skip scaling — show raw depth values as loaded")
    args = parser.parse_args()

    sources = [s.strip() for s in args.source.split(",")]
    masks_dir = os.path.join(args.scene_dir, "masks")
    id_strs   = get_id_strs(args.scene_dir)
    bins      = np.linspace(DEPTH_MIN, DEPTH_MAX, N_BINS + 1)

    out_path = args.out or os.path.join(DEBUG_DIR, f"depth_hist_{'_'.join(sources)}.mp4")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Compute per-source scale factors
    depth_dirs = []
    scales     = []
    labels     = []
    colors     = []
    for src in sources:
        if src not in DEPTH_DIRS:
            print(f"Unknown source: {src}, skipping")
            continue
        d_dir = os.path.join(args.scene_dir, DEPTH_DIRS[src])
        if not os.path.isdir(d_dir):
            print(f"Missing depth dir for {src}: {d_dir}, skipping")
            continue
        scale = 1.0 if args.no_scale else compute_scale(d_dir, masks_dir, id_strs)
        print(f"  {src}: scale={scale:.4f}")
        depth_dirs.append(d_dir)
        scales.append(scale)
        labels.append(src)
        colors.append(COLORS.get(src, "#888888"))

    if not depth_dirs:
        print("No valid sources found.")
        return

    # Sample every 10th frame to find global max density for fixed y-axis
    print("Sampling frames to compute global y-axis limit...")
    global_ymax = 0.0
    sample_ids = id_strs[::10]
    for id_str in sample_ids:
        for d_dir, scale in zip(depth_dirs, scales):
            d = load_depth(d_dir, id_str)
            if d is None:
                continue
            vals = (d * scale)[(d * scale > DEPTH_MIN) & (d * scale < DEPTH_MAX)]
            if len(vals) == 0:
                continue
            counts, _ = np.histogram(vals, bins=bins, density=True)
            global_ymax = max(global_ymax, float(counts.max()))
    global_ymax *= 1.05  # small headroom
    print(f"Global y-axis max: {global_ymax:.4f}")

    print(f"Rendering {len(id_strs)} frames → {out_path}")

    writer = None
    for i, id_str in enumerate(id_strs):
        depths_scaled = []
        for d_dir, scale in zip(depth_dirs, scales):
            d = load_depth(d_dir, id_str)
            depths_scaled.append(d * scale if d is not None else None)

        frame = render_histogram(depths_scaled, labels, colors, i, len(id_strs), bins, args.layout, global_ymax)

        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))

        writer.write(frame)

        if i % 100 == 0:
            print(f"  [{i}/{len(id_strs)}]")

    if writer:
        writer.release()
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
