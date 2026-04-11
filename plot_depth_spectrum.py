"""
Per-frame depth spectrum comparison across multiple depth sources.

For each source, computes mean depth over the duck mask per frame and plots:
  - Left panel:  raw (unscaled) depth values
  - Right panel: scaled depth using binary_search_depth at frame 0 (same as
                 tracking pipeline) so all sources are in true metric space.

Usage:
    python plot_depth_spectrum.py \
        --scene_dir /work/courses/3dv/team22/foundationpose/data/20250804_104715 \
        --mesh_file /work/courses/3dv/team22/foundationpose/data/object/duck/duck.obj \
        --out /work/courses/3dv/team22/foundationpose/debug/depth_spectrum.png
"""
import os, argparse, glob
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from estimater import *
from datareader import *
from tools import *
import trimesh


SCENE_DIR  = "/work/courses/3dv/team22/foundationpose/data/20250804_104715"
DEBUG_DIR  = "/work/courses/3dv/team22/foundationpose/debug"
MESH_FILE  = "/work/courses/3dv/team22/foundationpose/data/object/duck/duck.obj"

# ── Depth sources (name, subdir relative to scene_dir) ────────────────────────
SOURCES = [
    ("VDA",        "depth"),
    ("DA3",        "depth_da3"),
    ("Metric3D",   "depth_metric3d"),
    ("Depth Pro",  "depth_pro"),
    ("VGGT",       "depth_vggt"),
]

COLORS     = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]


def load_depth(depth_dir, id_str):
    raw = cv2.imread(os.path.join(depth_dir, f"{id_str}.png"), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    return raw.astype(np.float32) / 1000.0


def load_mask(masks_dir, id_str):
    m = cv2.imread(os.path.join(masks_dir, f"{id_str}.png"), cv2.IMREAD_GRAYSCALE)
    return (m > 127) if m is not None else None


def get_id_strs(scene_dir):
    files = sorted(glob.glob(os.path.join(scene_dir, "rgb", "*.png")))
    return [os.path.splitext(os.path.basename(f))[0] for f in files]


def compute_bsd_scale(depth_dir, masks_dir, id_str0, color0, K, est, mesh):
    """Compute depth scale via binary_search_depth (same as tracking pipeline)."""
    d0   = load_depth(depth_dir, id_str0)
    m0   = load_mask(masks_dir, id_str0)
    if d0 is None or m0 is None or not m0.any():
        return 1.0
    pose = binary_search_depth(est, mesh, color0, m0.astype(bool), K, debug=False)
    bsd_z = float(pose[2, 3])
    raw_z = float(d0[m0].mean()) if m0.any() else 1.0
    scale = bsd_z / raw_z if raw_z > 0 else 1.0
    print(f"    bsd_z={bsd_z:.3f}m  raw_z={raw_z:.3f}m  scale={scale:.4f}")
    return scale


def compute_series(depth_dir, masks_dir, id_strs):
    means, stds = [], []
    for id_str in id_strs:
        d = load_depth(depth_dir, id_str)
        m = load_mask(masks_dir, id_str)
        if d is None or m is None or not m.any():
            means.append(np.nan); stds.append(np.nan)
        else:
            vals = d[m]; vals = vals[vals > 0.01]
            means.append(float(np.nanmean(vals)) if len(vals) else np.nan)
            stds.append(float(np.nanstd(vals))   if len(vals) else np.nan)
    return np.array(means), np.array(stds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", default=SCENE_DIR)
    parser.add_argument("--mesh_file", default=MESH_FILE)
    parser.add_argument("--out", default=os.path.join(DEBUG_DIR, "depth_spectrum.png"))
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    masks_dir = os.path.join(args.scene_dir, "masks")
    reader    = YcbineoatReader(video_dir=args.scene_dir, shorter_side=None, zfar=np.inf)
    id_strs   = get_id_strs(args.scene_dir)
    frames    = np.arange(len(id_strs))
    color0    = reader.get_color(0)
    K         = reader.K

    print(f"Found {len(id_strs)} frames, initialising FoundationPose for BSD scaling...")
    mesh = trimesh.load(args.mesh_file)
    mesh.apply_scale(0.001)

    scorer  = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx   = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
        scorer=scorer, refiner=refiner, debug_dir="/tmp/bsd_cal", debug=0, glctx=glctx,
    )

    # ── Collect series + BSD scale per source ─────────────────────────────────
    all_means, all_stds, all_scales, labels = [], [], [], []
    for name, subdir in SOURCES:
        depth_dir = os.path.join(args.scene_dir, subdir)
        if not os.path.isdir(depth_dir):
            print(f"  Skipping {name}: {depth_dir} not found")
            continue
        print(f"  Processing {name}...")
        scale = compute_bsd_scale(depth_dir, masks_dir, id_strs[0], color0, K, est, mesh)
        means, stds = compute_series(depth_dir, masks_dir, id_strs)
        all_means.append(means)
        all_stds.append(stds)
        all_scales.append(scale)
        labels.append(name)

    if not labels:
        print("No depth sources found.")
        return

    # ── Plot ───────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=False)
    fig.suptitle("Per-frame mean depth over duck mask", fontsize=13, fontweight="bold")

    titles = ["Raw (unscaled)", "Scaled (binary_search_depth frame-0)"]
    for ax, title, use_scale in zip(axes, titles, [False, True]):
        for i, (means, stds, label) in enumerate(zip(all_means, all_stds, labels)):
            s  = all_scales[i] if use_scale else 1.0
            m  = means * s
            sd = stds  * s
            color = COLORS[i % len(COLORS)]
            ls    = LINESTYLES[i % len(LINESTYLES)]
            ax.plot(frames, m, color=color, linewidth=1.5, linestyle=ls, label=label)
            ax.fill_between(frames, m - sd, m + sd, color=color, alpha=0.12)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Depth (m)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, len(id_strs))

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
