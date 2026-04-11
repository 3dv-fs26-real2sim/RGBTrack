"""
Per-frame depth spectrum comparison across multiple depth sources.

For each source, computes mean depth over the duck mask per frame and plots:
  - Left panel:  raw (unscaled) depth values
  - Right panel: scaled depth (each source normalised so frame-0 duck mean matches VDA)

Usage:
    python plot_depth_spectrum.py \
        --scene_dir /work/courses/3dv/team22/foundationpose/data/20250804_104715 \
        --out /work/courses/3dv/team22/foundationpose/debug/depth_spectrum.png
"""
import os, argparse, glob
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


SCENE_DIR = "/work/courses/3dv/team22/foundationpose/data/20250804_104715"
DEBUG_DIR  = "/work/courses/3dv/team22/foundationpose/debug"

# ── Depth sources (name, subdir relative to scene_dir) ────────────────────────
SOURCES = [
    ("VDA",        "depth"),
    ("DA3",        "depth_da3"),
    ("Metric3D",   "depth_metric3d"),
    ("Depth Pro",  "depth_pro"),
    ("VGGT",       "depth_vggt"),
]

COLORS      = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
LINESTYLES  = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]


def load_depth(depth_dir, id_str):
    path = os.path.join(depth_dir, f"{id_str}.png")
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    return raw.astype(np.float32) / 1000.0  # mm → m


def load_mask(masks_dir, id_str):
    path = os.path.join(masks_dir, f"{id_str}.png")
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    return m > 127


def get_id_strs(scene_dir):
    rgb_files = sorted(glob.glob(os.path.join(scene_dir, "rgb", "*.png")))
    return [os.path.splitext(os.path.basename(f))[0] for f in rgb_files]


def compute_series(depth_dir, masks_dir, id_strs):
    """Return (means, stds) arrays over duck mask pixels per frame. NaN if missing."""
    means, stds = [], []
    for id_str in id_strs:
        d = load_depth(depth_dir, id_str)
        m = load_mask(masks_dir, id_str)
        if d is None or m is None or not m.any():
            means.append(np.nan)
            stds.append(np.nan)
        else:
            vals = d[m]
            vals = vals[vals > 0.01]
            means.append(float(np.nanmean(vals)) if len(vals) else np.nan)
            stds.append(float(np.nanstd(vals)) if len(vals) else np.nan)
    return np.array(means), np.array(stds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", default=SCENE_DIR)
    parser.add_argument("--out", default=os.path.join(DEBUG_DIR, "depth_spectrum.png"))
    args = parser.parse_args()

    masks_dir = os.path.join(args.scene_dir, "masks")
    id_strs   = get_id_strs(args.scene_dir)
    frames    = np.arange(len(id_strs))

    print(f"Found {len(id_strs)} frames, {len(SOURCES)} sources")

    # ── Collect series ─────────────────────────────────────────────────────────
    all_means, all_stds, labels = [], [], []
    for name, subdir in SOURCES:
        depth_dir = os.path.join(args.scene_dir, subdir)
        if not os.path.isdir(depth_dir):
            print(f"  Skipping {name}: {depth_dir} not found")
            continue
        print(f"  Processing {name}...")
        means, stds = compute_series(depth_dir, masks_dir, id_strs)
        all_means.append(means)
        all_stds.append(stds)
        labels.append(name)

    if not labels:
        print("No depth sources found.")
        return

    # ── Scale factors: normalise frame-0 duck mean to VDA (or first source) ───
    ref_mean0 = all_means[0][0]  # frame-0 mean of reference source (VDA)
    scales = []
    for means in all_means:
        m0 = means[0]
        scales.append(ref_mean0 / m0 if (m0 and not np.isnan(m0)) else 1.0)

    # ── Plot ───────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=False)
    fig.suptitle("Per-frame mean depth over duck mask", fontsize=13, fontweight="bold")

    for ax, title, use_scale in zip(axes, ["Raw (unscaled)", "Scaled (frame-0 normalised)"], [False, True]):
        for i, (means, stds, label) in enumerate(zip(all_means, all_stds, labels)):
            s = scales[i] if use_scale else 1.0
            m = means * s
            sd = stds * s
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
