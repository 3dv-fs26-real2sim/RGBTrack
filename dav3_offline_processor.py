"""
DA3 offline depth processor — Depth Anything 3, sliding-window batching.

Generates uint16 depth PNGs (mm) from a scene directory using DA3's
multi-view inference. Consecutive chunks overlap and are linearly blended
to prevent temporal seams at chunk boundaries.

Sliding-window blending:
  Chunk A covers frames [s, s+C).
  Chunk B covers frames [s+C-O, s+2C-O).
  In the overlap region [s+C-O, s+C), frame k gets:
      t = (k - (s+C-O)) / O          # 0 → 1 across overlap
      depth[k] = (1-t)*depthA[k] + t*depthB[k]
  This cross-fades smoothly between the two globally-consistent estimates.
  Scale alignment (median ratio) is applied before blending to correct
  any per-chunk scale shift.

Model choices (--model):
  da3-small        ~0.08B  fast, fits 5060 Ti easily
  da3-base         ~0.12B
  da3-large        ~0.35B  recommended general purpose
  da3-giant        ~1.15B  best quality, needs 5090
  da3metric-large  ~0.35B  metric depth + sky segmentation (default)
  da3nested-giant-large  ~1.40B  maximum quality

Usage:
  # CPU debug (no GPU, small model, tiny chunks)
  python dav3_offline_processor.py --scene_dir /path/to/scene --debug

  # Cluster (5060 Ti)
  python dav3_offline_processor.py \\
      --scene_dir /work/.../20250804_104715 \\
      --model da3metric-large \\
      --chunk_size 24 --overlap 6

  # Workstation (5090)
  python dav3_offline_processor.py \\
      --scene_dir /work/.../20250804_104715 \\
      --model da3-giant \\
      --chunk_size 64 --overlap 16
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

# ── Import DA3 from sibling DepthAnything3/ subfolder ──────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "DepthAnything3"))
from depth_anything_3.api import DepthAnything3
# ───────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="DA3 offline depth processor with sliding-window blending."
    )
    p.add_argument("--scene_dir", required=True,
                   help="Scene directory containing rgb/ subfolder.")
    p.add_argument("--out_dir", default=None,
                   help="Output dir for PNGs. Default: scene_dir/depth_da3v3")
    p.add_argument("--model", default="da3metric-large",
                   choices=["da3-small", "da3-base", "da3-large", "da3-giant",
                            "da3metric-large", "da3nested-giant-large"],
                   help="Model preset. Use da3-small for 5060 Ti, da3-giant for 5090.")
    p.add_argument("--chunk_size", type=int, default=24,
                   help="Frames per inference chunk. Reduce if OOM.")
    p.add_argument("--overlap", type=int, default=6,
                   help="Overlap frames between consecutive chunks for blending.")
    p.add_argument("--process_res", type=int, default=504,
                   help="Processing resolution (lower = faster/less VRAM).")
    p.add_argument("--cam_K", type=str, default=None,
                   help="Path to cam_K.txt (3x3 intrinsics). Enables pose-conditioned mode.")
    p.add_argument("--debug", action="store_true",
                   help="CPU mode: uses da3-small, tiny chunks. For local logic testing.")
    p.add_argument("--save_npy", action="store_true",
                   help="Also save raw float32 depth as .npy alongside PNGs.")
    return p.parse_args()


# ── Helpers ────────────────────────────────────────────────────────────────

def load_K(path):
    return np.loadtxt(path).reshape(3, 3).astype(np.float32)


def blend_weights(overlap):
    """
    Linear fade across `overlap` frames.
    w_old: 1 → 0  (previous chunk fades out)
    w_new: 0 → 1  (incoming chunk fades in)
    """
    t = np.linspace(0.0, 1.0, overlap, endpoint=True, dtype=np.float32)
    return 1.0 - t, t   # w_old, w_new


def align_scale(depth_ref, depth_new, mask=None):
    """
    Compute median-ratio scale so that depth_new ~= scale * depth_new
    matches depth_ref in the overlapping region.
    Returns scale factor (scalar float).
    """
    if mask is None:
        mask = (depth_ref > 0.01) & (depth_new > 0.01)
    if mask.sum() < 10:
        return 1.0
    ratio = depth_ref[mask] / depth_new[mask]
    return float(np.median(ratio))


def save_frame(depth_m, out_dir, id_str, save_npy=False):
    d_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
    cv2.imwrite(os.path.join(out_dir, f"{id_str}.png"), d_mm)
    if save_npy:
        np.save(os.path.join(out_dir, f"{id_str}.npy"), depth_m.astype(np.float32))


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Device / model override in debug mode ──────────────────────────────
    device     = "cpu" if args.debug else ("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "da3-small"      if args.debug else args.model
    chunk_size = min(args.chunk_size, 6) if args.debug else args.chunk_size
    overlap    = min(args.overlap,    2) if args.debug else args.overlap
    use_amp    = device == "cuda"

    assert overlap < chunk_size, "--overlap must be smaller than --chunk_size"

    out_dir = args.out_dir or os.path.join(args.scene_dir, "depth_da3v3")
    os.makedirs(out_dir, exist_ok=True)

    K = load_K(args.cam_K) if args.cam_K else None

    # ── Collect frames ──────────────────────────────────────────────────────
    color_files = sorted(glob.glob(os.path.join(args.scene_dir, "rgb", "*.png")))
    id_strs     = [os.path.splitext(os.path.basename(f))[0] for f in color_files]
    N           = len(color_files)
    print(f"Found {N} frames  →  {out_dir}")
    print(f"Model: {model_name}  device: {device}  chunk: {chunk_size}  overlap: {overlap}")

    if N == 0:
        raise RuntimeError(f"No PNGs found in {args.scene_dir}/rgb/")

    # ── Load model ──────────────────────────────────────────────────────────
    print("Loading model...")
    model = DepthAnything3(model_name=model_name).to(device)
    if use_amp:
        model = model.half()
    model.eval()
    print("Model ready.")

    # ── Sliding-window plan ─────────────────────────────────────────────────
    stride = chunk_size - overlap
    starts = list(range(0, N, stride))
    # Ensure the last chunk reaches the final frame
    if starts[-1] + chunk_size < N:
        starts.append(N - chunk_size)
    # Remove duplicate starts
    starts = sorted(set(starts))
    n_chunks = len(starts)

    w_old_arr, w_new_arr = blend_weights(overlap)  # shape (overlap,)

    # Depth accumulator: per-pixel weighted sum — allocate lazily after first chunk
    depth_acc  = None   # (N, H, W) float64
    weight_acc = None   # (N,)      float64  (scalar weight per frame is enough)

    # ── Process chunks ──────────────────────────────────────────────────────
    prev_depths = None   # depth output from previous chunk (for scale alignment)
    prev_end    = 0      # global end index of previous chunk

    for ci, start in enumerate(starts):
        end    = min(start + chunk_size, N)
        n      = end - start
        frames = color_files[start:end]
        print(f"[{ci+1}/{n_chunks}] frames {start}–{end-1}  ({n} frames)")

        images = [Image.open(f).convert("RGB") for f in frames]
        ixts   = (np.tile(K[None], (n, 1, 1)) if K is not None else None)

        # ── Inference ────────────────────────────────────────────────────
        with torch.no_grad():
            ctx = torch.autocast(device_type=device, dtype=torch.float16) if use_amp \
                  else torch.no_grad()
            with ctx:
                pred = model.inference(
                    image=images,
                    intrinsics=ixts,
                    ref_view_strategy="middle",
                    process_res=args.process_res,
                )

        chunk_depths = pred.depth.astype(np.float64)  # (n, H, W)
        H, W = chunk_depths.shape[1], chunk_depths.shape[2]

        # ── Lazy allocate accumulator ─────────────────────────────────────
        if depth_acc is None:
            depth_acc  = np.zeros((N, H, W), dtype=np.float64)
            weight_acc = np.zeros(N,          dtype=np.float64)

        # ── Scale alignment: match this chunk to previous in overlap region ─
        if ci > 0 and overlap > 0:
            ov_local_start = 0                     # overlap starts at j=0 of this chunk
            ov_local_end   = overlap
            ov_global      = slice(start, start + overlap)
            ref  = depth_acc[ov_global] / np.maximum(weight_acc[ov_global, None, None], 1e-9)
            new_ = chunk_depths[:overlap]
            scale = align_scale(ref.reshape(-1), new_.reshape(-1))
            chunk_depths *= scale
            print(f"  scale alignment: {scale:.4f}")

        # ── Accumulate with blend weights ─────────────────────────────────
        for j in range(n):
            fi = start + j   # global frame index

            # Determine weight for this frame
            if ci == 0:
                # First chunk: full weight for all frames
                w = 1.0
            else:
                ov_j = j  # j=0 is the first overlapping frame in this chunk
                if ov_j < overlap:
                    w = float(w_new_arr[ov_j])   # fade in (0 → 1)
                else:
                    w = 1.0                       # past overlap: full weight

            depth_acc[fi]  += w * chunk_depths[j]
            weight_acc[fi] += w

        # ── Save frames that are now fully determined ──────────────────────
        # A frame is finalised once no future chunk can update it.
        # Future chunks start at starts[ci+1] (if any); their overlap
        # region begins there, so frame fi < starts[ci+1] is done.
        save_up_to = starts[ci + 1] if ci + 1 < n_chunks else N
        for fi in range(prev_end, save_up_to):
            w_total = weight_acc[fi]
            if w_total < 1e-9:
                continue
            depth_final = (depth_acc[fi] / w_total).astype(np.float32)
            save_frame(depth_final, out_dir, id_strs[fi], save_npy=args.save_npy)

        prev_end = save_up_to

    # ── Save any remaining frames ───────────────────────────────────────────
    for fi in range(prev_end, N):
        w_total = weight_acc[fi]
        if w_total < 1e-9:
            continue
        depth_final = (depth_acc[fi] / w_total).astype(np.float32)
        save_frame(depth_final, out_dir, id_strs[fi], save_npy=args.save_npy)

    print(f"Done — {N} depth PNGs saved to {out_dir}")


if __name__ == "__main__":
    main()
