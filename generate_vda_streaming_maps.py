"""
Generate metric Video-Depth-Anything depth maps with temporal consistency.

Uses VideoDepthAnything streaming inference (infer_video_depth_one) so each
frame is processed with context from all previous frames.

Output: uint16 PNG per frame in millimetres — same format as VDA/DA3 PNGs.

Run from the cluster with the vda conda env:
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate /work/scratch/$USER/conda/envs/vda

Usage:
    python generate_vda_streaming_maps.py \
        --scene_dir  /work/courses/3dv/team22/foundationpose/data/20250804_104715 \
        --vda_repo   /home/jaerkim/Video-Depth-Anything \
        --ckpt_dir   /home/jaerkim/Video-Depth-Anything/checkpoints \
        --out_dir    /work/courses/3dv/team22/foundationpose/data/20250804_104715/depth_vda_streaming
"""
import argparse, os, sys, glob
import numpy as np
import cv2
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", required=True)
    parser.add_argument("--vda_repo",  required=True,
                        help="Path to Video-Depth-Anything repo root")
    parser.add_argument("--ckpt_dir",  default=None,
                        help="Checkpoint dir (defaults to vda_repo/checkpoints)")
    parser.add_argument("--out_dir",   required=True)
    parser.add_argument("--rgb_dir",   default=None,
                        help="Override RGB frames dir (default: scene_dir/rgb)")
    parser.add_argument("--encoder",   default="vitl", choices=["vits", "vitb", "vitl"])
    parser.add_argument("--input_size",type=int, default=518)
    parser.add_argument("--fp32",      action="store_true")
    args = parser.parse_args()

    # ── Import VideoDepthAnything from repo ────────────────────────────────────
    sys.path.insert(0, args.vda_repo)
    from video_depth_anything.video_depth_stream import VideoDepthAnything

    ckpt_dir = args.ckpt_dir or os.path.join(args.vda_repo, "checkpoints")
    ckpt_path = os.path.join(ckpt_dir, f"metric_video_depth_anything_{args.encoder}.pth")
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load model ─────────────────────────────────────────────────────────────
    model_configs = {
        "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }
    model = VideoDepthAnything(**model_configs[args.encoder])
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)
    model = model.to(device).eval()
    print(f"Loaded metric checkpoint: {ckpt_path}")

    # ── Process frames ─────────────────────────────────────────────────────────
    rgb_dir = args.rgb_dir or os.path.join(args.scene_dir, "rgb")
    color_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    id_strs = [os.path.splitext(os.path.basename(f))[0] for f in color_files]
    print(f"Processing {len(color_files)} frames → {args.out_dir}")

    for i, (color_file, id_str) in enumerate(zip(color_files, id_strs)):
        color_bgr = cv2.imread(color_file)
        frame_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

        # Streaming inference — maintains temporal context internally
        depth = model.infer_video_depth_one(frame_rgb, input_size=args.input_size,
                                             device=device, fp32=args.fp32)
        # depth is float32 in metres (metric model)
        depth_mm = (depth * 1000.0).clip(0, 65535).astype(np.uint16)
        cv2.imwrite(os.path.join(args.out_dir, f"{id_str}.png"), depth_mm)

        if i % 100 == 0:
            print(f"  [{i}/{len(color_files)}] depth range [{depth.min():.3f}, {depth.max():.3f}]m")

    print("Done.")


if __name__ == "__main__":
    main()
