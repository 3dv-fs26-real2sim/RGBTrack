"""SAM2VP with CAD-anchored point prompts on mask drop.

Two-pass:
  Pass 1: run SAM2VP normally with seed mask, record per-frame areas.
  Pass 2: re-init SAM2VP. For each frame whose mask area dropped below
          AREA_RATIO_LO * last_good_area (and continuing until area
          recovers above AREA_RATIO_HI * last_good_area), inject a
          positive POINT prompt at the centroid of the CAD silhouette
          rendered from the known FP pose. SAM2 finds the whole duck
          using the point as a "look here" suggestion.
"""
import argparse
import glob
import os

import cv2
import numpy as np
import torch
import trimesh

from sam2.build_sam import build_sam2_video_predictor
from datareader import YcbineoatReader
from tools import render_cad_mask

SAM2_CHECKPOINT = "/work/courses/3dv/team22/RGBTrack/segment-anything-2-real-time/sam2.1_hiera_small.pt"
SAM2_CONFIG     = "configs/sam2.1/sam2.1_hiera_s.yaml"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir",    required=True)
    ap.add_argument("--ob_in_cam",    required=True)
    ap.add_argument("--mesh_file",    required=True)
    ap.add_argument("--mask_out_dir", default=None)
    ap.add_argument("--jpg_dir",      default=None)
    ap.add_argument("--area_lo", type=float, default=0.85,
                    help="Mask area drop ratio that triggers point prompt")
    ap.add_argument("--area_hi", type=float, default=0.95,
                    help="Mask area recovery ratio that stops point prompts")
    ap.add_argument("--prompt_interval", type=int, default=30,
                    help="While lost, attempt point prompt every N frames")
    ap.add_argument("--sam2_checkpoint", default=SAM2_CHECKPOINT)
    ap.add_argument("--sam2_config",     default=SAM2_CONFIG)
    args = ap.parse_args()

    mesh = trimesh.load(args.mesh_file)
    mesh.apply_scale(0.001)

    reader   = YcbineoatReader(video_dir=args.scene_dir, shorter_side=None, zfar=np.inf)
    jpg_dir  = args.jpg_dir  or os.path.join(args.scene_dir, "rgb_jpg")
    mask_dir = args.mask_out_dir or os.path.join(args.scene_dir, "masks_cad_anchored")
    os.makedirs(mask_dir, exist_ok=True)

    h, w = reader.get_color(0).shape[:2]
    N    = len(reader.color_files)

    # Load frame-0 mask as seed
    mask0_path = os.path.join(args.scene_dir, "masks", "000000.png")
    mask0 = (cv2.imread(mask0_path, cv2.IMREAD_GRAYSCALE) > 127).astype(bool)

    predictor   = build_sam2_video_predictor(args.sam2_config, args.sam2_checkpoint,
                                             device="cuda")
    frame_files = sorted(glob.glob(os.path.join(args.scene_dir, "rgb", "*.png")))
    from scipy.ndimage import binary_fill_holes

    # ── Pass 1: free SAM2VP run, record per-frame mask areas ─────────────────
    print("Pass 1: free SAM2VP run to detect drops...")
    areas = np.zeros(N, dtype=np.float64)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_path=jpg_dir,
                                     offload_video_to_cpu=True,
                                     offload_state_to_cpu=True)
        predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=mask0)
        for fi, _, mlogits in predictor.propagate_in_video(state):
            m = (mlogits[0] > 0.0).cpu().numpy()
            if m.ndim == 3:
                m = m[0]
            areas[fi] = float(m.sum())

    # ── Identify frames needing CAD point prompt ─────────────────────────────
    # While in a "lost" period (mask area dropped below area_lo * last_good),
    # only attempt a prompt every prompt_interval frames. Stop on recovery.
    last_good = areas[0]
    frames_to_prompt = []
    in_drop = False
    drop_start = -1
    for i in range(1, N):
        if not in_drop and areas[i] < args.area_lo * last_good:
            in_drop = True
            drop_start = i
        if in_drop:
            if (i - drop_start) % args.prompt_interval == 0:
                frames_to_prompt.append(i)
            if areas[i] > args.area_hi * last_good:
                in_drop = False
        else:
            last_good = max(last_good, areas[i])
    print(f"Pass 2: prompting {len(frames_to_prompt)} frames with CAD centroid points")

    # Free Pass 1 state before allocating Pass 2 to avoid OOM on 15 GB GPU
    predictor.reset_state(state)
    del state
    torch.cuda.empty_cache()

    # ── Pass 2: re-init SAM2VP with point prompts at drop frames ─────────────
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_path=jpg_dir,
                                     offload_video_to_cpu=True,
                                     offload_state_to_cpu=True)
        predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=mask0)

        for i in frames_to_prompt:
            pose_path = os.path.join(args.ob_in_cam, f"{reader.id_strs[i]}.txt")
            if not os.path.exists(pose_path):
                continue
            pose     = np.loadtxt(pose_path).reshape(4, 4)
            cad_mask = render_cad_mask(pose, mesh, reader.K, w=w, h=h)
            if cad_mask is None or not cad_mask.any():
                continue
            ys, xs = np.where(cad_mask > 0)
            cy, cx = int(ys.mean()), int(xs.mean())
            point_coords = np.array([[cx, cy]], dtype=np.float32)
            point_labels = np.array([1], dtype=np.int32)
            predictor.add_new_points_or_box(
                state, frame_idx=i, obj_id=1,
                points=point_coords, labels=point_labels,
            )

        for fi, _, mlogits in predictor.propagate_in_video(state):
            m = (mlogits[0] > 0.0).cpu().numpy()
            if m.ndim == 3:
                m = m[0]
            m = binary_fill_holes(m).astype(np.uint8) * 255
            name = os.path.splitext(os.path.basename(frame_files[fi]))[0]
            cv2.imwrite(os.path.join(mask_dir, f"{name}.png"), m)
            if fi % 100 == 0:
                print(f"frame {fi}/{N}")

    print(f"Done → {mask_dir}")
