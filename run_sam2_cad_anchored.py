"""SAM2VP with continuous CAD anchoring.

For each frame, renders the CAD model at the known FP pose and injects it
as an add_new_mask prompt before propagation. SAM2 snaps to real image
edges while being continuously disciplined by the rigid duck shape.

Usage:
    python run_sam2_cad_anchored.py \
        --scene_dir   <scene> \
        --ob_in_cam   <debug>/ob_in_cam \
        --mesh_file   <duck.obj> \
        --anchor_every 1   # prompt every N frames (1 = every frame)
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
    ap.add_argument("--anchor_every", type=int, default=1,
                    help="Inject CAD prompt every N frames (default 1 = every frame)")
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

    predictor = build_sam2_video_predictor(args.sam2_config, args.sam2_checkpoint,
                                           device="cuda")

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(
            video_path=jpg_dir,
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
        )

        # Frame 0: seed with existing clean mask
        predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=mask0)

        # Inject CAD masks at subsequent frames as anchors
        for i in range(1, N):
            if i % args.anchor_every != 0:
                continue
            pose_path = os.path.join(args.ob_in_cam, f"{reader.id_strs[i]}.txt")
            if not os.path.exists(pose_path):
                continue
            pose     = np.loadtxt(pose_path).reshape(4, 4)
            cad_mask = render_cad_mask(pose, mesh, reader.K, w=w, h=h)
            if cad_mask is not None and cad_mask.any():
                predictor.add_new_mask(state, frame_idx=i, obj_id=1,
                                       mask=cad_mask.astype(bool))

        # Propagate — SAM2 snaps each frame to CAD prior + image edges
        from scipy.ndimage import binary_fill_holes
        frame_files = sorted(glob.glob(os.path.join(args.scene_dir, "rgb", "*.png")))
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            mask = (mask_logits[0] > 0.0).cpu().numpy()
            if mask.ndim == 3:
                mask = mask[0]
            mask = binary_fill_holes(mask).astype(np.uint8) * 255
            frame_name = os.path.splitext(os.path.basename(frame_files[frame_idx]))[0]
            cv2.imwrite(os.path.join(mask_dir, f"{frame_name}.png"), mask)
            if frame_idx % 100 == 0:
                print(f"frame {frame_idx}/{N}")

    print(f"Done → {mask_dir}")
