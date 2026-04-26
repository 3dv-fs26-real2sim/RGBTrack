"""SAM2VP guided by rendered CAD silhouettes from existing FP poses.

Single-pass: inject the rendered 3D CAD mask as add_new_mask every
anchor_interval frames. SAM2VP interpolates between anchors and
snaps to the full duck shape (including hat) rather than drifting
to a partial lower-body mask.
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
from scipy.ndimage import binary_fill_holes

SAM2_CHECKPOINT = "/work/courses/3dv/team22/RGBTrack/segment-anything-2-real-time/sam2.1_hiera_small.pt"
SAM2_CONFIG     = "configs/sam2.1/sam2.1_hiera_s.yaml"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir",       required=True)
    ap.add_argument("--ob_in_cam",       required=True,
                    help="Directory of FP pose .txt files (one per frame)")
    ap.add_argument("--mesh_file",       required=True)
    ap.add_argument("--mask_out_dir",    default=None)
    ap.add_argument("--jpg_dir",         default=None)
    ap.add_argument("--anchor_interval", type=int, default=5,
                    help="Inject CAD mask every N frames")
    ap.add_argument("--sam2_checkpoint", default=SAM2_CHECKPOINT)
    ap.add_argument("--sam2_config",     default=SAM2_CONFIG)
    args = ap.parse_args()

    mesh = trimesh.load(args.mesh_file)
    mesh.apply_scale(0.001)

    reader   = YcbineoatReader(video_dir=args.scene_dir, shorter_side=None, zfar=np.inf)
    jpg_dir  = args.jpg_dir or os.path.join(args.scene_dir, "rgb_jpg")
    mask_dir = args.mask_out_dir or os.path.join(args.scene_dir, "masks_cad_guided")
    os.makedirs(mask_dir, exist_ok=True)

    h, w = reader.get_color(0).shape[:2]
    N    = len(reader.color_files)

    frame_files = sorted(glob.glob(os.path.join(args.scene_dir, "rgb", "*.png")))

    # Seed mask from frame 0
    mask0_path = os.path.join(args.scene_dir, "masks", "000000.png")
    mask0 = (cv2.imread(mask0_path, cv2.IMREAD_GRAYSCALE) > 127).astype(bool)

    predictor = build_sam2_video_predictor(args.sam2_config, args.sam2_checkpoint,
                                           device="cuda")

    anchors_injected = 0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_path=jpg_dir,
                                     offload_video_to_cpu=True,
                                     offload_state_to_cpu=True)

        # Frame 0: seed with existing SAM2 mask
        predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=mask0)

        # Inject rendered CAD silhouette every anchor_interval frames
        for i in range(1, N):
            if i % args.anchor_interval != 0:
                continue
            pose_path = os.path.join(args.ob_in_cam, f"{reader.id_strs[i]}.txt")
            if not os.path.exists(pose_path):
                continue
            pose     = np.loadtxt(pose_path).reshape(4, 4)
            cad_mask = render_cad_mask(pose, mesh, reader.K, w=w, h=h)
            if cad_mask is None or not cad_mask.any():
                continue
            predictor.add_new_mask(state, frame_idx=i, obj_id=1,
                                   mask=cad_mask.astype(bool))
            anchors_injected += 1

        print(f"Injected {anchors_injected} CAD anchors every {args.anchor_interval} frames")
        print("Propagating...")

        for fi, _, mlogits in predictor.propagate_in_video(state):
            m = (mlogits[0] > 0.0).cpu().numpy()
            if m.ndim == 3:
                m = m[0]
            m = binary_fill_holes(m).astype(np.uint8) * 255
            name = os.path.splitext(os.path.basename(frame_files[fi]))[0]
            cv2.imwrite(os.path.join(mask_dir, f"{name}.png"), m)
            if fi % 100 == 0:
                print(f"  frame {fi}/{N}")

    print(f"Done → {mask_dir}")
