"""Generate hand masks for all frames using GroundingDINO + SAM2 image predictor.

For each frame, GroundingDINO detects the robot hand bounding box,
then SAM2 image predictor segments it precisely. Output: one binary
PNG per frame in --hand_mask_dir.
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch

GDINO_REPO = "/work/courses/3dv/team22/FoundationPose-plus-plus/sam-hq/seginw/GroundingDINO"
GDINO_CFG  = f"{GDINO_REPO}/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GDINO_CKPT = "/work/courses/3dv/team22/foundationpose/weights/groundingdino_swint_ogc.pth"
SAM2_CHECKPOINT = "/work/courses/3dv/team22/RGBTrack/segment-anything-2-real-time/sam2.1_hiera_small.pt"
SAM2_CONFIG     = "configs/sam2.1/sam2.1_hiera_s.yaml"

sys.path.insert(0, GDINO_REPO)

from groundingdino.util.inference import load_model, predict
from groundingdino.util import box_ops
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def load_image_gdino(path):
    from groundingdino.util.inference import load_image
    return load_image(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir",     required=True)
    ap.add_argument("--hand_mask_dir", default=None)
    ap.add_argument("--prompt",        default="robotic hand")
    ap.add_argument("--box_thresh",    type=float, default=0.30)
    ap.add_argument("--text_thresh",   type=float, default=0.25)
    ap.add_argument("--interval",      type=int, default=1,
                    help="Process every N frames (1=all frames)")
    ap.add_argument("--sam2_checkpoint", default=SAM2_CHECKPOINT)
    ap.add_argument("--sam2_config",     default=SAM2_CONFIG)
    args = ap.parse_args()

    hand_dir = args.hand_mask_dir or os.path.join(args.scene_dir, "masks_hand")
    os.makedirs(hand_dir, exist_ok=True)

    frame_files = sorted(glob.glob(os.path.join(args.scene_dir, "rgb", "*.png")))
    N = len(frame_files)
    print(f"Found {N} frames, processing every {args.interval}")

    print("Loading GroundingDINO...")
    gdino = load_model(GDINO_CFG, GDINO_CKPT)
    gdino.eval()

    print("Loading SAM2 image predictor...")
    sam2 = build_sam2(args.sam2_config, args.sam2_checkpoint, device="cuda")
    sam2_predictor = SAM2ImagePredictor(sam2)

    for i, fpath in enumerate(frame_files):
        if i % args.interval != 0:
            continue

        name = os.path.splitext(os.path.basename(fpath))[0]
        out_path = os.path.join(hand_dir, f"{name}.png")

        img_np, img_t = load_image_gdino(fpath)
        h, w = img_np.shape[:2]

        with torch.no_grad():
            boxes, logits, phrases = predict(
                model=gdino,
                image=img_t,
                caption=args.prompt,
                box_threshold=args.box_thresh,
                text_threshold=args.text_thresh,
            )

        if boxes.shape[0] == 0:
            # No detection — save empty mask
            cv2.imwrite(out_path, np.zeros((h, w), np.uint8))
            if i % 100 == 0:
                print(f"  frame {i}/{N}: no detection")
            continue

        # Take highest-confidence box
        best = logits.argmax()
        box_cx, box_cy, box_w, box_h = boxes[best].tolist()
        x1 = int((box_cx - box_w / 2) * w)
        y1 = int((box_cy - box_h / 2) * h)
        x2 = int((box_cx + box_w / 2) * w)
        y2 = int((box_cy + box_h / 2) * h)
        xyxy = np.array([[x1, y1, x2, y2]])

        sam2_predictor.set_image(img_np)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            masks, _, _ = sam2_predictor.predict(
                box=xyxy,
                multimask_output=False,
            )
        mask = (masks[0] > 0).astype(np.uint8) * 255
        cv2.imwrite(out_path, mask)

        if i % 100 == 0:
            print(f"  frame {i}/{N}: detected '{phrases[best]}' conf={logits[best]:.2f}")

    print(f"Done → {hand_dir}")
