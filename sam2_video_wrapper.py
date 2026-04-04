import os
import glob
import numpy as np
import cv2
import torch
from sam2.build_sam import build_sam2_video_predictor


def generate_masks_sam2_video(
    scene_dir,
    checkpoint_path,
    config="configs/sam2.1/sam2.1_hiera_s.yaml",
    device="cuda",
):
    """
    Run SAM2VideoPredictor on all RGB frames in scene_dir/rgb/.
    Uses the existing frame-0 mask from scene_dir/masks/000000.png.
    Saves per-frame masks to scene_dir/masks/<frame_id>.png (overwrites frame 0 too).
    """
    rgb_dir = os.path.join(scene_dir, "rgb")
    jpg_dir = os.path.join(scene_dir, "rgb_jpg")
    mask_dir = os.path.join(scene_dir, "masks")
    os.makedirs(mask_dir, exist_ok=True)

    frame0_mask_path = os.path.join(mask_dir, "000000.png")
    assert os.path.exists(frame0_mask_path), f"Frame 0 mask not found: {frame0_mask_path}"

    frame0_mask = cv2.imread(frame0_mask_path, cv2.IMREAD_GRAYSCALE)
    frame0_mask = (frame0_mask > 127).astype(bool)

    predictor = build_sam2_video_predictor(config, checkpoint_path, device=device)

    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        state = predictor.init_state(
            video_path=jpg_dir,
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
        )

        # Add frame-0 mask as the prompt
        predictor.add_new_mask(
            inference_state=state,
            frame_idx=0,
            obj_id=1,
            mask=frame0_mask,
        )

        # Propagate forward through all frames
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            mask = (mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8) * 255
            if mask.ndim == 3:
                mask = mask[0]
            frame_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
            frame_name = os.path.splitext(os.path.basename(frame_files[frame_idx]))[0]
            cv2.imwrite(os.path.join(mask_dir, f"{frame_name}.png"), mask)
            if frame_idx % 100 == 0:
                print(f"Processed frame {frame_idx}/{len(frame_files)}")

    print(f"Done. Masks saved to {mask_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", required=True)
    parser.add_argument("--checkpoint", default="/work/courses/3dv/team22/RGBTrack/segment-anything-2-real-time/sam2.1_hiera_small.pt")
    args = parser.parse_args()

    generate_masks_sam2_video(
        scene_dir=args.scene_dir,
        checkpoint_path=args.checkpoint,
    )
