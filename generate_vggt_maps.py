"""
Offline VGGT depth map generation.
VGGT outputs relative depth — saved as uint16 PNG with scale=10000.
Use --calibrate in the tracking job to fix the absolute scale.

Run with the vggt_env conda environment.
"""
import os, argparse, glob
import numpy as np
import cv2
import torch
import torch.nn.functional as F

VGGT_RESOLUTION = 518
DEPTH_SCALE = 10000  # relative depth * DEPTH_SCALE -> uint16 (relative, not metric)


def load_model(model_path, device):
    from vggt.models.vggt import VGGT
    model = VGGT()
    if model_path and os.path.exists(model_path):
        print(f"Loading weights from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    else:
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        print(f"Downloading weights from HuggingFace...")
        model.load_state_dict(torch.hub.load_state_dict_from_url(_URL, map_location="cpu"))
    model.eval()
    return model.to(device)


def preprocess_image(color_bgr, resolution=VGGT_RESOLUTION):
    """BGR uint8 HxWx3 → float tensor 1x3xRxR on GPU, square-cropped."""
    rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)  # 1x3xHxW
    t = F.interpolate(t, size=(resolution, resolution), mode="bilinear", align_corners=False)
    return t


def predict_depth(model, img_tensor, dtype, device):
    """img_tensor: 1x3xHxW → depth (H, W) numpy, conf (H, W) numpy."""
    img = img_tensor.to(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = img[None]  # 1x1x3xHxW  (batch x frames x C x H x W)
            aggregated_tokens_list, ps_idx = model.aggregator(images)
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    depth = depth_map.squeeze().cpu().float().numpy()   # (H, W)
    conf  = depth_conf.squeeze().cpu().float().numpy()
    return depth, conf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_scene_dir", type=str,
                        default="/work/courses/3dv/team22/foundationpose/data/20250804_104715")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Local path to VGGT model.pt (downloads if not given)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output dir. Defaults to test_scene_dir/depth_vggt/")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.test_scene_dir, "depth_vggt")
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    print(f"Device: {device}  dtype: {dtype}")

    model = load_model(args.model_path, device)
    print("VGGT loaded")

    color_files = sorted(glob.glob(os.path.join(args.test_scene_dir, "rgb", "*.png")))
    id_strs = [os.path.splitext(os.path.basename(f))[0] for f in color_files]
    print(f"Processing {len(color_files)} frames -> {out_dir}")

    for i, (color_file, id_str) in enumerate(zip(color_files, id_strs)):
        color = cv2.imread(color_file)
        img_t = preprocess_image(color)

        depth, conf = predict_depth(model, img_t, dtype, device)

        # Resize depth back to original resolution
        H, W = color.shape[:2]
        depth_resized = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)

        # Save as uint16 PNG with fixed scale (relative depth)
        depth_u16 = np.clip(depth_resized * DEPTH_SCALE, 0, 65535).astype(np.uint16)
        out_path = os.path.join(out_dir, f"{id_str}.png")
        cv2.imwrite(out_path, depth_u16)

        if i % 50 == 0:
            print(f"  [{i}/{len(color_files)}] depth range [{depth_resized.min():.4f}, {depth_resized.max():.4f}]  saved {out_path}")

    print(f"Done — {len(color_files)} depth maps saved to {out_dir}")
    print(f"Use --depth_dir {out_dir} --calibrate when tracking.")


if __name__ == "__main__":
    main()
