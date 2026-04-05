"""
Metric3D depth estimator wrapper (torch.hub based, no pip install needed).

Checkpoint download (run once):
    wget https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_large_800k.pth \
         -O /work/courses/3dv/team22/metric3d_vit_large.pth
"""

import cv2
import numpy as np
import torch
import sys
import os

METRIC3D_DIR = "/work/courses/3dv/team22/Metric3D"

# Fixed canonical input size for Metric3D ViT-Large (crop_size from vit.raft5.large.py)
CANONICAL_SIZE = (616, 1064)  # (H, W)


class Metric3DWrapper:
    def __init__(self, checkpoint_path, model_type="metric3d_vit_large", device="cuda"):
        self.device = device
        model = torch.hub.load(METRIC3D_DIR, model_type, source="local", pretrained=False)
        state = torch.load(checkpoint_path, map_location="cpu")
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)
        self.model = model.to(device).eval()

    def estimate(self, rgb, K):
        """
        rgb: (H, W, 3) uint8 RGB
        K:   (3, 3) intrinsic matrix

        Returns depth: (H, W) float32 metric depth in metres
        """
        h, w = rgb.shape[:2]
        fx = float(K[0, 0])

        # resize to canonical size
        target_h, target_w = CANONICAL_SIZE
        rgb_resized = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        scale_w = target_w / w
        scale_h = target_h / h
        # adjusted focal length for the rescaled image
        fx_scaled = fx * scale_w

        mean = torch.tensor([123.675, 116.28, 103.53], device=self.device).view(3, 1, 1)
        std  = torch.tensor([58.395, 57.12, 57.375],  device=self.device).view(3, 1, 1)

        img_t = torch.from_numpy(rgb_resized).permute(2, 0, 1).float().to(self.device)
        img_t = (img_t - mean) / std
        img_t = img_t.unsqueeze(0)  # (1, 3, H, W)

        with torch.no_grad():
            pred_depth, _, _ = self.model.inference({"input": img_t})

        depth = pred_depth.squeeze().cpu().numpy()
        # resize depth back to original resolution
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        # correct metric scale: depth from canonical is for fx_scaled, rescale to fx
        depth = depth * (fx_scaled / fx)
        return depth.astype(np.float32)
