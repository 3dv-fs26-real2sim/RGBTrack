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
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        intrinsic = [fx, fy, cx, cy]

        # canonical focal length used by Metric3D (ViT models)
        canonical_focal = 1000.0
        scale = canonical_focal / fx
        new_w = int(w * scale)
        new_h = int(h * scale)
        rgb_resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        intrinsic_scaled = [canonical_focal, canonical_focal * fy / fx, cx * scale, cy * scale]

        mean = torch.tensor([123.675, 116.28, 103.53], device=self.device).view(3, 1, 1)
        std  = torch.tensor([58.395, 57.12, 57.375],  device=self.device).view(3, 1, 1)

        img_t = torch.from_numpy(rgb_resized).permute(2, 0, 1).float().to(self.device)
        img_t = (img_t - mean) / std
        img_t = img_t.unsqueeze(0)  # (1, 3, H, W)

        with torch.no_grad():
            pred_depth, _, _ = self.model.inference({"input": img_t})

        depth = pred_depth.squeeze().cpu().numpy()
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        # rescale back from canonical focal
        depth = depth / scale
        return depth.astype(np.float32)
