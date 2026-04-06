"""
Apple Depth Pro wrapper.

Install (run once on cluster):
    cd /work/courses/3dv/team22
    git clone https://github.com/apple/ml-depth-pro.git
    cd ml-depth-pro && pip install -e .
    source get_pretrained_models.sh   # downloads checkpoints/depth_pro.pt

Usage matches metric3d_wrapper: estimate(rgb, K) → (H,W) float32 metres.
"""

import numpy as np
import torch
from PIL import Image
import depth_pro

DEPTH_PRO_DIR = "/work/courses/3dv/team22/ml-depth-pro"


class DepthProWrapper:
    def __init__(self, checkpoint_path=f"{DEPTH_PRO_DIR}/checkpoints/depth_pro.pt", device="cuda"):
        self.device = device
        self.model, self.transform = depth_pro.create_model_and_transforms(
            checkpoint_uri=checkpoint_path,
            device=torch.device(device),
        )
        self.model.eval()

    def estimate(self, rgb, K):
        """
        rgb: (H, W, 3) uint8 RGB
        K:   (3, 3) intrinsic matrix

        Returns depth: (H, W) float32 metric depth in metres
        """
        f_px = float(K[0, 0])  # focal length in pixels

        pil_img = Image.fromarray(rgb)
        img_t   = self.transform(pil_img)

        with torch.no_grad():
            prediction = self.model.infer(img_t, f_px=f_px)

        depth = prediction["depth"].squeeze().cpu().numpy()
        return depth.astype(np.float32)
