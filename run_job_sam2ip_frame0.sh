#!/bin/bash
#SBATCH --account=3dv
#SBATCH --time=00:15:00
#SBATCH --output=/work/courses/3dv/team22/RGBTrack/logs/job_sam2ip_frame0_%j.out
#SBATCH --error=/work/courses/3dv/team22/RGBTrack/logs/job_sam2ip_frame0_%j.err

. /etc/profile.d/modules.sh
module load cuda/12.8
export PATH=/work/courses/3dv/team22/py310_env/bin:$PATH
export CUDA_HOME=/cluster/data/cuda/12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/work/courses/3dv/team22/py310_env/lib/python3.10/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/work/courses/3dv/team22/py310_env/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH

cd /work/courses/3dv/team22/RGBTrack

/work/courses/3dv/team22/py310_env/bin/python - <<'EOF'
import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

SCENE = "/work/courses/3dv/team22/foundationpose/data/20250804_104715"
CKPT  = "/work/courses/3dv/team22/RGBTrack/segment-anything-2-real-time/sam2.1_hiera_small.pt"
CFG   = "configs/sam2.1/sam2.1_hiera_s.yaml"

model = build_sam2(CFG, CKPT, device="cuda")
pred  = SAM2ImagePredictor(model)

# Use painted frame 0
color = cv2.cvtColor(cv2.imread(f"{SCENE}/rgb_painted/000000.png"), cv2.COLOR_BGR2RGB)

# Use centroid of existing mask as positive point prompt
mask0 = cv2.imread(f"{SCENE}/masks/000000.png", cv2.IMREAD_GRAYSCALE)
mask0 = (mask0 > 127)
ys, xs = np.where(mask0)
cx, cy = int(xs.mean()), int(ys.mean())
point_coords = np.array([[cx, cy]])
point_labels = np.array([1])  # positive

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    pred.set_image(color)
    masks, _, _ = pred.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=False,
    )

out = (masks[0] > 0).astype(np.uint8) * 255
cv2.imwrite(f"{SCENE}/masks_painted/000000.png", out)
print("saved mask, area:", out.sum() // 255)
EOF
