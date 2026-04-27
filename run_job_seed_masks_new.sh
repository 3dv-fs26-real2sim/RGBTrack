#!/bin/bash
#SBATCH --account=3dv
#SBATCH --time=00:30:00
#SBATCH --output=/work/courses/3dv/team22/RGBTrack/logs/job_seed_masks_new_%j.out
#SBATCH --error=/work/courses/3dv/team22/RGBTrack/logs/job_seed_masks_new_%j.err

. /etc/profile.d/modules.sh
module load cuda/12.8

export PATH=/work/courses/3dv/team22/py310_env/bin:$PATH
export CUDA_HOME=/cluster/data/cuda/12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/work/courses/3dv/team22/py310_env/lib/python3.10/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/work/courses/3dv/team22/py310_env/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /work/courses/3dv/team22/RGBTrack

/work/courses/3dv/team22/py310_env/bin/python - <<'PYEOF'
import os, cv2, torch
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

DATA = "/work/courses/3dv/team22/foundationpose/data"
SAM2_CKPT = "/work/courses/3dv/team22/RGBTrack/segment-anything-2-real-time/sam2.1_hiera_small.pt"
SAM2_CFG  = "configs/sam2.1/sam2.1_hiera_s.yaml"

# (scene, absolute_xy_from_top_left)  — None means use centre + offset path below
SCENES = [
    ("20250804_113654", (345, 278)),  # 345x from L, 202 from bottom (480-202=278)
    ("20250804_124203", None),         # already correct from previous run, skip
    ("20250806_102854", (385, 308)),  # 385x, 172 from bottom (480-172=308)
]

sam2 = build_sam2(SAM2_CFG, SAM2_CKPT, device="cuda")
predictor = SAM2ImagePredictor(sam2)

for scene, click in SCENES:
    if click is None:
        print(f"[{scene}] skipping (already correct)")
        continue
    img_path = f"{DATA}/{scene}/rgb/000000.png"
    out_dir  = f"{DATA}/{scene}/masks"
    out_path = f"{out_dir}/000000.png"
    os.makedirs(out_dir, exist_ok=True)

    img = cv2.imread(img_path)
    if img is None:
        print(f"[SKIP] {scene}: no frame 0 at {img_path}")
        continue
    h, w = img.shape[:2]
    cx, cy = click
    print(f"[{scene}] click=({cx},{cy}) on {w}x{h}")

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb)
    masks, scores, _ = predictor.predict(
        point_coords=np.array([[cx, cy]]),
        point_labels=np.array([1]),
        multimask_output=True,
    )
    best = masks[int(np.argmax(scores))]
    cv2.imwrite(out_path, (best.astype(np.uint8) * 255))
    print(f"  saved {out_path} ({int(best.sum())} px)")
PYEOF
