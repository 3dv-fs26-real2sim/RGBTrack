#!/bin/bash
#SBATCH --account=3dv
#SBATCH --time=02:00:00
#SBATCH --output=/work/courses/3dv/team22/RGBTrack/logs/job_sam2vp_masks_%j.out
#SBATCH --error=/work/courses/3dv/team22/RGBTrack/logs/job_sam2vp_masks_%j.err

# Propagate seed mask through full video using SAM2VP
# Usage: sbatch run_job_sam2vp_masks.sh <SCENE_NAME>
SCENE=${1:?need scene name}

. /etc/profile.d/modules.sh
module load cuda/12.8

export PATH=/work/courses/3dv/team22/py310_env/bin:$PATH
export CUDA_HOME=/cluster/data/cuda/12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/work/courses/3dv/team22/py310_env/lib/python3.10/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/work/courses/3dv/team22/py310_env/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCENE_DIR=/work/courses/3dv/team22/foundationpose/data/$SCENE

cd /work/courses/3dv/team22/RGBTrack
mkdir -p logs

# Convert PNGs to JPGs for SAM2VP
JPG_DIR=$SCENE_DIR/rgb_jpg
mkdir -p $JPG_DIR
/work/courses/3dv/team22/py310_env/bin/python -c "
import cv2, glob, os
paths = sorted(glob.glob('$SCENE_DIR/rgb/*.png'))
for p in paths:
    out = '$JPG_DIR/' + os.path.splitext(os.path.basename(p))[0] + '.jpg'
    if not os.path.exists(out):
        cv2.imwrite(out, cv2.imread(p), [cv2.IMWRITE_JPEG_QUALITY, 95])
print('jpgs ready:', len(paths))
"

/work/courses/3dv/team22/py310_env/bin/python - <<PYEOF
import glob, os, torch, cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from sam2.build_sam import build_sam2_video_predictor

SCENE_DIR = "$SCENE_DIR"
JPG_DIR   = "$JPG_DIR"
OUT_DIR   = f"{SCENE_DIR}/masks"
SAM2_CKPT = "/work/courses/3dv/team22/RGBTrack/segment-anything-2-real-time/sam2.1_hiera_small.pt"
SAM2_CFG  = "configs/sam2.1/sam2.1_hiera_s.yaml"

frame_files = sorted(glob.glob(f"{SCENE_DIR}/rgb/*.png"))
N = len(frame_files)

seed = cv2.imread(f"{OUT_DIR}/000000.png", cv2.IMREAD_GRAYSCALE)
assert seed is not None, "No seed mask at masks/000000.png"
duck_mask0 = (seed > 127).astype(bool)
print(f"Seed mask: {int(duck_mask0.sum())} px, propagating {N} frames")

predictor = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device="cuda")
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    state = predictor.init_state(video_path=JPG_DIR,
                                  offload_video_to_cpu=True,
                                  offload_state_to_cpu=True)
    predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=duck_mask0)
    for fi, obj_ids, mlogits in predictor.propagate_in_video(state):
        m = (mlogits[0] > 0.0).cpu().numpy()
        if m.ndim == 3: m = m[0]
        m = binary_fill_holes(m).astype(np.uint8) * 255
        name = os.path.splitext(os.path.basename(frame_files[fi]))[0]
        cv2.imwrite(f"{OUT_DIR}/{name}.png", m)
        if fi % 200 == 0: print(f"  frame {fi}/{N}")

print(f"Done -> {OUT_DIR}")
PYEOF
