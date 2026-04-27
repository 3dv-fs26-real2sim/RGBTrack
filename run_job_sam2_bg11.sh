#!/bin/bash
#SBATCH --account=3dv
#SBATCH --time=01:00:00
#SBATCH --output=/work/courses/3dv/team22/RGBTrack/logs/job_sam2_bg11_%j.out
#SBATCH --error=/work/courses/3dv/team22/RGBTrack/logs/job_sam2_bg11_%j.err

. /etc/profile.d/modules.sh
module load cuda/12.8
export PATH=/work/courses/3dv/team22/py310_env/bin:$PATH
export CUDA_HOME=/cluster/data/cuda/12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/work/courses/3dv/team22/py310_env/lib/python3.10/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/work/courses/3dv/team22/py310_env/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCENE=/work/courses/3dv/team22/foundationpose/data/20250804_104715

cd /work/courses/3dv/team22/RGBTrack
mkdir -p logs

# Convert rgb_bg11 PNGs to JPG for SAM2
JPG_DIR=$SCENE/rgb_bg11_jpg
mkdir -p $JPG_DIR
for f in $SCENE/rgb_bg11/*.png; do
    base=$(basename $f .png)
    cv2_out=$JPG_DIR/$base.jpg
    if [ ! -f $cv2_out ]; then
        /work/courses/3dv/team22/py310_env/bin/python -c "
import cv2; img=cv2.imread('$f'); cv2.imwrite('$cv2_out', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
"
    fi
done
echo "JPGs ready"

/work/courses/3dv/team22/py310_env/bin/python - <<'PYEOF'
import glob, os, torch, cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from sam2.build_sam import build_sam2_video_predictor

SCENE     = "/work/courses/3dv/team22/foundationpose/data/20250804_104715"
JPG_DIR   = f"{SCENE}/rgb_bg11_jpg"
OUT_DIR   = f"{SCENE}/masks_bg11"
SAM2_CKPT = "/work/courses/3dv/team22/RGBTrack/segment-anything-2-real-time/sam2.1_hiera_small.pt"
SAM2_CFG  = "configs/sam2.1/sam2.1_hiera_s.yaml"

os.makedirs(OUT_DIR, exist_ok=True)
frame_files = sorted(glob.glob(f"{SCENE}/rgb_bg11/*.png"))
N = len(frame_files)

# Frame 0 seeds from original masks
duck_mask0 = cv2.imread(f"{SCENE}/masks/000000.png", cv2.IMREAD_GRAYSCALE)
duck_mask0 = (duck_mask0 > 127).astype(bool)
hand_mask0 = cv2.imread(f"{SCENE}/masks_hand_final/000000.png", cv2.IMREAD_GRAYSCALE)
hand_mask0 = (hand_mask0 > 127).astype(bool) if hand_mask0 is not None else None

predictor = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device="cuda")
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    state = predictor.init_state(video_path=JPG_DIR,
                                  offload_video_to_cpu=True,
                                  offload_state_to_cpu=True)
    predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=duck_mask0)
    if hand_mask0 is not None and hand_mask0.any():
        predictor.add_new_mask(state, frame_idx=0, obj_id=2, mask=hand_mask0)
        print("Hand boundary added as obj_id=2")

    for fi, obj_ids, mlogits in predictor.propagate_in_video(state):
        duck_idx = list(obj_ids).index(1) if 1 in obj_ids else 0
        m = (mlogits[duck_idx] > 0.0).cpu().numpy()
        if m.ndim == 3: m = m[0]
        m = binary_fill_holes(m).astype(np.uint8) * 255
        name = os.path.splitext(os.path.basename(frame_files[fi]))[0]
        cv2.imwrite(f"{OUT_DIR}/{name}.png", m)
        if fi % 100 == 0: print(f"frame {fi}/{N}")

print(f"Done -> {OUT_DIR}")
PYEOF
