#!/bin/bash
#SBATCH --account=3dv
#SBATCH --time=01:00:00
#SBATCH --output=/work/courses/3dv/team22/RGBTrack/logs/job_sam2_nobg_%j.out
#SBATCH --error=/work/courses/3dv/team22/RGBTrack/logs/job_sam2_nobg_%j.err

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

# Convert rgb_masked PNGs to JPEGs for SAM2VP
echo "Converting rgb_masked to JPEGs..."
mkdir -p $SCENE/rgb_masked_jpg
/work/courses/3dv/team22/py310_env/bin/python -c "
import cv2, glob, os
paths = sorted(glob.glob('$SCENE/rgb_masked/*.png'))
for p in paths:
    img  = cv2.imread(p)
    name = os.path.splitext(os.path.basename(p))[0] + '.jpg'
    cv2.imwrite(os.path.join('$SCENE/rgb_masked_jpg', name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
print('converted', len(paths))
"

# SAM2VP: seed at frame 100 with point at arm location (90% right, 65% up)
/work/courses/3dv/team22/py310_env/bin/python - <<'PYEOF'
import glob, os, torch, cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from sam2.build_sam import build_sam2_video_predictor

SCENE     = "/work/courses/3dv/team22/foundationpose/data/20250804_104715"
JPG_DIR   = f"{SCENE}/rgb_masked_jpg"
OUT_DIR   = f"{SCENE}/masks_hand_sam2"
SAM2_CKPT = "/work/courses/3dv/team22/RGBTrack/segment-anything-2-real-time/sam2.1_hiera_small.pt"
SAM2_CFG  = "configs/sam2.1/sam2.1_hiera_s.yaml"

os.makedirs(OUT_DIR, exist_ok=True)
frame_files = sorted(glob.glob(f"{SCENE}/rgb/*.png"))
N = len(frame_files)
h, w = cv2.imread(frame_files[0]).shape[:2]

px = int(0.90 * w)
py = int((1 - 0.65) * h)
print(f"Seeding at frame 100, point=({px},{py})  image={w}x{h}")

predictor = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device="cuda")
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    state = predictor.init_state(video_path=JPG_DIR,
                                  offload_video_to_cpu=True,
                                  offload_state_to_cpu=True)
    predictor.add_new_points_or_box(
        state, frame_idx=100, obj_id=1,
        points=np.array([[px, py]], dtype=np.float32),
        labels=np.array([1], dtype=np.int32),
    )
    for fi, _, mlogits in predictor.propagate_in_video(state):
        m = (mlogits[0] > 0.0).cpu().numpy()
        if m.ndim == 3: m = m[0]
        m = binary_fill_holes(m).astype(np.uint8) * 255
        name = os.path.splitext(os.path.basename(frame_files[fi]))[0]
        cv2.imwrite(f"{OUT_DIR}/{name}.png", m)
        if fi % 100 == 0: print(f"frame {fi}/{N}")

print(f"Done -> {OUT_DIR}")
PYEOF
