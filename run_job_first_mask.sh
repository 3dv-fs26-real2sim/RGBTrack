#!/bin/bash
#SBATCH --account=3dv
#SBATCH --time=00:15:00
#SBATCH --output=/work/courses/3dv/team22/RGBTrack/logs/job_first_mask_%j.out
#SBATCH --error=/work/courses/3dv/team22/RGBTrack/logs/job_first_mask_%j.err

# Replicates scripts/generate_mask.py — the very first mask we made
# for 20250804_104715. Original SAM (vit_h), single click, multimask, best score.
#
# Usage: sbatch run_job_first_mask.sh <SCENE> <CLICK_X> <CLICK_Y>
SCENE=${1:?need scene}
CX=${2:?click x}
CY=${3:?click y}

. /etc/profile.d/modules.sh
module load cuda/12.8

export PATH=/work/courses/3dv/team22/py310_env/bin:$PATH
export CUDA_HOME=/cluster/data/cuda/12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/work/courses/3dv/team22/py310_env/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH

cd /work/courses/3dv/team22/RGBTrack

SCENE=$SCENE CX=$CX CY=$CY \
/work/courses/3dv/team22/py310_env/bin/python - <<'PYEOF'
import os, cv2, numpy as np
from segment_anything import sam_model_registry, SamPredictor

SCENE_DIR  = f"/work/courses/3dv/team22/foundationpose/data/{os.environ['SCENE']}"
DEBUG      = "/work/courses/3dv/team22/foundationpose/debug"
CHECKPOINT = "/work/courses/3dv/team22/foundationpose/weights/sam_vit_h_4b8939.pth.1"
MODEL_TYPE = "vit_h"
CX, CY     = int(os.environ['CX']), int(os.environ['CY'])

frame0 = cv2.imread(f"{SCENE_DIR}/rgb/000000.png")
rgb    = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT).cuda()
predictor = SamPredictor(sam)
predictor.set_image(rgb)

masks, scores, _ = predictor.predict(
    point_coords=np.array([[CX, CY]]),
    point_labels=np.array([1]),
    multimask_output=True,
)
best = masks[int(np.argmax(scores))]

os.makedirs(f"{SCENE_DIR}/masks", exist_ok=True)
out_path = f"{SCENE_DIR}/masks/000000.png"
cv2.imwrite(out_path, (best * 255).astype(np.uint8))
print(f"saved {out_path}  px={int(best.sum())}  score={scores.max():.3f}")

# Preview overlay
prev = frame0.copy()
prev[best] = (0.4*prev[best] + 0.6*np.array([0,0,255])).astype(np.uint8)
cv2.circle(prev, (CX, CY), 4, (0,255,0), -1)
os.makedirs(DEBUG, exist_ok=True)
prev_path = f"{DEBUG}/first_mask_{os.environ['SCENE']}.png"
cv2.imwrite(prev_path, prev)
print(f"preview -> {prev_path}")
PYEOF
