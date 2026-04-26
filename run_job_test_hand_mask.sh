#!/bin/bash
#SBATCH --account=3dv
#SBATCH --time=00:10:00
#SBATCH --output=/work/courses/3dv/team22/RGBTrack/logs/job_test_hand_%j.out
#SBATCH --error=/work/courses/3dv/team22/RGBTrack/logs/job_test_hand_%j.err

. /etc/profile.d/modules.sh
module load cuda/12.8
source /work/courses/3dv/team22/miniconda3/bin/activate /work/courses/3dv/team22/miniconda3/envs/gsam_env
export CUDA_HOME=/cluster/data/cuda/12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

cd /work/courses/3dv/team22/RGBTrack

python - <<'EOF'
import sys, os, cv2, numpy as np, torch
sys.path.insert(0, '/work/courses/3dv/team22/GroundingDINO')
from groundingdino.util.inference import load_model, predict
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from groundingdino.util.inference import load_image
from PIL import Image

GDINO_CFG  = '/work/courses/3dv/team22/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
GDINO_CKPT = '/work/courses/3dv/team22/foundationpose/weights/groundingdino_swint_ogc.pth'
SAM2_CKPT  = '/work/courses/3dv/team22/RGBTrack/segment-anything-2-real-time/sam2.1_hiera_small.pt'
SAM2_CFG   = 'configs/sam2.1/sam2.1_hiera_s.yaml'
FRAME      = '/work/courses/3dv/team22/foundationpose/data/20250804_104715/rgb/000000.png'
OUT        = '/work/courses/3dv/team22/foundationpose/data/20250804_104715/test_hand_mask.png'

img_np, img_t = load_image(FRAME)
h, w = img_np.shape[:2]

gdino = load_model(GDINO_CFG, GDINO_CKPT)
boxes, logits, phrases = predict(model=gdino, image=img_t, caption='robotic hand',
                                  box_threshold=0.30, text_threshold=0.25)
print(f'Detections: {len(boxes)}')
for i in range(len(boxes)):
    cx,cy,bw,bh = boxes[i].tolist()
    print(f'  [{i}] {phrases[i]} conf={logits[i]:.2f} center=({int(cx*w)},{int(cy*h)}) size=({int(bw*w)}x{int(bh*h)})')

if len(boxes) == 0:
    print('No detection — try lowering box_thresh')
else:
    best = logits.argmax()
    cx,cy,bw,bh = boxes[best].tolist()
    x1,y1,x2,y2 = int((cx-bw/2)*w), int((cy-bh/2)*h), int((cx+bw/2)*w), int((cy+bh/2)*h)
    sam2 = build_sam2(SAM2_CFG, SAM2_CKPT, device='cuda')
    predictor = SAM2ImagePredictor(sam2)
    predictor.set_image(img_np)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        masks, _, _ = predictor.predict(box=np.array([[x1,y1,x2,y2]]), multimask_output=False)
    mask = (masks[0] > 0).astype(np.uint8) * 255
    cv2.imwrite(OUT, mask)
    print(f'Saved mask to {OUT}')
EOF
