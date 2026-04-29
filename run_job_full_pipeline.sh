#!/bin/bash
#SBATCH --account=3dv
#SBATCH --time=12:00:00
#SBATCH --output=/work/courses/3dv/team22/RGBTrack/logs/job_pipeline_%j.out
#SBATCH --error=/work/courses/3dv/team22/RGBTrack/logs/job_pipeline_%j.err

# Usage: sbatch run_job_full_pipeline.sh <SCENE_NAME>
# Example: sbatch run_job_full_pipeline.sh 20250804_104715
SCENE_NAME=${1:-20250804_104715}
SCENE_DIR=/work/courses/3dv/team22/foundationpose/data/$SCENE_NAME
MESH=/work/courses/3dv/team22/foundationpose/data/object/duck/duck.obj
DEBUG=/work/scratch/hudela

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
mkdir -p logs

PY=/work/courses/3dv/team22/py310_env/bin/python

echo "=================================================="
echo " PIPELINE START — scene=$SCENE_NAME"
echo "=================================================="

# ── 1. SAM2VP masks ─────────────────────────────────────────────────────────
echo ""
echo "[1/3] SAM2VP masks → $SCENE_DIR/masks"
echo "--------------------------------------------------"

JPG_DIR=$SCENE_DIR/rgb_jpg
mkdir -p $JPG_DIR
$PY -c "
import cv2, glob, os
paths = sorted(glob.glob('$SCENE_DIR/rgb/*.png'))
for p in paths:
    out = '$JPG_DIR/' + os.path.splitext(os.path.basename(p))[0] + '.jpg'
    if not os.path.exists(out):
        cv2.imwrite(out, cv2.imread(p), [cv2.IMWRITE_JPEG_QUALITY, 95])
print('jpg ready:', len(paths))
"

$PY - <<PYEOF
import glob, os, torch, cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from sam2.build_sam import build_sam2_video_predictor

SCENE     = "$SCENE_DIR"
JPG_DIR   = f"{SCENE}/rgb_jpg"
OUT_DIR   = f"{SCENE}/masks"
SAM2_CKPT = "/work/courses/3dv/team22/RGBTrack/segment-anything-2-real-time/sam2.1_hiera_small.pt"
SAM2_CFG  = "configs/sam2.1/sam2.1_hiera_s.yaml"

os.makedirs(OUT_DIR, exist_ok=True)
frame_files = sorted(glob.glob(f"{SCENE}/rgb/*.png"))
N = len(frame_files)

# Frame 0 seed must already exist at masks/000000.png (manual or CAD-anchored)
seed_path = f"{OUT_DIR}/000000.png"
assert os.path.exists(seed_path), f"Need seed mask at {seed_path}"
duck_mask0 = cv2.imread(seed_path, cv2.IMREAD_GRAYSCALE)
duck_mask0 = (duck_mask0 > 127).astype(bool)

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
        if fi % 100 == 0: print(f"  mask frame {fi}/{N}")
print(f"masks done -> {OUT_DIR}")
PYEOF

# ── 2. VDA streaming depth ──────────────────────────────────────────────────
echo ""
echo "[2/3] VDA streaming depth → $SCENE_DIR/depth_vda"
echo "--------------------------------------------------"

$PY generate_vda_streaming_maps.py \
    --scene_dir $SCENE_DIR \
    --vda_repo  /work/courses/3dv/team22/Video-Depth-Anything-Internal \
    --rgb_dir   $SCENE_DIR/rgb \
    --out_dir   $SCENE_DIR/depth_vda

# ── 3. FoundationPose tightly coupled ───────────────────────────────────────
echo ""
echo "[3/3] FoundationPose tracking → $DEBUG/duck_fp_$SCENE_NAME"
echo "--------------------------------------------------"

$PY run_demo_tightly_coupled.py \
    --mesh_file       $MESH \
    --test_scene_dir  $SCENE_DIR \
    --rgb_dir         $SCENE_DIR/rgb \
    --depth_dir       $SCENE_DIR/depth_vda \
    --masks_dir       $SCENE_DIR/masks \
    --debug_dir       $DEBUG/duck_fp_$SCENE_NAME \
    --est_refine_iter 2 \
    --track_refine_iter 2 \
    --debug 2

echo ""
echo "=================================================="
echo " PIPELINE DONE — scene=$SCENE_NAME"
echo "=================================================="
