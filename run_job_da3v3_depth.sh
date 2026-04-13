#!/bin/bash
#SBATCH --account=3dv
#SBATCH --time=04:00:00
#SBATCH --output=/work/courses/3dv/team22/RGBTrack/logs/job_da3v3_depth_%j.out
#SBATCH --error=/work/courses/3dv/team22/RGBTrack/logs/job_da3v3_depth_%j.err

. /etc/profile.d/modules.sh
module load cuda/12.8

source /work/courses/3dv/team22/miniconda3/bin/activate /work/courses/3dv/team22/py310_env

export CUDA_HOME=/cluster/data/cuda/12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/work/courses/3dv/team22/py310_env/lib/python3.10/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/work/courses/3dv/team22/py310_env/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /work/courses/3dv/team22/RGBTrack
mkdir -p logs

# ── Step 1: install DA3 deps if missing ────────────────────────────────────
pip install -q einops huggingface_hub omegaconf safetensors plyfile e3nn \
               typer imageio moviepy easydict pycolmap 2>/dev/null || true

# ── Step 2: install DA3 package itself ─────────────────────────────────────
pip install -q -e DepthAnything3 2>/dev/null || true

# ── Step 3: generate depth PNGs ────────────────────────────────────────────
# Model options:
#   da3-small        → fast, low VRAM (safe on 5060 Ti)
#   da3metric-large  → metric depth, recommended
#   da3-giant        → best quality (needs 5090)
python dav3_offline_processor.py \
    --scene_dir /work/courses/3dv/team22/foundationpose/data/20250804_104715 \
    --out_dir   /work/courses/3dv/team22/foundationpose/data/20250804_104715/depth_da3v3 \
    --model     da3metric-large \
    --chunk_size 24 \
    --overlap    6 \
    --cam_K     /work/courses/3dv/team22/foundationpose/data/20250804_104715/cam_K.txt
