#!/bin/bash
#SBATCH --account=3dv
#SBATCH --time=06:00:00
#SBATCH --output=/work/courses/3dv/team22/RGBTrack/logs/job_vda_%j.out
#SBATCH --error=/work/courses/3dv/team22/RGBTrack/logs/job_vda_%j.err

# ── Init module system ─────────────────────────────────────────────────────────
. /etc/profile.d/modules.sh
module load cuda/12.8

# ── Activate env ───────────────────────────────────────────────────────────────
export PATH=/work/courses/3dv/team22/py310_env/bin:$PATH

# ── CUDA paths ─────────────────────────────────────────────────────────────────
export CUDA_HOME=/cluster/data/cuda/12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/work/courses/3dv/team22/py310_env/lib/python3.10/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/work/courses/3dv/team22/py310_env/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH

# ── Run ────────────────────────────────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /work/courses/3dv/team22/RGBTrack
mkdir -p logs

/work/courses/3dv/team22/py310_env/bin/python run_demo_vda_depth.py \
    --mesh_file /work/courses/3dv/team22/foundationpose/data/object/duck/duck.obj \
    --test_scene_dir /work/courses/3dv/team22/foundationpose/data/20250804_104715 \
    --debug_dir /work/courses/3dv/team22/foundationpose/debug/duck_vda \
    --est_refine_iter 2 \
    --track_refine_iter 2 \
    --debug 2
