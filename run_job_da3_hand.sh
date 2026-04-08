#!/bin/bash
#SBATCH --account=3dv
#SBATCH --time=04:00:00
#SBATCH --output=/work/courses/3dv/team22/RGBTrack/logs/job_da3_hand_%j.out
#SBATCH --error=/work/courses/3dv/team22/RGBTrack/logs/job_da3_hand_%j.err

. /etc/profile.d/modules.sh
module load cuda/12.8

export PATH=/work/courses/3dv/team22/py310_env/bin:$PATH
export CUDA_HOME=/cluster/data/cuda/12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /work/courses/3dv/team22/RGBTrack
mkdir -p logs

# Step 1: convert NPZ to PNGs
/work/courses/3dv/team22/py310_env/bin/python convert_da3_npz.py

# Step 2: run tracking with DA3 depth
/work/courses/3dv/team22/py310_env/bin/python run_demo_vda_hand.py \
    --mesh_file /work/courses/3dv/team22/foundationpose/data/object/duck/duck.obj \
    --test_scene_dir /work/courses/3dv/team22/foundationpose/data/20250804_104715 \
    --debug_dir /work/courses/3dv/team22/foundationpose/debug/duck_da3_hand \
    --est_refine_iter 2 \
    --track_refine_iter 2 \
    --debug 2 \
    --depth_dir /work/courses/3dv/team22/foundationpose/data/20250804_104715/depth_da3
