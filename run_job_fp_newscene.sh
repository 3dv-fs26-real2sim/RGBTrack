#!/bin/bash
#SBATCH --account=3dv
#SBATCH --time=12:00:00
#SBATCH --output=/work/courses/3dv/team22/RGBTrack/logs/job_fp_newscene_%j.out
#SBATCH --error=/work/courses/3dv/team22/RGBTrack/logs/job_fp_newscene_%j.err

# Tightly coupled FP for a new scene (has rgb/, masks/, depth_vda/)
# Usage: sbatch run_job_fp_newscene.sh <SCENE_NAME>
SCENE=${1:?need scene name}
SCENE_DIR=/work/courses/3dv/team22/foundationpose/data/$SCENE

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

/work/courses/3dv/team22/py310_env/bin/python run_demo_tightly_coupled.py \
    --mesh_file       /work/courses/3dv/team22/foundationpose/data/object/duck/duck.obj \
    --test_scene_dir  $SCENE_DIR \
    --rgb_dir         $SCENE_DIR/rgb \
    --depth_dir       $SCENE_DIR/depth_vda \
    --masks_dir       $SCENE_DIR/masks \
    --debug_dir       /work/scratch/hudela/duck_fp_$SCENE \
    --est_refine_iter 2 \
    --track_refine_iter 2 \
    --debug 2
