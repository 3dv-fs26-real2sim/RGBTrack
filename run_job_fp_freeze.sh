#!/bin/bash
#SBATCH --account=3dv
#SBATCH --time=08:00:00
#SBATCH --output=/work/courses/3dv/team22/RGBTrack/logs/job_fp_freeze_%j.out
#SBATCH --error=/work/courses/3dv/team22/RGBTrack/logs/job_fp_freeze_%j.err

# FP with anomaly detection + freeze rotation
# Usage: sbatch run_job_fp_freeze.sh <SCENE_NAME>
SCENE=${1:?need scene name}
SCENE_DIR=/work/scratch/hudela/$SCENE

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

/work/courses/3dv/team22/py310_env/bin/python run_demo_fp_freeze.py \
    --mesh_file        /work/courses/3dv/team22/foundationpose/data/object/duck/duck.obj \
    --test_scene_dir   $SCENE_DIR \
    --masks_dir        $SCENE_DIR/masks \
    --depth_dir        $SCENE_DIR/depth_vda \
    --debug_dir        $SCENE_DIR/fp \
    --est_refine_iter  2 \
    --track_refine_iter 2 \
    --debug 2
