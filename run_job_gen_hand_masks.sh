#!/bin/bash
#SBATCH --account=3dv
#SBATCH --time=02:00:00
#SBATCH --output=/work/courses/3dv/team22/RGBTrack/logs/job_hand_masks_%j.out
#SBATCH --error=/work/courses/3dv/team22/RGBTrack/logs/job_hand_masks_%j.err

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

/work/courses/3dv/team22/py310_env/bin/python gen_hand_masks_gdino.py \
    --scene_dir    /work/courses/3dv/team22/foundationpose/data/20250804_104715 \
    --hand_mask_dir /work/courses/3dv/team22/foundationpose/data/20250804_104715/masks_hand \
    --prompt       "robotic hand" \
    --box_thresh   0.30 \
    --text_thresh  0.25 \
    --interval     1
