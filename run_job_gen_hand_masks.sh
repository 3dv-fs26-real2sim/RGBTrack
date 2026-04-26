#!/bin/bash
#SBATCH --account=3dv
#SBATCH --time=02:00:00
#SBATCH --output=/work/courses/3dv/team22/RGBTrack/logs/job_hand_masks_%j.out
#SBATCH --error=/work/courses/3dv/team22/RGBTrack/logs/job_hand_masks_%j.err

. /etc/profile.d/modules.sh
module load cuda/12.8
source /work/courses/3dv/team22/miniconda3/bin/activate /work/courses/3dv/team22/miniconda3/envs/gsam_env
export CUDA_HOME=/cluster/data/cuda/12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /work/courses/3dv/team22/RGBTrack
mkdir -p logs

python gen_hand_masks_gdino.py \
    --scene_dir    /work/courses/3dv/team22/foundationpose/data/20250804_104715 \
    --hand_mask_dir /work/courses/3dv/team22/foundationpose/data/20250804_104715/masks_hand \
    --prompt       "white and black robotic hand" \
    --box_thresh   0.25 \
    --text_thresh  0.20 \
    --interval     1
