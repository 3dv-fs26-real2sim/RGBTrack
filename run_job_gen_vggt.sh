#!/bin/bash
#SBATCH --account=3dv
#SBATCH --time=02:00:00
#SBATCH --output=/work/courses/3dv/team22/RGBTrack/logs/job_gen_vggt_%j.out
#SBATCH --error=/work/courses/3dv/team22/RGBTrack/logs/job_gen_vggt_%j.err

. /etc/profile.d/modules.sh
module load cuda/12.8

source /work/courses/3dv/team22/miniconda3/bin/activate vggt_env

export CUDA_HOME=/cluster/data/cuda/12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /work/courses/3dv/team22/RGBTrack
mkdir -p logs

python generate_vggt_maps.py \
    --test_scene_dir /work/courses/3dv/team22/foundationpose/data/20250804_104715 \
    --model_path /work/courses/3dv/team22/vggt_model.pt
