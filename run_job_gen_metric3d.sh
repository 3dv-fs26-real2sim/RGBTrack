#!/bin/bash
#SBATCH --account=3dv
#SBATCH --time=02:00:00
#SBATCH --output=/work/courses/3dv/team22/RGBTrack/logs/job_gen_metric3d_%j.out
#SBATCH --error=/work/courses/3dv/team22/RGBTrack/logs/job_gen_metric3d_%j.err

. /etc/profile.d/modules.sh
module load cuda/12.8

export PATH=/work/courses/3dv/team22/py310_env/bin:$PATH
export CUDA_HOME=/cluster/data/cuda/12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /work/courses/3dv/team22/RGBTrack
mkdir -p logs

/work/courses/3dv/team22/py310_env/bin/python generate_metric3d_maps.py \
    --test_scene_dir /work/courses/3dv/team22/foundationpose/data/20250804_104715 \
    --metric3d_ckpt /work/courses/3dv/team22/metric3d_vit_large.pth
