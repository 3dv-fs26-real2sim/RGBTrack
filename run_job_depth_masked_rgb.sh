#!/bin/bash
#SBATCH --account=3dv
#SBATCH --time=00:30:00
#SBATCH --output=/work/courses/3dv/team22/RGBTrack/logs/job_depth_masked_rgb_%j.out
#SBATCH --error=/work/courses/3dv/team22/RGBTrack/logs/job_depth_masked_rgb_%j.err

. /etc/profile.d/modules.sh
module load cuda/12.8
export PATH=/work/courses/3dv/team22/py310_env/bin:$PATH
export CUDA_HOME=/cluster/data/cuda/12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/work/courses/3dv/team22/py310_env/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH

cd /work/courses/3dv/team22/RGBTrack
mkdir -p logs

/work/courses/3dv/team22/py310_env/bin/python make_depth_masked_rgb.py \
    --scene_dir  /work/courses/3dv/team22/foundationpose/data/20250804_104715 \
    --out_video  /work/courses/3dv/team22/foundationpose/debug/rgb_depth_masked.mp4 \
    --out_png_dir /work/courses/3dv/team22/foundationpose/data/20250804_104715/rgb_masked \
    --fps        50
