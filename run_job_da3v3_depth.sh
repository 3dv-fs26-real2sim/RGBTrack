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

python dav3_offline_processor.py \
    --scene_dir /work/courses/3dv/team22/foundationpose/data/20250804_104715 \
    --out_dir   /work/courses/3dv/team22/foundationpose/data/20250804_104715/depth_da3v3 \
    --model     da3metric-large \
    --chunk_size 24 \
    --overlap    6 \
    --cam_K     /work/courses/3dv/team22/foundationpose/data/20250804_104715/cam_K.txt
