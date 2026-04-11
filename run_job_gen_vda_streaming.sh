#!/bin/bash
#SBATCH --account=3dv
#SBATCH --time=04:00:00
#SBATCH --output=/work/courses/3dv/team22/RGBTrack/logs/job_gen_vda_streaming_%j.out
#SBATCH --error=/work/courses/3dv/team22/RGBTrack/logs/job_gen_vda_streaming_%j.err

. /etc/profile.d/modules.sh
module load cuda/12.8

# Use the vda conda env (has Video-Depth-Anything deps)
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /work/scratch/$USER/conda/envs/vda

export CUDA_HOME=/cluster/data/cuda/12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /work/courses/3dv/team22/RGBTrack
mkdir -p logs

python generate_vda_streaming_maps.py \
    --scene_dir /work/courses/3dv/team22/foundationpose/data/20250804_104715 \
    --vda_repo  /home/jaerkim/Video-Depth-Anything \
    --out_dir   /work/courses/3dv/team22/foundationpose/data/20250804_104715/depth_vda_streaming
