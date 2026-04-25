#!/bin/bash
#SBATCH --account=3dv
#SBATCH --time=01:00:00
#SBATCH --output=/work/courses/3dv/team22/RGBTrack/logs/sam2video_%j.out
#SBATCH --error=/work/courses/3dv/team22/RGBTrack/logs/sam2video_%j.err

. /etc/profile.d/modules.sh
module load cuda/12.8
export CUDA_HOME=/cluster/data/cuda/12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=/work/courses/3dv/team22/py310_env/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /work/courses/3dv/team22/RGBTrack

/work/courses/3dv/team22/py310_env/bin/python sam2_video_wrapper.py \
    --scene_dir /work/courses/3dv/team22/foundationpose/data/20250804_104715 \
    --checkpoint /work/courses/3dv/team22/RGBTrack/segment-anything-2-real-time/sam2.1_hiera_small.pt \
    --jpg_dir /work/courses/3dv/team22/foundationpose/data/20250804_104715/rgb_painted_jpg \
    --mask_out_dir /work/courses/3dv/team22/foundationpose/data/20250804_104715/masks_painted
