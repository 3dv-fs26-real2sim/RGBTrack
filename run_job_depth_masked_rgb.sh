#!/bin/bash
#SBATCH --account=3dv
#SBATCH --time=00:30:00
#SBATCH --output=/work/courses/3dv/team22/RGBTrack/logs/job_depth_masked_rgb_%j.out
#SBATCH --error=/work/courses/3dv/team22/RGBTrack/logs/job_depth_masked_rgb_%j.err

. /etc/profile.d/modules.sh
source /work/courses/3dv/team22/miniconda3/bin/activate /work/courses/3dv/team22/miniconda3/envs/gsam_env

cd /work/courses/3dv/team22/RGBTrack
mkdir -p logs

python make_depth_masked_rgb.py \
    --scene_dir  /work/courses/3dv/team22/foundationpose/data/20250804_104715 \
    --out_video  /work/courses/3dv/team22/foundationpose/debug/rgb_depth_masked.mp4 \
    --fps        50
