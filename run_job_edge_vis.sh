#!/bin/bash
#SBATCH --account=3dv
#SBATCH --time=00:30:00
#SBATCH --output=/work/courses/3dv/team22/RGBTrack/logs/job_edge_vis_%j.out
#SBATCH --error=/work/courses/3dv/team22/RGBTrack/logs/job_edge_vis_%j.err

. /etc/profile.d/modules.sh
source /work/courses/3dv/team22/miniconda3/bin/activate /work/courses/3dv/team22/miniconda3/envs/gsam_env

cd /work/courses/3dv/team22/RGBTrack
mkdir -p logs

python make_edge_vis.py \
    --scene_dir  /work/courses/3dv/team22/foundationpose/data/20250804_104715 \
    --mask_dir   /work/courses/3dv/team22/foundationpose/data/20250804_104715/masks_cad_guided \
    --out_video  /work/courses/3dv/team22/foundationpose/debug/edge_vis_v1.mp4 \
    --amplify    100.0 \
    --fps        50
