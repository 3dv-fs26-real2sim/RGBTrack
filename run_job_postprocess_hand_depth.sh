#!/bin/bash
#SBATCH --account=3dv
#SBATCH --time=00:10:00
#SBATCH --output=/work/courses/3dv/team22/RGBTrack/logs/job_pp_hand_depth_%j.out
#SBATCH --error=/work/courses/3dv/team22/RGBTrack/logs/job_pp_hand_depth_%j.err

. /etc/profile.d/modules.sh
source /work/courses/3dv/team22/miniconda3/bin/activate /work/courses/3dv/team22/miniconda3/envs/gsam_env

cd /work/courses/3dv/team22/RGBTrack
mkdir -p logs

python postprocess_hand_masks_depth.py \
    --hand_mask_dir /work/courses/3dv/team22/foundationpose/data/20250804_104715/masks_hand_nobg \
    --duck_mask_dir /work/courses/3dv/team22/foundationpose/data/20250804_104715/masks_cad_guided \
    --depth_dir     /work/courses/3dv/team22/foundationpose/data/20250804_104715/depth \
    --out_dir       /work/courses/3dv/team22/foundationpose/data/20250804_104715/masks_hand_final
