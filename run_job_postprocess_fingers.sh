#!/bin/bash
#SBATCH --account=3dv
#SBATCH --time=00:30:00
#SBATCH --output=/work/courses/3dv/team22/RGBTrack/logs/job_pp_fingers_%j.out
#SBATCH --error=/work/courses/3dv/team22/RGBTrack/logs/job_pp_fingers_%j.err

. /etc/profile.d/modules.sh
source /work/courses/3dv/team22/miniconda3/bin/activate /work/courses/3dv/team22/miniconda3/envs/gsam_env

cd /work/courses/3dv/team22/RGBTrack
mkdir -p logs

python postprocess_finger_masks.py \
    --in_dir        /work/courses/3dv/team22/foundationpose/data/20250804_104715/masks_hand_fingers \
    --rgb_dir       /work/courses/3dv/team22/foundationpose/data/20250804_104715/rgb \
    --out_dir       /work/courses/3dv/team22/foundationpose/data/20250804_104715/masks_hand_fingers_clean \
    --dilation      4 \
    --min_blob_area 100
