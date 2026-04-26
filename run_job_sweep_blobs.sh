#!/bin/bash
#SBATCH --account=3dv
#SBATCH --time=00:10:00
#SBATCH --output=/work/courses/3dv/team22/RGBTrack/logs/job_sweep_blobs_%j.out
#SBATCH --error=/work/courses/3dv/team22/RGBTrack/logs/job_sweep_blobs_%j.err

. /etc/profile.d/modules.sh
source /work/courses/3dv/team22/miniconda3/bin/activate /work/courses/3dv/team22/miniconda3/envs/gsam_env

cd /work/courses/3dv/team22/RGBTrack
mkdir -p logs

python sweep_remove_blobs.py \
    --in_dir        /work/courses/3dv/team22/foundationpose/data/20250804_104715/masks_hand_refined \
    --out_dir       /work/courses/3dv/team22/foundationpose/data/20250804_104715/masks_hand_swept \
    --min_blob_area 300
