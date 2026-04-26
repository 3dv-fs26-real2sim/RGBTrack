#!/bin/bash
#SBATCH --account=3dv
#SBATCH --time=00:15:00
#SBATCH --output=/work/courses/3dv/team22/RGBTrack/logs/job_yellow_boost_%j.out
#SBATCH --error=/work/courses/3dv/team22/RGBTrack/logs/job_yellow_boost_%j.err

. /etc/profile.d/modules.sh
source /work/courses/3dv/team22/miniconda3/bin/activate /work/courses/3dv/team22/miniconda3/envs/gsam_env

cd /work/courses/3dv/team22/RGBTrack
mkdir -p logs

python preprocess_yellow_boost.py \
    --in_dir    /work/courses/3dv/team22/foundationpose/data/20250804_104715/rgb \
    --out_dir   /work/courses/3dv/team22/foundationpose/data/20250804_104715/rgb_yellow \
    --hue_lo    15 \
    --hue_hi    35 \
    --sat_boost 2.5 \
    --val_boost 1.2
