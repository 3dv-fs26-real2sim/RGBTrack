#!/bin/bash
#SBATCH --account=3dv
#SBATCH --time=02:00:00
#SBATCH --output=/work/courses/3dv/team22/RGBTrack/logs/job_depth_hist_vda_%j.out
#SBATCH --error=/work/courses/3dv/team22/RGBTrack/logs/job_depth_hist_vda_%j.err

export PATH=/work/courses/3dv/team22/py310_env/bin:$PATH

cd /work/courses/3dv/team22/RGBTrack
mkdir -p logs

/work/courses/3dv/team22/py310_env/bin/python plot_depth_histogram_video.py \
    --source vda \
    --scene_dir /work/courses/3dv/team22/foundationpose/data/20250804_104715 \
    --out /work/courses/3dv/team22/foundationpose/debug/depth_hist_vda.mp4
