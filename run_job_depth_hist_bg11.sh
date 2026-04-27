#!/bin/bash
#SBATCH --account=3dv
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=/work/courses/3dv/team22/RGBTrack/logs/job_depth_hist_bg11_%j.out
#SBATCH --error=/work/courses/3dv/team22/RGBTrack/logs/job_depth_hist_bg11_%j.err

export PATH=/work/courses/3dv/team22/py310_env/bin:$PATH

cd /work/courses/3dv/team22/RGBTrack
mkdir -p logs

# Full scene comparison
/work/courses/3dv/team22/py310_env/bin/python plot_depth_histogram_video.py \
    --source    vda_nobg,vda_bg11 \
    --layout    overlay \
    --scene_dir /work/courses/3dv/team22/foundationpose/data/20250804_104715 \
    --out       /work/courses/3dv/team22/foundationpose/debug/depth_hist_bg11_full.mp4

# Duck-only comparison
/work/courses/3dv/team22/py310_env/bin/python plot_depth_histogram_video.py \
    --source    vda_nobg,vda_bg11 \
    --layout    overlay \
    --duck_only \
    --scene_dir /work/courses/3dv/team22/foundationpose/data/20250804_104715 \
    --out       /work/courses/3dv/team22/foundationpose/debug/depth_hist_bg11_duck.mp4
