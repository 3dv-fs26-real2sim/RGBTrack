#!/bin/bash
#SBATCH --account=3dv
#SBATCH --time=02:00:00
#SBATCH --output=/work/courses/3dv/team22/RGBTrack/logs/job_vis_depth_vggt_da3_%j.out
#SBATCH --error=/work/courses/3dv/team22/RGBTrack/logs/job_vis_depth_vggt_da3_%j.err

export PATH=/work/courses/3dv/team22/py310_env/bin:$PATH

cd /work/courses/3dv/team22/RGBTrack
mkdir -p logs

SCENE=/work/courses/3dv/team22/foundationpose/data/20250804_104715
DEBUG=/work/courses/3dv/team22/foundationpose/debug

/work/courses/3dv/team22/py310_env/bin/python visualize_depth.py \
    --source custom \
    --depth_dir $SCENE/depth_vggt \
    --label vggt \
    --test_scene_dir $SCENE \
    --out_video $DEBUG/depth_vggt.mp4

/work/courses/3dv/team22/py310_env/bin/python visualize_depth.py \
    --source custom \
    --depth_dir $SCENE/depth_da3 \
    --label da3 \
    --test_scene_dir $SCENE \
    --out_video $DEBUG/depth_da3.mp4
