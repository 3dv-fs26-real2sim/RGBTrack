#!/bin/bash
#SBATCH --account=3dv
#SBATCH --time=01:00:00
#SBATCH --output=/work/courses/3dv/team22/RGBTrack/logs/job_vis_depth_bg11_%j.out
#SBATCH --error=/work/courses/3dv/team22/RGBTrack/logs/job_vis_depth_bg11_%j.err

. /etc/profile.d/modules.sh
module load cuda/12.8
export PATH=/work/courses/3dv/team22/py310_env/bin:$PATH
export CUDA_HOME=/cluster/data/cuda/12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/work/courses/3dv/team22/py310_env/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /work/courses/3dv/team22/RGBTrack
mkdir -p logs

/work/courses/3dv/team22/py310_env/bin/python visualize_depth.py \
    --source         custom \
    --depth_dir      /work/courses/3dv/team22/foundationpose/data/20250804_104715/depth_vda_bg11 \
    --test_scene_dir /work/courses/3dv/team22/foundationpose/data/20250804_104715 \
    --mesh_file      /work/courses/3dv/team22/foundationpose/data/object/duck/duck.obj \
    --calibrate \
    --no_table_cutoff \
    --fps            50 \
    --out_video      /work/courses/3dv/team22/foundationpose/debug/depth_vda_bg11_vis.mp4
