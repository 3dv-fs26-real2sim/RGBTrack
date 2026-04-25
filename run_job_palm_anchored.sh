#!/bin/bash
#SBATCH --account=3dv
#SBATCH --time=06:00:00
#SBATCH --output=/work/courses/3dv/team22/RGBTrack/logs/job_palm_anchored_%j.out
#SBATCH --error=/work/courses/3dv/team22/RGBTrack/logs/job_palm_anchored_%j.err

. /etc/profile.d/modules.sh
module load cuda/12.8

export PATH=/work/courses/3dv/team22/py310_env/bin:$PATH
export CUDA_HOME=/cluster/data/cuda/12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/work/courses/3dv/team22/py310_env/lib/python3.10/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/work/courses/3dv/team22/py310_env/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /work/courses/3dv/team22/RGBTrack
mkdir -p logs

/work/courses/3dv/team22/py310_env/bin/python run_demo_palm_anchored.py \
    --mesh_file /work/courses/3dv/team22/foundationpose/data/object/duck/duck.obj \
    --test_scene_dir /work/courses/3dv/team22/foundationpose/data/20250804_104715 \
    --palm_poses_npz /work/courses/3dv/team22/RGBTrack/palm_poses_20250804_104715_right_from-palm_frame-aria_o0_3_0.npz \
    --debug_dir /work/courses/3dv/team22/foundationpose/debug/duck_palm_anchored \
    --grasp_start_frame 250 \
    --grasp_end_frame 450 \
    --est_refine_iter 2 \
    --track_refine_iter 2 \
    --debug 2
