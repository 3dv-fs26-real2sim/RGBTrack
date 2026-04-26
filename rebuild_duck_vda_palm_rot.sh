#!/bin/bash
# Recreate duck_vda_palm_rot from scratch in two stages:
#   Stage A: run_demo_vda_hand.py  (FP++ on VDA depth, ScoreNet disabled)
#            -> /work/courses/3dv/team22/foundationpose/debug/duck_vda_hand
#   Stage B: postprocess_palm_rotation.py
#            -> /work/courses/3dv/team22/foundationpose/debug/duck_vda_palm_rot
#
# Inputs:
#   - mesh:        duck.obj
#   - masks:       <scene>/masks/  (original SAM2 masks, NOT cad_anchored)
#   - depth:       <scene>/depth/  (VDA depth PNGs)
#   - palm_poses:  RGBTrack/palm_poses_*.npz  (MediaPipe-derived T_cam_palm)
#
# Grasp window: 220–450 (matches the original duck_vda_palm_rot defaults)

#SBATCH --account=3dv
#SBATCH --time=06:00:00
#SBATCH --output=/work/courses/3dv/team22/RGBTrack/logs/job_rebuild_palm_rot_%j.out
#SBATCH --error=/work/courses/3dv/team22/RGBTrack/logs/job_rebuild_palm_rot_%j.err

. /etc/profile.d/modules.sh
module load cuda/12.8
export PATH=/work/courses/3dv/team22/py310_env/bin:$PATH
export CUDA_HOME=/cluster/data/cuda/12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/work/courses/3dv/team22/py310_env/lib/python3.10/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/work/courses/3dv/team22/py310_env/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCENE=/work/courses/3dv/team22/foundationpose/data/20250804_104715
MESH=/work/courses/3dv/team22/foundationpose/data/object/duck/duck.obj
PALM_NPZ=/work/courses/3dv/team22/RGBTrack/palm_poses_20250804_104715_right_from-palm_frame-aria_o0_3_0.npz
HAND_DIR=/work/courses/3dv/team22/foundationpose/debug/duck_vda_hand
PALM_ROT_DIR=/work/courses/3dv/team22/foundationpose/debug/duck_vda_palm_rot

cd /work/courses/3dv/team22/RGBTrack
mkdir -p logs

echo "=== Stage A: run_demo_vda_hand.py (FP++ tracking, no ScoreNet) ==="
/work/courses/3dv/team22/py310_env/bin/python run_demo_vda_hand.py \
    --mesh_file       $MESH \
    --test_scene_dir  $SCENE \
    --debug_dir       $HAND_DIR \
    --est_refine_iter 2 \
    --track_refine_iter 2 \
    --debug 2

echo "=== Stage B: postprocess_palm_rotation.py ==="
/work/courses/3dv/team22/py310_env/bin/python postprocess_palm_rotation.py \
    --mesh_file        $MESH \
    --test_scene_dir   $SCENE \
    --fp_debug_dir     $HAND_DIR \
    --out_debug_dir    $PALM_ROT_DIR \
    --palm_poses_npz   $PALM_NPZ \
    --grasp_start_frame 220 \
    --grasp_end_frame   450

echo "=== Done. Output at $PALM_ROT_DIR ==="
