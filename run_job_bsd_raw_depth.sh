#!/bin/bash
#SBATCH --account=3dv
#SBATCH --time=06:00:00
#SBATCH --output=/work/courses/3dv/team22/RGBTrack/logs/job_bsd_raw_depth_%j.out
#SBATCH --error=/work/courses/3dv/team22/RGBTrack/logs/job_bsd_raw_depth_%j.err

# Pure FP tracking with BSD init + BSD-scaled depth.
# Reproduces the Apr 14 (f811b0f) "tracking_bsd" pipeline; the original used
# a colleague-scaled depth set at scaled_with_desk/depth_png. Since that was
# cleaned, we scale Depth Pro on the fly using BSD on frame 0 (same anchor
# the colleague used).

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

SCENE=/work/courses/3dv/team22/foundationpose/data/20250804_104715
SCALED_DEPTH=/work/scratch/hudela/depth_pro_scaled_104715
MESH=/work/courses/3dv/team22/foundationpose/data/object/duck/duck.obj

# ── Step 1: BSD-scale Depth Pro depth (one-time, ~1 min) ─────────────────────
N_SCALED=$(ls $SCALED_DEPTH/*.png 2>/dev/null | wc -l)
N_RGB=$(ls $SCENE/rgb/*.png 2>/dev/null | wc -l)
if [ "$N_SCALED" -eq "$N_RGB" ] && [ "$N_SCALED" -gt 0 ]; then
    echo "[scale] $SCALED_DEPTH already complete ($N_SCALED), skipping"
else
    echo "[scale] BSD-scaling depth_pro → $SCALED_DEPTH"
    /work/courses/3dv/team22/py310_env/bin/python scale_depth_via_bsd.py \
        --depth_dir $SCENE/depth_pro \
        --rgb       $SCENE/rgb/000000.png \
        --mask      $SCENE/masks/000000.png \
        --cam_K     $SCENE/cam_K.txt \
        --mesh      $MESH \
        --out_dir   $SCALED_DEPTH
fi

# ── Step 2: bsd_raw_depth FP tracking on the scaled depth ────────────────────
/work/courses/3dv/team22/py310_env/bin/python run_demo_bsd_raw_depth.py \
    --mesh_file      $MESH \
    --test_scene_dir $SCENE \
    --pred_depth_dir $SCALED_DEPTH \
    --debug_dir      /work/courses/3dv/team22/foundationpose/debug/duck_bsd_raw \
    --est_refine_iter 2 \
    --track_refine_iter 2 \
    --debug 2
