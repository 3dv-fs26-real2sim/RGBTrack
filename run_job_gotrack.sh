#!/bin/bash
#SBATCH --account=3dv
#SBATCH --partition=jobs
#SBATCH --nodelist=studgpu-node05            # 2080 Ti (gotrack env is cu11, no Blackwell support)
#SBATCH --time=00:30:00
#SBATCH --output=/work/courses/3dv/team22/RGBTrack/logs/gotrack_batch_%j.out
#SBATCH --error=/work/courses/3dv/team22/RGBTrack/logs/gotrack_batch_%j.err

# Batch-refine an FP pose stream with GoTrack.
#
# Prereqs (one-time setup, see RGBTrack/GOTRACK_INTEGRATION_PLAN.md):
#   - gotrack repo cloned at /work/courses/3dv/team22/gotrack
#   - gotrack conda env at /work/courses/3dv/team22/gotrack_env
#     (cu11 build of pytorch/xformers — only works on 2080 Ti, not 5060 Ti)
#   - gotrack_checkpoint.pt downloaded into /work/courses/3dv/team22/gotrack/
#
# Usage:
#   sbatch run_job_gotrack.sh <SCENE_NAME> [start] [end] [n_iter] [rotation_only]
#
# Examples:
#   # Full sequence, full 6DoF refinement, 3 iters
#   sbatch run_job_gotrack.sh 20250804_104715_full 0 -1 3
#
#   # Frames 200-400 only, rotation only (keep init translation)
#   sbatch run_job_gotrack.sh 20250804_104715_full 200 400 3 rotation_only
#
# Inputs expected at /work/scratch/hudela/<SCENE>/:
#   rgb/             <frame>.png
#   fp/ob_in_cam/    <frame>.txt   (4×4 cam_from_model init poses)
#   cam_K.txt
#
# Outputs at /work/scratch/hudela/<SCENE>/fp_gotrack/:
#   ob_in_cam/   refined 4×4 poses
#   track_vis/   green=init, red=refined bbox overlay
#
# Runtime: ~0.3 s/frame on 2080 Ti (≈6 min for 1199 frames).

SCENE=${1:?need scene name, e.g. 20250804_104715_full}
START=${2:-0}
END=${3:--1}                 # -1 = all frames
N_ITER=${4:-3}
ROT_ONLY=${5:-no}            # pass 'rotation_only' to keep init translation

. /work/courses/3dv/team22/miniconda3/bin/activate /work/courses/3dv/team22/gotrack_env

cd /work/courses/3dv/team22/RGBTrack
git pull

# EGL via NVIDIA driver for headless offscreen rendering.
export PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

ROT_FLAG=""
if [ "$ROT_ONLY" = "rotation_only" ]; then ROT_FLAG="--rotation_only"; fi

# Clean previous output so we don't mix runs
rm -rf /work/scratch/hudela/$SCENE/fp_gotrack

python run_gotrack_batch.py \
    --scene_dir /work/scratch/hudela/$SCENE \
    --ckpt /work/courses/3dv/team22/gotrack/gotrack_checkpoint.pt \
    --mesh /work/courses/3dv/team22/foundationpose/data/object/duck/duck.obj \
    --start $START \
    --end $END \
    --n_iter $N_ITER \
    $ROT_FLAG
