#!/bin/bash
#SBATCH --account=3dv
#SBATCH --time=01:00:00
#SBATCH --output=/work/courses/3dv/team22/RGBTrack/logs/job_bsd_raw_depth_%j.out
#SBATCH --error=/work/courses/3dv/team22/RGBTrack/logs/job_bsd_raw_depth_%j.err

# Pure-FP tracking with BSD init + raw (pre-scaled) depth.
# Same recipe that produced duck_bsd_raw (the "tracking_bsd" video, Apr 14).
#
# Usage:
#   sbatch run_job_bsd_raw_depth.sh \
#       [--scene_dir <DIR>]   default: data/20250804_104715
#       [--pred_depth_dir <DIR>]  default: <scene_dir>/depth_pro
#       [--debug_dir <DIR>]   default: foundationpose/debug/duck_bsd_raw
#       [--mesh <obj>]        default: duck.obj
#       [--est_refine_iter N] [--track_refine_iter N]

SCENE_DIR=/work/courses/3dv/team22/foundationpose/data/20250804_104715
DEPTH_DIR=""
DEBUG_DIR=/work/courses/3dv/team22/foundationpose/debug/duck_bsd_raw
MESH=/work/courses/3dv/team22/foundationpose/data/object/duck/duck.obj
EST_ITER=2
TRACK_ITER=2

while [[ $# -gt 0 ]]; do
    case $1 in
        --scene_dir)        SCENE_DIR=$2;  shift 2 ;;
        --pred_depth_dir)   DEPTH_DIR=$2;  shift 2 ;;
        --debug_dir)        DEBUG_DIR=$2;  shift 2 ;;
        --mesh)             MESH=$2;       shift 2 ;;
        --est_refine_iter)  EST_ITER=$2;   shift 2 ;;
        --track_refine_iter) TRACK_ITER=$2; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done
[ -z "$DEPTH_DIR" ] && DEPTH_DIR=$SCENE_DIR/depth_pro

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
git pull
mkdir -p logs

echo "=================================================="
echo " BSD raw-depth FP tracking"
echo "  scene_dir       = $SCENE_DIR"
echo "  pred_depth_dir  = $DEPTH_DIR"
echo "  debug_dir       = $DEBUG_DIR"
echo "  iters           = est=$EST_ITER  track=$TRACK_ITER"
echo "=================================================="

/work/courses/3dv/team22/py310_env/bin/python run_demo_bsd_raw_depth.py \
    --mesh_file       $MESH \
    --test_scene_dir  $SCENE_DIR \
    --pred_depth_dir  $DEPTH_DIR \
    --debug_dir       $DEBUG_DIR \
    --est_refine_iter $EST_ITER \
    --track_refine_iter $TRACK_ITER \
    --debug 2

echo "=================================================="
echo " bsd_raw_depth DONE → $DEBUG_DIR"
echo "=================================================="
