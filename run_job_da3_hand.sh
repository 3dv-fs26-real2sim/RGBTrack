#!/bin/bash
#SBATCH --account=3dv
#SBATCH --time=06:00:00
#SBATCH --output=/work/courses/3dv/team22/RGBTrack/logs/job_da3_hand_%j.out
#SBATCH --error=/work/courses/3dv/team22/RGBTrack/logs/job_da3_hand_%j.err

. /etc/profile.d/modules.sh
module load cuda/12.8

export CUDA_HOME=/cluster/data/cuda/12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /work/courses/3dv/team22/RGBTrack
mkdir -p logs

# ── Step 1: generate DA3 depth PNGs via Video Depth Anything streaming ────────
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /work/scratch/$USER/conda/envs/vda

python generate_vda_streaming_maps.py \
    --scene_dir /work/courses/3dv/team22/foundationpose/data/20250804_104715 \
    --vda_repo  /home/jaerkim/Video-Depth-Anything \
    --out_dir   /work/courses/3dv/team22/foundationpose/data/20250804_104715/depth_da3

conda deactivate

# ── Step 2: run tracking with freshly generated DA3 depth ─────────────────────
export PATH=/work/courses/3dv/team22/py310_env/bin:$PATH
export LD_LIBRARY_PATH=/work/courses/3dv/team22/py310_env/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH

/work/courses/3dv/team22/py310_env/bin/python run_demo_vda_hand.py \
    --mesh_file /work/courses/3dv/team22/foundationpose/data/object/duck/duck.obj \
    --test_scene_dir /work/courses/3dv/team22/foundationpose/data/20250804_104715 \
    --debug_dir /work/courses/3dv/team22/foundationpose/debug/duck_da3_hand \
    --est_refine_iter 2 \
    --track_refine_iter 2 \
    --debug 2 \
    --depth_dir /work/courses/3dv/team22/foundationpose/data/20250804_104715/depth_da3
